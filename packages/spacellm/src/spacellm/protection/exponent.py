"""Exponent-flip range guard, a low-overhead alternative to TMR.

The damaging bit-flips are in the exponent / high MSBs of a float weight
(the bit-30 chart in the README; FlipGuard EMNLP'25 and Global Clipper
IJCAI'24 target the same bits). An exponent flip multiplies a weight by a
power of two, sending it to ~10³⁸ or NaN; a mantissa-LSB flip shifts it by
< 1 % and is individually harmless.

That asymmetry permits a defense far cheaper than triple-modular
redundancy. At hardening time, record the trusted ``[min, max]`` value
range of each protected weight tensor, two scalars rather than a 2× replica.
On every forward, clamp the weight back into that range and replace any
non-finite element with the trusted bound. An exponent flip lands outside
``[min, max]`` (or becomes NaN) and is neutralised; a mantissa flip stays
in range and passes through untouched.

This is detection-plus-clamp, not exact restoration: a clamped element
returns to the trusted boundary, not its original value. For bit-exact
recovery use :class:`~spacellm.protection.tmr.SelectiveTMR` or
:class:`~spacellm.protection.abft.ABFTProtection`; this strategy trades a
hair of precision for an overhead under 0.001 % of weight memory.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from spacellm.protection._module_paths import walk_named_submodules
from spacellm.protection.base import ProtectionReport, Strategy


class ExponentRangeGuard(Strategy):
    """Clamp protected weights to their trusted value range on every forward.

    Args:
        top_k_percent: Percentage of eligible modules to protect, in
            ``(0, 100]``. Defaults to 100, the guard is cheap enough to
            apply everywhere.
        target_class: Module class to protect. Defaults to
            :class:`torch.nn.Linear`.
        margin: Fractional slack added to the trusted range so benign
            numerical drift is not clamped, e.g. ``0.0`` clamps to the
            exact observed range; ``0.01`` widens it by 1 %. Defaults to
            ``0.0``.

    Raises:
        ValueError: If ``top_k_percent`` is outside ``(0, 100]`` or
            ``margin`` is negative.
    """

    name = "exponent_range_guard"

    def __init__(
        self,
        top_k_percent: float = 100.0,
        *,
        target_class: type[nn.Module] = nn.Linear,
        margin: float = 0.0,
    ) -> None:
        if not 0.0 < top_k_percent <= 100.0:
            raise ValueError(f"top_k_percent must be in (0, 100], got {top_k_percent}")
        if margin < 0.0:
            raise ValueError(f"margin must be non-negative, got {margin}")
        self.top_k_percent = float(top_k_percent)
        self.target_class = target_class
        self.margin = float(margin)
        self._bounds: dict[str, tuple[float, float]] = {}
        self._handles: list[RemovableHandle] = []
        self._owned_modules: list[str] = []
        self._n_checks = 0
        self._n_detected = 0
        self._n_recovered = 0

    def apply(self, model: nn.Module) -> nn.Module:
        candidates: list[tuple[str, nn.Module, int]] = []
        for path, module in walk_named_submodules(model):
            if isinstance(module, self.target_class):
                params = [p for p in module.parameters(recurse=False) if p.numel() > 0]
                if params:
                    size = sum(p.numel() for p in params)
                    candidates.append((path, module, size))
        if not candidates:
            return model

        candidates.sort(key=lambda item: -item[2])
        n_select = max(1, round(len(candidates) * self.top_k_percent / 100.0))

        for path, module, _ in candidates[:n_select]:
            for p_name, param in module.named_parameters(recurse=False):
                if param.numel() == 0:
                    continue
                key = f"{path}.{p_name}" if path else p_name
                lo = float(param.detach().min().item())
                hi = float(param.detach().max().item())
                span = hi - lo
                lo -= self.margin * abs(span)
                hi += self.margin * abs(span)
                self._bounds[key] = (lo, hi)
            self._owned_modules.append(path)
            handle = module.register_forward_pre_hook(self._make_hook(path))
            self._handles.append(handle)
        return model

    def _make_hook(
        self,
        path: str,
    ) -> Callable[[nn.Module, tuple[object, ...]], None]:
        def hook(module: nn.Module, _args: tuple[object, ...]) -> None:
            with torch.no_grad():
                for p_name, param in module.named_parameters(recurse=False):
                    key = f"{path}.{p_name}" if path else p_name
                    bounds = self._bounds.get(key)
                    if bounds is None:
                        continue
                    self._n_checks += 1
                    lo, hi = bounds
                    data = param.data
                    finite = torch.isfinite(data)
                    out_of_range = (~finite) | (data < lo) | (data > hi)
                    n_bad = int(out_of_range.sum().item())
                    if n_bad == 0:
                        continue
                    self._n_detected += n_bad
                    # Replace non-finite with the upper bound, then clamp.
                    if not bool(finite.all().item()):
                        data[~finite] = hi
                    param.data = data.clamp_(min=lo, max=hi)
                    self._n_recovered += n_bad

        return hook

    def collect_report(self) -> ProtectionReport:
        return ProtectionReport(
            strategy_name=self.name,
            n_corruptions_detected=self._n_detected,
            n_corruptions_recovered=self._n_recovered,
            n_checks_performed=self._n_checks,
            overhead_bytes=len(self._bounds) * 2 * 8,  # two float64 scalars per tensor
            extra={
                "n_protected_modules": float(len(self._owned_modules)),
                "top_k_percent": self.top_k_percent,
            },
        )

    def reset(self) -> None:
        self._n_checks = 0
        self._n_detected = 0
        self._n_recovered = 0

    def detach(self) -> None:
        """Remove forward hooks. Idempotent."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    @property
    def protected_modules(self) -> tuple[str, ...]:
        """Dotted paths of guarded modules. Useful for tests."""
        return tuple(self._owned_modules)


__all__ = ["ExponentRangeGuard"]
