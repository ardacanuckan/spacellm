"""Attention-layer integrity check.

This is the v0.1 take on attention-layer ABFT. It is *detection-only*:
when a per-parameter Frobenius-norm fingerprint diverges from its trusted
reference, we record a detection event. Recovery (e.g. ATTNChecker-style
recompute, or restore-from-replica) lands in v0.5.

Why Frobenius norm?
-------------------
A bit-flip in an exponent bit of a float scalar shifts the value's
magnitude by a power of two, these are the flips that do the
most damage to model accuracy (per the protection module docs
§4, PrisonBreak finds 94 % of high-impact attack bits live in the top
three exponent bits). Such a flip changes ``‖W‖_F`` enough to be
detected with a tight numerical tolerance. Mantissa-bit flips are not
always caught by this check, but they are also not dangerous individually
, a fact the v0.5 successor strategy will exploit by combining this with
a row-sum ABFT detector.

Cost
----
* One ``(rtol, atol)`` close-comparison per parameter per protected forward
  call. The overhead is one float scalar per parameter, < 0.001 % of
  weight memory.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from spacellm.protection.base import ProtectionReport, Strategy


def _looks_like_attention(name: str, module: nn.Module) -> bool:
    """Return ``True`` if ``module`` is plausibly an attention layer."""
    if isinstance(module, nn.MultiheadAttention):
        return True
    cls_lower = type(module).__name__.lower()
    name_lower = name.lower()
    if "attention" in cls_lower or "attn" in cls_lower:
        return True
    return ("attention" in name_lower) or (".attn" in name_lower)


class AttentionChecksum(Strategy):
    """Detect corruption in attention-layer parameters via norm fingerprints.

    Args:
        rtol: Relative tolerance for fingerprint comparison.
        atol: Absolute tolerance for fingerprint comparison.

    Notes:
        ``apply`` registers a forward pre-hook on every detected attention
        module. After ``apply``, ``model.eval()`` and ``model(x)`` continue
        to work unchanged, the only side effect is that detection
        counters update on each forward.
    """

    name = "attention_checksum"

    def __init__(self, *, rtol: float = 1e-5, atol: float = 1e-8) -> None:
        self.rtol = rtol
        self.atol = atol
        self._fingerprints: dict[str, dict[str, torch.Tensor]] = {}
        self._handles: list[RemovableHandle] = []
        self._n_checks: int = 0
        self._n_detected: int = 0
        self._owned_modules: list[str] = []

    def apply(self, model: nn.Module) -> nn.Module:
        for name, module in list(model.named_modules()):
            if not _looks_like_attention(name, module):
                continue
            params = dict(module.named_parameters(recurse=True))
            if not params:
                continue
            self._fingerprints[name] = {
                p_name: p.detach().norm().clone() for p_name, p in params.items()
            }
            self._owned_modules.append(name)
            self._handles.append(module.register_forward_pre_hook(self._make_hook(name)))
        return model

    def _make_hook(
        self,
        path: str,
    ) -> Callable[[nn.Module, tuple[object, ...]], None]:
        def hook(
            module: nn.Module,
            _args: tuple[object, ...],
        ) -> None:
            self._n_checks += 1
            params = dict(module.named_parameters(recurse=True))
            for p_name, expected in self._fingerprints[path].items():
                p = params.get(p_name)
                if p is None:  # pragma: no cover, unreachable unless the model graph mutates
                    continue
                actual = p.detach().norm()
                if not torch.isclose(actual, expected, rtol=self.rtol, atol=self.atol):
                    self._n_detected += 1

        return hook

    def collect_report(self) -> ProtectionReport:
        return ProtectionReport(
            strategy_name=self.name,
            n_corruptions_detected=self._n_detected,
            n_corruptions_recovered=0,
            n_checks_performed=self._n_checks,
            overhead_bytes=sum(
                fp.element_size() * fp.numel()
                for fps in self._fingerprints.values()
                for fp in fps.values()
            ),
            extra={
                "n_protected_modules": float(len(self._owned_modules)),
            },
        )

    def reset(self) -> None:
        self._n_checks = 0
        self._n_detected = 0

    def detach(self) -> None:
        """Remove forward hooks. Idempotent."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()


__all__ = ["AttentionChecksum"]
