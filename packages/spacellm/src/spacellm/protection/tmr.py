"""Selective triple-modular-redundancy strategy."""

from __future__ import annotations

from torch import nn

from spacellm.nn.tmr import TMRLinear
from spacellm.protection._module_paths import set_submodule, walk_named_submodules
from spacellm.protection.base import ProtectionReport, Strategy


class SelectiveTMR(Strategy):
    """Apply triple-modular redundancy to the largest ``top_k_percent`` of layers.

    Sensitivity ranking in v0.1 is *parameter count*: bigger layers carry
    more bit-area and therefore more SEU surface. v0.5 will swap this for
    a real sensitivity profile (per-layer accuracy degradation under
    fault injection, see ``spacellm.profiling``).

    Args:
        top_k_percent: Percentage of eligible modules to protect, in
            ``(0, 100]``. Defaults to 5, protecting the top 5% by size
            typically captures > 50% of total parameters in transformer
            architectures.
        target_class: Module class to consider for protection. Defaults
            to :class:`torch.nn.Linear`. v0.5 will add
            :class:`torch.nn.Embedding` and the LayerNorm family.

    Raises:
        ValueError: If ``top_k_percent`` is outside ``(0, 100]``.
    """

    name = "selective_tmr"

    def __init__(
        self,
        top_k_percent: float = 5.0,
        *,
        target_class: type[nn.Module] = nn.Linear,
    ) -> None:
        if not 0.0 < top_k_percent <= 100.0:
            raise ValueError(
                f"top_k_percent must be in (0, 100], got {top_k_percent}",
            )
        self.top_k_percent = float(top_k_percent)
        self.target_class = target_class
        self._wrapped_paths: list[str] = []
        self._overhead_bytes: int = 0

    def apply(self, model: nn.Module) -> nn.Module:
        candidates: list[tuple[str, nn.Module, int]] = []
        for path, module in walk_named_submodules(model):
            if isinstance(module, self.target_class) and not isinstance(module, TMRLinear):
                size = sum(p.numel() for p in module.parameters(recurse=False))
                candidates.append((path, module, size))

        if not candidates:
            return model

        candidates.sort(key=lambda item: -item[2])
        n_select = max(1, round(len(candidates) * self.top_k_percent / 100.0))
        selected = candidates[:n_select]

        for path, module, _ in selected:
            if not isinstance(module, nn.Linear):
                # v0.1 only handles nn.Linear inside this strategy. Other
                # target_class values are reserved for v0.5+ and will raise.
                raise NotImplementedError(
                    f"SelectiveTMR currently supports nn.Linear only "
                    f"(got {type(module).__name__} at '{path}')",
                )
            wrapped = TMRLinear(module)
            set_submodule(model, path, wrapped)
            self._wrapped_paths.append(path)
            self._overhead_bytes += sum(
                p.element_size() * p.numel() for p in wrapped.parameters() if p is not None
            ) - sum(p.element_size() * p.numel() for p in module.parameters())

        return model

    def collect_report(self) -> ProtectionReport:
        return ProtectionReport(
            strategy_name=self.name,
            n_corruptions_detected=0,
            n_corruptions_recovered=0,
            n_checks_performed=0,
            overhead_bytes=self._overhead_bytes,
            extra={
                "n_wrapped_modules": float(len(self._wrapped_paths)),
                "top_k_percent": self.top_k_percent,
            },
        )

    @property
    def wrapped_paths(self) -> tuple[str, ...]:
        """Tuple of dotted paths that were wrapped with TMR. Useful for tests."""
        return tuple(self._wrapped_paths)


__all__ = ["SelectiveTMR"]
