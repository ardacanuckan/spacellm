"""LayerNorm / RMSNorm TMR strategy."""

from __future__ import annotations

from torch import nn

from spacellm.nn.layer_norm import TMRLayerNorm, _is_rmsnorm
from spacellm.protection._module_paths import set_submodule, walk_named_submodules
from spacellm.protection.base import ProtectionReport, Strategy


class LayerNormTMR(Strategy):
    """Triple-modular redundancy for every LayerNorm and RMSNorm in a model.

    Norms are tiny in parameter count but disproportionately critical ,
    a corrupted γ scales every activation through the layer. Wrapping
    them is one of the cheapest high-impact defences (FT-Transformer
    PPoPP'25: 13.9 % memory overhead, 97.2 % detection;
    the protection module docs).

    Args:
        include_rmsnorm: If ``True`` (default), :class:`nn.RMSNorm`
            modules are wrapped alongside :class:`nn.LayerNorm`. RMSNorm
            requires PyTorch 2.4+; older versions silently skip.
    """

    name = "layernorm_tmr"

    def __init__(self, *, include_rmsnorm: bool = True) -> None:
        self.include_rmsnorm = include_rmsnorm
        self._wrapped_paths: list[str] = []
        self._overhead_bytes: int = 0

    def apply(self, model: nn.Module) -> nn.Module:
        for path, module in walk_named_submodules(model):
            if isinstance(module, TMRLayerNorm):
                continue
            is_layernorm = isinstance(module, nn.LayerNorm)
            is_rmsnorm = self.include_rmsnorm and _is_rmsnorm(module)
            if not (is_layernorm or is_rmsnorm):
                continue
            if getattr(module, "weight", None) is None:
                # No learnable affine; nothing to TMR.
                continue
            wrapped = TMRLayerNorm(module)
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
                "include_rmsnorm": float(self.include_rmsnorm),
            },
        )

    @property
    def wrapped_paths(self) -> tuple[str, ...]:
        """Tuple of dotted paths that were wrapped. Useful for tests."""
        return tuple(self._wrapped_paths)


__all__ = ["LayerNormTMR"]
