"""Embedding-layer integrity check, :class:`EmbeddingChecksum`.

Same pattern as :class:`AttentionChecksum`: Frobenius-norm fingerprint
recorded at hardening time, recomputed and compared on every forward.
Detection-only in v0.1; recovery (via a stored replica or per-row TMR)
lands in v0.5, see the protection module docs, which
flags the embedding/EOS row as a major attack surface (PrisonBreak).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from spacellm.protection.base import ProtectionReport, Strategy


class EmbeddingChecksum(Strategy):
    """Detect bit-flip corruption in :class:`nn.Embedding` modules.

    On :meth:`apply` the strategy walks the module tree, identifies every
    :class:`nn.Embedding`, records a per-parameter Frobenius-norm fingerprint
    of its current weights, and registers a forward pre-hook. Each forward
    recomputes the fingerprint and compares against the trusted reference;
    mismatches are surfaced via the ``ProtectionReport``.

    Args:
        rtol: Relative tolerance for fingerprint comparison.
        atol: Absolute tolerance for fingerprint comparison.

    Notes
    -----
    Detection-only, recovery is reserved for v0.5 once a replica strategy
    can supply a clean copy.
    """

    name = "embedding_checksum"

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
            if not isinstance(module, nn.Embedding):
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
        def hook(module: nn.Module, _args: tuple[object, ...]) -> None:
            self._n_checks += 1
            params = dict(module.named_parameters(recurse=True))
            for p_name, expected in self._fingerprints[path].items():
                p = params.get(p_name)
                if p is None:  # pragma: no cover, unreachable unless graph mutates
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


__all__ = ["EmbeddingChecksum"]
