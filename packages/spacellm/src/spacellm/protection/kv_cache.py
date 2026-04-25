"""KV-cache parity protection.

The HuggingFace decoder cache is a per-layer tuple of ``(key, value)``
tensors that grows by one row per token. A bit-flip in cached state
silently distorts long-context inference because the corrupted key
participates in *every* subsequent attention step. There is no published
defense for this in the LLM literature, see
the protection module docs.

This strategy maintains a per-row XOR-parity word over each guarded KV
tensor. On verification, the parity is recomputed and compared. Single-bit
flips on a row are detected; the row index is recoverable from the
mismatching parity entry.

API surface
-----------
``KVCacheParity`` is a *primitive* strategy: ``apply()`` is a no-op
because PyTorch's :class:`torch.nn.Module` does not expose decoder caches
in a uniform way. Users (and the higher-level decode loop in v0.5+) wire
``guard()`` and ``verify()`` calls at cache write/read points by name.
"""

from __future__ import annotations

import torch
from torch import nn

from spacellm.protection.base import ProtectionReport, Strategy

_INT_FOR_SIZE: dict[int, torch.dtype] = {
    1: torch.int8,
    2: torch.int16,
    4: torch.int32,
    8: torch.int64,
}


def _row_xor_parity(t: torch.Tensor) -> torch.Tensor:
    """XOR-reduce a tensor along its last dimension as bit patterns.

    Returns a tensor with one fewer dimension, of the matching-size
    integer dtype.
    """
    if t.is_complex():
        raise TypeError(
            f"Complex tensors are not supported (got {t.dtype})",
        )
    bytes_per_element = t.element_size()
    int_dtype = _INT_FOR_SIZE.get(bytes_per_element)
    if int_dtype is None:
        raise TypeError(f"Unsupported element size {bytes_per_element}")
    int_view = t.contiguous().view(int_dtype)
    if int_view.ndim == 0:
        return int_view.clone()
    if int_view.ndim == 1:
        # No row axis, compute global parity scalar.
        result = torch.zeros((), dtype=int_dtype, device=t.device)
        for i in range(int_view.shape[0]):
            result = result ^ int_view[i]
        return result
    n_last = int_view.shape[-1]
    result = int_view.select(-1, 0).clone()
    for i in range(1, n_last):
        result = result ^ int_view.select(-1, i)
    return result


class KVCacheParity(Strategy):
    """Per-row XOR-parity guard for KV-cache tensors.

    Memory overhead per guarded tensor: one int word per row (i.e. shape
    matches the input minus the last dim, with the matching-size int
    dtype). For an FP16 cache of shape ``(batch, heads, seq, head_dim)``
    the overhead is ``batch × heads × seq × 2 bytes``, typically < 1 %
    of the cache itself.
    """

    name = "kv_cache_parity"

    def __init__(self) -> None:
        self._parities: dict[str, torch.Tensor] = {}
        self._n_checks: int = 0
        self._n_detected: int = 0

    def apply(self, model: nn.Module) -> nn.Module:
        # KV-cache lives outside the nn.Module graph. apply() is a no-op
        # by design; the user (or v0.5+ runtime) wires guard/verify into
        # their decode loop.
        return model

    def guard(self, name: str, key: torch.Tensor, value: torch.Tensor) -> None:
        """Record the parity of a (key, value) pair at the given name.

        ``name`` is used to associate later ``verify()`` calls with this
        record. Convention: ``"layer{i}.k"`` / ``"layer{i}.v"``.
        """
        self._parities[f"{name}.k"] = _row_xor_parity(key).clone()
        self._parities[f"{name}.v"] = _row_xor_parity(value).clone()

    def verify(self, name: str, key: torch.Tensor, value: torch.Tensor) -> int:
        """Verify a previously-guarded ``(key, value)`` pair.

        Returns:
            Number of mismatched tensors (0, 1, or 2).
        """
        return self._verify_one(f"{name}.k", key) + self._verify_one(f"{name}.v", value)

    def _verify_one(self, key: str, t: torch.Tensor) -> int:
        self._n_checks += 1
        expected = self._parities.get(key)
        if expected is None:
            return 0
        actual = _row_xor_parity(t)
        if torch.equal(actual, expected):
            return 0
        self._n_detected += 1
        return 1

    def forget(self, name: str) -> None:
        """Discard parity records for ``name`` (e.g. when a session ends)."""
        self._parities.pop(f"{name}.k", None)
        self._parities.pop(f"{name}.v", None)

    def collect_report(self) -> ProtectionReport:
        return ProtectionReport(
            strategy_name=self.name,
            n_corruptions_detected=self._n_detected,
            n_corruptions_recovered=0,
            n_checks_performed=self._n_checks,
            overhead_bytes=sum(p.element_size() * p.numel() for p in self._parities.values()),
            extra={
                "n_guarded_pairs": float(len(self._parities) // 2),
            },
        )

    def reset(self) -> None:
        self._n_checks = 0
        self._n_detected = 0


__all__ = ["KVCacheParity"]
