"""Algorithm-Based Fault Tolerance (ABFT) linear layer.

Where :class:`~spacellm.nn.tmr.TMRLinear` protects a weight by storing it
three times, ABFT protects the *matrix multiply itself* with a checksum
that costs a vector, not a 2× copy. This is the family of techniques used
by ALBERTA (NVIDIA, arXiv:2310.03841, < 0.2 % runtime, < 0.01 % memory,
> 99 % coverage) and ATTNChecker (PPoPP'25): augment the GEMM with row /
column checksums and verify the result against an independently-computed
invariant.

Two invariants are maintained, both derived from the trusted weight at
construction time:

1. **Weight integrity (row + column checksum).** ``row[o] = Σ_i W[o, i]``
   and ``col[i] = Σ_o W[o, i]``. A *single* corrupted element
   ``W[o*, i*]`` shifts exactly one row sum and exactly one column sum by
   the same δ; that pair localises the error and ``δ`` corrects it. This
   is classical single-error ABFT correction, and it recovers the
   persistent weight flips this framework injects, at the cost of two
   checksum vectors instead of two full replicas.

2. **Compute-path checksum.** For ``y = x Wᵀ + b`` the column sum of the
   output obeys ``Σ_o y[..., o] = x · col + Σ_o b[o]``. The right-hand
   side is recomputed independently with an ``O(in_features)`` mat-vec
   and compared to the actual ``y.sum(-1)``. A *transient* fault in the
   GEMM (a flipped bit in a register file or tensor-core accumulator that
   never touched the stored weight, the failure mode ECC at rest cannot
   see) diverges here and triggers a recompute-replay.

What this does **not** do
-------------------------
* Sub-tolerance mantissa-LSB flips do not move a sum past ``(rtol, atol)``
  and are intentionally ignored, they are individually harmless (see the
  protection-module docs and ``AttentionChecksum``). Exponent / MSB flips,
  the dangerous ones, shift a sum by a power of two and are always caught.
* Two or more simultaneously-corrupted weight elements exceed the
  single-error correction capability; they are *detected* (the checksums
  still mismatch) but not corrected. Compose with ``SelectiveTMR`` or
  scrubbing for multi-error coverage.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa: N812, torch convention


class ABFTLinear(nn.Module):
    """Drop-in :class:`torch.nn.Linear` with algorithm-based fault tolerance.

    Args:
        original: The :class:`torch.nn.Linear` to wrap. Its weight and bias
            are cloned, not moved.
        rtol: Relative tolerance for both checksum comparisons.
        atol: Absolute tolerance for both checksum comparisons.
        correct: If ``True`` (default), forward runs weight verification
            and single-error correction *before* the matmul, and replays
            the matmul once on a compute-path mismatch. If ``False`` the
            layer is detect-only and never mutates state.
        trainable: If ``True`` the weight/bias are trainable parameters.
            Defaults to ``False`` for inference safety. Note that training
            changes the weight and therefore invalidates the trusted
            checksums; call :meth:`refresh_checksums` after an optimiser
            step (or use :class:`~spacellm.protection.tmr.SelectiveTMR`
            for training-side protection).
    """

    def __init__(
        self,
        original: nn.Linear,
        *,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        correct: bool = True,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = original.in_features
        self.out_features = original.out_features
        self._has_bias = original.bias is not None
        self.rtol = float(rtol)
        self.atol = float(atol)
        self.correct = bool(correct)

        weight = original.weight.detach().clone()
        self.weight = nn.Parameter(weight, requires_grad=trainable)
        if self._has_bias:
            assert original.bias is not None
            self.bias = nn.Parameter(original.bias.detach().clone(), requires_grad=trainable)
        else:
            self.register_parameter("bias", None)

        self._row_checksum: Tensor = weight.sum(dim=1).clone()
        self._col_checksum: Tensor = weight.sum(dim=0).clone()
        self._bias_sum: Tensor = (
            self.bias.detach().sum().clone()
            if self._has_bias
            else torch.zeros((), dtype=weight.dtype)
        )

        self.n_checks: int = 0
        self.n_detected: int = 0
        self.n_corrected: int = 0

    def refresh_checksums(self) -> None:
        """Recompute the trusted checksums from the current weight/bias.

        Call after a legitimate weight change (e.g. a training step) so the
        new values become the trusted reference.
        """
        with torch.no_grad():
            self._row_checksum = self.weight.detach().sum(dim=1).clone()
            self._col_checksum = self.weight.detach().sum(dim=0).clone()
            if self._has_bias:
                assert self.bias is not None
                self._bias_sum = self.bias.detach().sum().clone()

    def verify_weight(self) -> int:
        """Check the stored weight against the trusted row/column checksums.

        When ``correct`` is set and exactly one row and one column diverge,
        the single offending element is repaired in place.

        Returns:
            Number of anomalies detected (0 when the weight is intact). A
            corrected single-element flip still counts as one detection.
        """
        self.n_checks += 1
        with torch.no_grad():
            row_now = self.weight.detach().sum(dim=1)
            col_now = self.weight.detach().sum(dim=0)
            row_bad = ~torch.isclose(row_now, self._row_checksum, rtol=self.rtol, atol=self.atol)
            col_bad = ~torch.isclose(col_now, self._col_checksum, rtol=self.rtol, atol=self.atol)
            n_row_bad = int(row_bad.sum().item())
            n_col_bad = int(col_bad.sum().item())
            if n_row_bad == 0 and n_col_bad == 0:
                return 0

            self.n_detected += 1
            if self.correct and n_row_bad == 1 and n_col_bad == 1:
                o = int(torch.nonzero(row_bad, as_tuple=False)[0].item())
                i = int(torch.nonzero(col_bad, as_tuple=False)[0].item())
                delta = (row_now[o] - self._row_checksum[o]).to(self.weight.dtype)
                self.weight[o, i] -= delta
                self.n_corrected += 1
            return 1

    def _compute_checksum_mismatch(self, x: Tensor, y: Tensor) -> bool:
        predicted = F.linear(x, self._col_checksum.unsqueeze(0), self._bias_sum.reshape(1))
        actual = y.sum(dim=-1, keepdim=True)
        return not bool(
            torch.isclose(actual, predicted, rtol=self.rtol, atol=self.atol).all().item(),
        )

    def forward(self, x: Tensor) -> Tensor:
        if self.correct:
            self.verify_weight()
        y = F.linear(x, self.weight, self.bias)
        # Compute-path (transient) checksum: only meaningful once the weight
        # is known-good, so it runs after weight verification/correction.
        if self._compute_checksum_mismatch(x, y):
            self.n_detected += 1
            if self.correct:
                y_replay = F.linear(x, self.weight, self.bias)
                if not self._compute_checksum_mismatch(x, y_replay):
                    self.n_corrected += 1
                return y_replay
        return y

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self._has_bias}, abft=row+col+compute"
        )


__all__ = ["ABFTLinear"]
