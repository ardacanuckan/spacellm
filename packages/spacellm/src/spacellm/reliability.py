"""Compose protection layers into a single residual silent-error rate.

Defenses are usually stacked: ECC on memory at rest, periodic scrubbing,
and a compute-level checksum
(:class:`~spacellm.protection.abft.ABFTProtection`). This module composes
them into the silent-error rate that survives all layers, and reports how
much each layer contributes, with no layer assumed perfect.

The layers compose as independent escape probabilities applied to the raw
upset stream ``λ_raw = seu_rate · n_bits``:

* **SECDED ECC + scrubbing.** A single-bit upset in an ECC word is
  *corrected* (silently, at rest). It only escapes correction if a second
  bit in the *same word* flips before the next scrub, or if a multi-cell
  strike deposits ≥2 bits in one word at once. The accumulation term is
  ``word_bits · λ_bit · (T_scrub / 2)`` (mean exposure half-window); the
  immediate term is the multi-cell double fraction. Shorter scrub periods
  and wider ECC words both shrink the escape probability, which is exactly
  the knob a scrubber tunes.
* **Compute ABFT.** Whatever escapes ECC reaches the matmul, where the
  checksum catches a fraction ``abft_coverage`` (ALBERTA reports > 0.99).
  The residual escape is ``1 − abft_coverage``.

The product is the residual rate; ``improvement_factor`` is how many times
lower that is than the unprotected stream.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

_SECONDS_PER_DAY = 86_400.0

EccScheme = Literal["none", "secded"]


@dataclass(frozen=True, slots=True)
class ReliabilityStack:
    """A stack of at-rest and compute-level defenses.

    Attributes:
        raw_seu_rate_per_bit_per_s: Uncorrected upset rate per bit.
        n_bits: Bits of protected model state.
        ecc: ``"secded"`` (single-error-correct, double-error-detect) or
            ``"none"``.
        word_data_bits: ECC word width in data bits. Defaults to 64.
        scrub_period_s: Seconds between memory-scrub passes. Defaults to
            1.0.
        abft_coverage: Fraction of post-ECC corruptions the compute
            checksum catches, in ``[0, 1]``. Defaults to 0.0 (no ABFT).
        mcu_double_fraction: Fraction of strikes that deposit ≥2 bits in a
            single ECC word at once (defeating SECDED immediately), in
            ``[0, 1]``. Defaults to 0.0. Derive from the MCU module for
            fine-node COTS silicon.

    Raises:
        ValueError: On out-of-range inputs.
    """

    raw_seu_rate_per_bit_per_s: float
    n_bits: int
    ecc: EccScheme = "secded"
    word_data_bits: int = 64
    scrub_period_s: float = 1.0
    abft_coverage: float = 0.0
    mcu_double_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.raw_seu_rate_per_bit_per_s < 0:
            raise ValueError("raw_seu_rate_per_bit_per_s must be non-negative")
        if self.n_bits <= 0:
            raise ValueError("n_bits must be positive")
        if self.ecc not in ("none", "secded"):
            raise ValueError("ecc must be 'none' or 'secded'")
        if self.word_data_bits <= 0:
            raise ValueError("word_data_bits must be positive")
        if self.scrub_period_s <= 0:
            raise ValueError("scrub_period_s must be positive")
        if not 0.0 <= self.abft_coverage <= 1.0:
            raise ValueError("abft_coverage must be in [0, 1]")
        if not 0.0 <= self.mcu_double_fraction <= 1.0:
            raise ValueError("mcu_double_fraction must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ResidualRateReport:
    """Composed residual silent-error budget.

    Attributes:
        raw_rate_per_s: Unprotected corrupting-upset rate ``λ_raw``.
        ecc_escape_prob: Probability an upset escapes ECC+scrubbing
            correction.
        abft_escape_prob: ``1 − abft_coverage``.
        residual_silent_rate_per_s: Rate of upsets that survive *all*
            layers.
        improvement_factor: ``raw / residual`` (``inf`` if residual is 0).
        expected_silent_per_day: ``residual · 86400``.
        mean_time_between_silent_s: ``1 / residual`` (``inf`` if 0).
    """

    raw_rate_per_s: float
    ecc_escape_prob: float
    abft_escape_prob: float
    residual_silent_rate_per_s: float
    improvement_factor: float
    expected_silent_per_day: float
    mean_time_between_silent_s: float

    def summary(self) -> str:
        imp = "∞" if math.isinf(self.improvement_factor) else f"{self.improvement_factor:.1f}×"
        mtbs = (
            "∞"
            if math.isinf(self.mean_time_between_silent_s)
            else f"{self.mean_time_between_silent_s:.3e} s"
        )
        return (
            "SpaceLLM residual silent-error budget\n"
            f"  raw upset rate     : {self.raw_rate_per_s:.3e} /s\n"
            f"  ECC escape prob    : {self.ecc_escape_prob:.3e}\n"
            f"  ABFT escape prob   : {self.abft_escape_prob:.3e}\n"
            f"  residual silent    : {self.residual_silent_rate_per_s:.3e} /s\n"
            f"  improvement        : {imp}\n"
            f"  silent / day       : {self.expected_silent_per_day:.3e}\n"
            f"  MTB silent error   : {mtbs}"
        )


def compose_residual_rate(stack: ReliabilityStack) -> ResidualRateReport:
    """Compose a :class:`ReliabilityStack` into a residual silent-error rate.

    Args:
        stack: The configured defense layers.

    Returns:
        A :class:`ResidualRateReport`.
    """
    raw_rate = stack.raw_seu_rate_per_bit_per_s * stack.n_bits

    if stack.ecc == "none":
        ecc_escape = 1.0
    else:
        # SECDED corrects singles; escape requires a same-word double, either
        # accumulated within the scrub half-window or deposited at once.
        accumulation = (
            stack.word_data_bits * stack.raw_seu_rate_per_bit_per_s * (stack.scrub_period_s / 2.0)
        )
        ecc_escape = min(1.0, stack.mcu_double_fraction + accumulation)

    abft_escape = 1.0 - stack.abft_coverage
    residual = raw_rate * ecc_escape * abft_escape

    if residual > 0.0:
        improvement = raw_rate / residual
        mtbs = 1.0 / residual
    else:
        improvement = math.inf
        mtbs = math.inf

    return ResidualRateReport(
        raw_rate_per_s=raw_rate,
        ecc_escape_prob=ecc_escape,
        abft_escape_prob=abft_escape,
        residual_silent_rate_per_s=residual,
        improvement_factor=improvement,
        expected_silent_per_day=residual * _SECONDS_PER_DAY,
        mean_time_between_silent_s=mtbs,
    )


__all__ = ["EccScheme", "ReliabilityStack", "ResidualRateReport", "compose_residual_rate"]
