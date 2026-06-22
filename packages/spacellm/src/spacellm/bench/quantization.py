"""Number-format × radiation sensitivity sweep.

The weight number format (FP32 / FP16 / BF16 / INT8) trades memory, power,
and throughput, and also radiation robustness, which is usually invisible
at selection time. This module measures the robustness axis.

Using the same bit-level :func:`~spacellm._internal.bitops.flip_bit` as the
rest of the framework: for a population of representative values cast to
each dtype, flip every bit of every element one at a time and record the
relative magnitude shift and whether the result went non-finite. The
expected asymmetry follows:

* **FP32** spends 8 of 32 bits on the exponent (25 %); a high-exponent
  flip multiplies the value by a huge power of two or yields NaN/Inf.
* **FP16** has 5 exponent bits of 16 (31 %), smaller dynamic range.
* **BF16** keeps FP32's 8 exponent bits in 16 total (50 %), the *highest*
  exponent-bit fraction, so per stored bit it is the most flip-sensitive
  float, even though each value is half the size.
* **INT8** has no exponent: a flip is a bounded additive shift, it can
  never produce a NaN/Inf or a 10³⁸ blow-up. The most catastrophe-robust
  format, at the cost of dynamic range.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from spacellm._internal.bitops import flip_bit

# Exponent-bit counts per supported dtype (sign / exponent / mantissa).
_EXPONENT_BITS: dict[torch.dtype, int] = {
    torch.float32: 8,
    torch.float16: 5,
    torch.bfloat16: 8,
    torch.float64: 11,
}

_TINY = 1e-30


@dataclass(frozen=True, slots=True)
class NumberFormatSensitivity:
    """Bit-flip sensitivity summary for one number format.

    Attributes:
        dtype_name: e.g. ``"torch.float32"``.
        bits_per_element: Storage width of one element.
        exponent_bit_fraction: Fraction of bits that are exponent bits
            (0 for integer formats).
        mean_relative_shift: Mean ``|Δ| / |original|`` over every
            (element, bit) flip, with non-finite results counted at the
            cap value.
        max_relative_shift: Maximum finite relative shift observed.
        catastrophic_bit_fraction: Fraction of (element, bit) flips that
            either went non-finite or exceeded the catastrophic threshold.
        n_nonfinite_flips: Number of flips that produced NaN/Inf.
        n_flips: Total flips evaluated.
    """

    dtype_name: str
    bits_per_element: int
    exponent_bit_fraction: float
    mean_relative_shift: float
    max_relative_shift: float
    catastrophic_bit_fraction: float
    n_nonfinite_flips: int
    n_flips: int


def format_sensitivity(
    values: torch.Tensor,
    dtype: torch.dtype,
    *,
    catastrophic_threshold: float = 1.0,
    nonfinite_cap: float = 1e30,
) -> NumberFormatSensitivity:
    """Measure per-bit flip sensitivity of ``values`` stored as ``dtype``.

    Args:
        values: 1-D tensor of representative (non-zero) values. Cast to
            ``dtype`` internally.
        dtype: Target number format (``float32/16``, ``bfloat16``,
            ``float64``, ``int8/16/32``).
        catastrophic_threshold: A flip whose relative magnitude shift
            exceeds this (or goes non-finite) is "catastrophic". Defaults
            to 1.0 (a ≥100 % shift).
        nonfinite_cap: Relative-shift value attributed to a non-finite
            result when forming means. Defaults to 1e30.

    Returns:
        A :class:`NumberFormatSensitivity`.

    Raises:
        ValueError: If ``values`` is not 1-D / is empty, or the dtype
            element size is unsupported.
    """
    if values.ndim != 1:
        raise ValueError("values must be 1-D")
    if values.numel() == 0:
        raise ValueError("values must not be empty")

    bits = torch.empty(0, dtype=dtype).element_size() * 8
    if dtype.is_floating_point:
        work = values.to(dtype).clone().contiguous()
    else:
        # Quantise the float population into this integer format's range so
        # the representative values are non-zero and span the dynamic range.
        max_abs = float(values.abs().max().item()) or 1.0
        scale = (2 ** (bits - 1) - 1) / max_abs
        work = torch.round(values.to(torch.float64) * scale).to(dtype).clone().contiguous()
    flat = work.reshape(-1)
    n = flat.numel()

    total_shift = 0.0
    max_finite_shift = 0.0
    n_nonfinite = 0
    n_catastrophic = 0
    n_flips = 0

    for idx in range(n):
        for bit in range(bits):
            before = float(flat[idx].item())
            flip_bit(work, idx, bit)
            after = float(flat[idx].item())
            flip_bit(work, idx, bit)  # restore
            n_flips += 1

            if math.isnan(after) or math.isinf(after):
                n_nonfinite += 1
                n_catastrophic += 1
                total_shift += nonfinite_cap
                continue
            rel = abs(after - before) / max(abs(before), _TINY)
            total_shift += rel
            max_finite_shift = max(max_finite_shift, rel)
            if rel > catastrophic_threshold:
                n_catastrophic += 1

    exp_bits = _EXPONENT_BITS.get(dtype, 0)
    return NumberFormatSensitivity(
        dtype_name=str(dtype),
        bits_per_element=bits,
        exponent_bit_fraction=exp_bits / bits,
        mean_relative_shift=total_shift / n_flips,
        max_relative_shift=max_finite_shift,
        catastrophic_bit_fraction=n_catastrophic / n_flips,
        n_nonfinite_flips=n_nonfinite,
        n_flips=n_flips,
    )


def compare_number_formats(
    values: torch.Tensor,
    dtypes: Sequence[torch.dtype] = (
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.int8,
    ),
    *,
    catastrophic_threshold: float = 1.0,
) -> dict[str, NumberFormatSensitivity]:
    """Run :func:`format_sensitivity` across several formats.

    Args:
        values: 1-D tensor of representative non-zero values.
        dtypes: Formats to compare. Defaults to the four an orbital
            operator actually chooses between.
        catastrophic_threshold: Forwarded to :func:`format_sensitivity`.

    Returns:
        ``dtype_name → NumberFormatSensitivity``.
    """
    return {
        str(dtype): format_sensitivity(
            values,
            dtype,
            catastrophic_threshold=catastrophic_threshold,
        )
        for dtype in dtypes
    }


__all__ = [
    "NumberFormatSensitivity",
    "compare_number_formats",
    "format_sensitivity",
]
