"""Tests for ``spacellm.bench.quantization`` (number-format sensitivity)."""

from __future__ import annotations

import pytest
import torch

from spacellm.bench import compare_number_formats, format_sensitivity


@pytest.fixture
def values() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(16) * 0.1 + 0.2  # non-zero, weight-scale magnitudes


def test_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="1-D"):
        format_sensitivity(torch.zeros(2, 2), torch.float32)
    with pytest.raises(ValueError, match="empty"):
        format_sensitivity(torch.zeros(0), torch.float32)


def test_fp32_exponent_flips_blow_up_unbounded(values: torch.Tensor) -> None:
    s = format_sensitivity(values, torch.float32)
    assert s.bits_per_element == 32
    assert s.exponent_bit_fraction == pytest.approx(8 / 32)
    # A high-exponent flip multiplies a small weight by a huge power of two.
    assert s.max_relative_shift > 1e30
    assert s.n_flips == values.numel() * 32


def test_bfloat16_has_highest_exponent_fraction(values: torch.Tensor) -> None:
    results = compare_number_formats(values)
    bf16 = results["torch.bfloat16"]
    fp16 = results["torch.float16"]
    fp32 = results["torch.float32"]
    # BF16 keeps FP32's 8 exponent bits in 16 total → highest fraction.
    assert bf16.exponent_bit_fraction > fp16.exponent_bit_fraction
    assert bf16.exponent_bit_fraction > fp32.exponent_bit_fraction
    assert bf16.exponent_bit_fraction == pytest.approx(0.5)


def test_int8_never_goes_nonfinite(values: torch.Tensor) -> None:
    s = format_sensitivity(values, torch.int8)
    assert s.bits_per_element == 8
    assert s.exponent_bit_fraction == 0.0
    assert s.n_nonfinite_flips == 0  # integers cannot produce NaN/Inf


def test_float_blowup_is_unbounded_int_is_bounded(values: torch.Tensor) -> None:
    results = compare_number_formats(values)
    int8 = results["torch.int8"]
    fp32 = results["torch.float32"]
    # A float exponent flip is unbounded (~10³⁸); an int flip is a bounded
    # additive shift. That bound is the whole radiation-robustness argument
    # for integer formats.
    assert fp32.max_relative_shift > 1e30
    assert int8.max_relative_shift < 1e3


def test_compare_returns_all_requested(values: torch.Tensor) -> None:
    results = compare_number_formats(
        values,
        dtypes=(torch.float32, torch.int8),
    )
    assert set(results) == {"torch.float32", "torch.int8"}
