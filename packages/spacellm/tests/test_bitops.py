"""Tests for ``spacellm._internal.bitops``."""

from __future__ import annotations

import pytest
import torch

from spacellm._internal.bitops import flip_bit, flip_bit_uniform


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.float16,
        torch.bfloat16,
        torch.float64,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ],
)
def test_flip_bit_inverts_specified_bit(dtype: torch.dtype) -> None:
    t = torch.zeros(8, dtype=dtype)
    before, after = flip_bit(t, flat_index=3, bit_position=0)
    assert before == 0
    assert after == 1


def test_flip_bit_is_idempotent_under_double_flip() -> None:
    t = torch.zeros(8, dtype=torch.float32)
    flip_bit(t, 0, 5)
    flip_bit(t, 0, 5)
    assert (t == 0).all()


def test_flip_bit_propagates_to_underlying_storage() -> None:
    t = torch.zeros(4, dtype=torch.float32)
    flip_bit(t, 1, 31)  # sign bit of FP32 element 1
    # FP32 sign bit set on a zero element produces -0.0 in float-space.
    assert torch.signbit(t[1]).item()


def test_flip_bit_rejects_out_of_range_indices() -> None:
    t = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(IndexError):
        flip_bit(t, 10, 0)
    with pytest.raises(IndexError):
        flip_bit(t, 0, 100)
    with pytest.raises(IndexError):
        flip_bit(t, -1, 0)
    with pytest.raises(IndexError):
        flip_bit(t, 0, -1)


def test_flip_bit_uniform_decomposes_high_index() -> None:
    t = torch.zeros(2, dtype=torch.float32)
    before, after, bit_position = flip_bit_uniform(t, bit_index=33)
    assert bit_position == 1
    assert before == 0
    assert after == 0b10


def test_flip_bit_handles_already_set_bit() -> None:
    t = torch.zeros(2, dtype=torch.int32)
    # Set bit 0 of element 0
    flip_bit(t, 0, 0)
    # Now flip again — should clear
    before, after = flip_bit(t, 0, 0)
    assert before == 1
    assert after == 0


def test_flip_bit_int8_roundtrips_through_signed_storage() -> None:
    """Sign bit of a signed int8 — make sure two's-complement plumbing works."""
    t = torch.zeros(4, dtype=torch.int8)
    before, after = flip_bit(t, 0, 7)  # MSB = sign bit for int8
    assert before == 0
    assert after == 0b1000_0000  # 128 unsigned == -128 signed


def test_flip_bit_complex_dtype_unsupported() -> None:
    t = torch.zeros(4, dtype=torch.complex64)
    with pytest.raises(TypeError):
        flip_bit(t, 0, 0)
