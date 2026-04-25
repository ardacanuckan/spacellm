"""Tests for ``spacellm.protection.kv_cache``."""

from __future__ import annotations

import pytest
import torch

from spacellm._internal.bitops import flip_bit
from spacellm.protection import KVCacheParity
from spacellm.protection.kv_cache import _row_xor_parity


def test_row_xor_parity_zero_tensor_is_zero() -> None:
    t = torch.zeros(4, 8, dtype=torch.float32)
    parity = _row_xor_parity(t)
    assert parity.shape == (4,)
    assert (parity == 0).all()


def test_row_xor_parity_changes_when_bit_flipped() -> None:
    t = torch.zeros(4, 8, dtype=torch.float32)
    before = _row_xor_parity(t).clone()
    flip_bit(t, flat_index=5, bit_position=3)  # row 0, mid-row
    after = _row_xor_parity(t)
    assert not torch.equal(before, after)


def test_row_xor_parity_rejects_complex() -> None:
    t = torch.zeros(2, 2, dtype=torch.complex64)
    with pytest.raises(TypeError):
        _row_xor_parity(t)


def test_kv_cache_parity_apply_is_no_op() -> None:
    from torch import nn  # noqa: PLC0415

    model = nn.Linear(4, 4)
    state_before = {k: v.clone() for k, v in model.state_dict().items()}
    KVCacheParity().apply(model)
    state_after = model.state_dict()
    for k, v in state_before.items():
        assert torch.equal(v, state_after[k])


def test_kv_cache_parity_clean_round_trip() -> None:
    strategy = KVCacheParity()
    key = torch.randn(2, 3, 8)
    value = torch.randn(2, 3, 8)
    strategy.guard("layer0", key, value)
    mismatches = strategy.verify("layer0", key, value)
    assert mismatches == 0
    report = strategy.collect_report()
    assert report.n_corruptions_detected == 0
    assert report.n_checks_performed == 2


def test_kv_cache_parity_detects_key_corruption() -> None:
    strategy = KVCacheParity()
    key = torch.zeros(2, 3, 8, dtype=torch.float32)
    value = torch.zeros(2, 3, 8, dtype=torch.float32)
    strategy.guard("layer0", key, value)
    flip_bit(key, flat_index=5, bit_position=29)  # exponent bit, row 0
    mismatches = strategy.verify("layer0", key, value)
    assert mismatches == 1
    report = strategy.collect_report()
    assert report.n_corruptions_detected == 1


def test_kv_cache_parity_detects_value_corruption() -> None:
    strategy = KVCacheParity()
    key = torch.zeros(2, 3, 8, dtype=torch.float32)
    value = torch.zeros(2, 3, 8, dtype=torch.float32)
    strategy.guard("layer0", key, value)
    flip_bit(value, flat_index=10, bit_position=15)
    mismatches = strategy.verify("layer0", key, value)
    assert mismatches == 1


def test_kv_cache_parity_unknown_name_returns_zero() -> None:
    strategy = KVCacheParity()
    key = torch.zeros(1, 2, 4)
    value = torch.zeros(1, 2, 4)
    # Verify without prior guard — no record exists, so no mismatch is reported.
    mismatches = strategy.verify("unguarded", key, value)
    assert mismatches == 0


def test_kv_cache_parity_forget_drops_record() -> None:
    strategy = KVCacheParity()
    key = torch.zeros(1, 2, 4)
    value = torch.zeros(1, 2, 4)
    strategy.guard("layer0", key, value)
    strategy.forget("layer0")
    flip_bit(key, flat_index=0, bit_position=0)
    # Record was dropped, so this corrupt key produces no mismatch.
    assert strategy.verify("layer0", key, value) == 0


def test_kv_cache_parity_reset_clears_counters() -> None:
    strategy = KVCacheParity()
    key = torch.zeros(1, 2, 4)
    value = torch.zeros(1, 2, 4)
    strategy.guard("layer0", key, value)
    strategy.verify("layer0", key, value)
    assert strategy.collect_report().n_checks_performed > 0
    strategy.reset()
    assert strategy.collect_report().n_checks_performed == 0
    assert strategy.collect_report().n_corruptions_detected == 0
