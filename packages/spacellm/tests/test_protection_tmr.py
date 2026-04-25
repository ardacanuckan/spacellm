"""Tests for ``spacellm.protection.tmr`` and ``spacellm.nn.tmr``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spacellm._internal.bitops import flip_bit
from spacellm.nn import TMRLinear
from spacellm.protection import SelectiveTMR


def test_tmr_linear_matches_original_when_clean() -> None:
    torch.manual_seed(0)
    original = nn.Linear(8, 4)
    wrapped = TMRLinear(original)
    x = torch.randn(2, 8)
    assert torch.allclose(original(x), wrapped(x))


def test_tmr_linear_masks_single_replica_corruption() -> None:
    torch.manual_seed(0)
    original = nn.Linear(8, 4)
    wrapped = TMRLinear(original)
    x = torch.randn(2, 8)
    clean = wrapped(x).clone()
    # Corrupt replica `a` by injecting a sign-bit flip on element 0
    flip_bit(wrapped.weight_a.data, flat_index=0, bit_position=31)
    # Median of (corrupted, clean, clean) == clean
    assert torch.allclose(wrapped(x), clean)


def test_tmr_linear_handles_no_bias() -> None:
    original = nn.Linear(8, 4, bias=False)
    wrapped = TMRLinear(original)
    assert wrapped.bias_a is None
    x = torch.randn(2, 8)
    assert torch.allclose(original(x), wrapped(x))


def test_tmr_linear_replicas_are_frozen_by_default() -> None:
    original = nn.Linear(4, 2)
    wrapped = TMRLinear(original)
    for p in (wrapped.weight_a, wrapped.weight_b, wrapped.weight_c):
        assert not p.requires_grad


def test_tmr_linear_trainable_flag_propagates() -> None:
    original = nn.Linear(4, 2)
    wrapped = TMRLinear(original, trainable=True)
    for p in (wrapped.weight_a, wrapped.weight_b, wrapped.weight_c):
        assert p.requires_grad


def test_selective_tmr_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k_percent"):
        SelectiveTMR(top_k_percent=0.0)
    with pytest.raises(ValueError, match="top_k_percent"):
        SelectiveTMR(top_k_percent=101.0)


def test_selective_tmr_wraps_top_modules_in_tiny_transformer(tiny_transformer: nn.Module) -> None:
    n_linear_before = sum(1 for m in tiny_transformer.modules() if isinstance(m, nn.Linear))
    strategy = SelectiveTMR(top_k_percent=50.0)
    strategy.apply(tiny_transformer)
    n_tmr = sum(1 for m in tiny_transformer.modules() if isinstance(m, TMRLinear))
    assert n_tmr > 0
    assert n_tmr <= n_linear_before  # cannot exceed number of candidates
    # Wrapped paths recorded
    assert len(strategy.wrapped_paths) == n_tmr


def test_selective_tmr_does_not_break_forward(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    clean_out = tiny_transformer(tiny_input_ids).clone()
    SelectiveTMR(top_k_percent=100.0).apply(tiny_transformer)
    hardened_out = tiny_transformer(tiny_input_ids)
    assert torch.allclose(clean_out, hardened_out, atol=1e-6)


def test_selective_tmr_report_has_overhead(tiny_transformer: nn.Module) -> None:
    strategy = SelectiveTMR(top_k_percent=100.0)
    strategy.apply(tiny_transformer)
    report = strategy.collect_report()
    assert report.strategy_name == "selective_tmr"
    assert report.overhead_bytes > 0  # we replaced unprotected linears with 3× ones
    assert report.extra["n_wrapped_modules"] > 0


def test_selective_tmr_no_double_wrap(tiny_transformer: nn.Module) -> None:
    """Re-applying should not wrap already-protected modules."""
    SelectiveTMR(top_k_percent=100.0).apply(tiny_transformer)
    n_after_first = sum(1 for m in tiny_transformer.modules() if isinstance(m, TMRLinear))
    second = SelectiveTMR(top_k_percent=100.0)
    second.apply(tiny_transformer)
    n_wrapped_second_pass = len(second.wrapped_paths)
    assert n_wrapped_second_pass == 0
    n_after_second = sum(1 for m in tiny_transformer.modules() if isinstance(m, TMRLinear))
    assert n_after_first == n_after_second
