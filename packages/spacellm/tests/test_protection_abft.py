"""Tests for ``spacellm.nn.abft`` and ``spacellm.protection.abft``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spacellm._internal.bitops import flip_bit
from spacellm.nn import ABFTLinear
from spacellm.protection import ABFTProtection


def test_abft_linear_matches_original_when_clean() -> None:
    torch.manual_seed(0)
    original = nn.Linear(8, 4)
    wrapped = ABFTLinear(original)
    x = torch.randn(2, 8)
    assert torch.allclose(original(x), wrapped(x), atol=1e-6)


def test_abft_linear_handles_no_bias() -> None:
    original = nn.Linear(8, 4, bias=False)
    wrapped = ABFTLinear(original)
    assert wrapped.bias is None
    x = torch.randn(2, 8)
    assert torch.allclose(original(x), wrapped(x), atol=1e-6)


def test_abft_detects_and_corrects_single_weight_flip() -> None:
    torch.manual_seed(0)
    original = nn.Linear(8, 4)
    wrapped = ABFTLinear(original)
    x = torch.randn(2, 8)
    clean = wrapped(x).clone()
    # Corrupt one weight element with a sign-bit flip (large δ → above tolerance).
    flip_bit(wrapped.weight.data, flat_index=5, bit_position=31)
    out = wrapped(x)
    # forward() verified the weight, located the single bad element, and
    # corrected it, so the output is restored.
    assert torch.allclose(out, clean, atol=1e-5)
    assert wrapped.n_detected >= 1
    assert wrapped.n_corrected >= 1


def test_abft_verify_weight_clean_is_noop() -> None:
    torch.manual_seed(1)
    wrapped = ABFTLinear(nn.Linear(6, 3))
    assert wrapped.verify_weight() == 0
    assert wrapped.n_detected == 0
    assert wrapped.n_corrected == 0


def test_abft_detect_only_does_not_mutate() -> None:
    torch.manual_seed(2)
    original = nn.Linear(8, 4)
    wrapped = ABFTLinear(original, correct=False)
    flip_bit(wrapped.weight.data, flat_index=3, bit_position=30)
    corrupted_weight = wrapped.weight.detach().clone()
    n = wrapped.verify_weight()
    assert n == 1
    assert wrapped.n_corrected == 0
    # weight unchanged because correction is disabled
    assert torch.equal(wrapped.weight.detach(), corrupted_weight)


def test_abft_refresh_checksums_after_weight_change() -> None:
    torch.manual_seed(3)
    wrapped = ABFTLinear(nn.Linear(6, 3), trainable=True)
    with torch.no_grad():
        wrapped.weight += 0.5  # legitimate change invalidates old checksums
    # Stale checksums would flag this as corruption.
    assert wrapped.verify_weight() >= 1
    wrapped.refresh_checksums()
    assert wrapped.verify_weight() == 0


def test_abft_protection_rejects_invalid_top_k() -> None:
    with pytest.raises(ValueError, match="top_k_percent"):
        ABFTProtection(top_k_percent=0.0)
    with pytest.raises(ValueError, match="top_k_percent"):
        ABFTProtection(top_k_percent=150.0)


def test_abft_protection_wraps_linears(tiny_transformer: nn.Module) -> None:
    n_linear_before = sum(1 for m in tiny_transformer.modules() if isinstance(m, nn.Linear))
    strategy = ABFTProtection(top_k_percent=100.0)
    strategy.apply(tiny_transformer)
    n_abft = sum(1 for m in tiny_transformer.modules() if isinstance(m, ABFTLinear))
    assert n_abft > 0
    assert n_abft <= n_linear_before
    assert len(strategy.wrapped_paths) == n_abft


def test_abft_protection_overhead_far_below_tmr(tiny_transformer: nn.Module) -> None:
    strategy = ABFTProtection(top_k_percent=100.0)
    strategy.apply(tiny_transformer)
    report = strategy.collect_report()
    assert report.strategy_name == "abft"
    assert report.overhead_bytes > 0
    # Checksums are two vectors per layer; total must be a small fraction of
    # the weights they guard.
    total_weight_bytes = sum(
        p.element_size() * p.numel()
        for m in tiny_transformer.modules()
        if isinstance(m, ABFTLinear)
        for p in (m.weight,)
    )
    assert report.overhead_bytes < 0.25 * total_weight_bytes


def test_abft_protection_does_not_break_forward(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    clean = tiny_transformer(tiny_input_ids).clone()
    ABFTProtection(top_k_percent=100.0).apply(tiny_transformer)
    assert torch.allclose(clean, tiny_transformer(tiny_input_ids), atol=1e-5)


def test_abft_protection_reset_clears_counters(tiny_transformer: nn.Module) -> None:
    strategy = ABFTProtection(top_k_percent=100.0)
    strategy.apply(tiny_transformer)
    # Drive at least one check via a forward.
    tiny_transformer(torch.randint(0, 100, (2, 8)))
    assert strategy.collect_report().n_checks_performed > 0
    strategy.reset()
    assert strategy.collect_report().n_checks_performed == 0
