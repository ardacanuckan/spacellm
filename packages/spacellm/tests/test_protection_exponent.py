"""Tests for ``spacellm.protection.exponent`` (exponent-flip range guard)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spacellm._internal.bitops import flip_bit
from spacellm.protection import ExponentRangeGuard


def test_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="top_k_percent"):
        ExponentRangeGuard(top_k_percent=0.0)
    with pytest.raises(ValueError, match="margin"):
        ExponentRangeGuard(margin=-0.1)


def test_clean_forward_unchanged(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    clean = tiny_transformer(tiny_input_ids).clone()
    ExponentRangeGuard(top_k_percent=100.0).apply(tiny_transformer)
    assert torch.allclose(clean, tiny_transformer(tiny_input_ids), atol=1e-6)


def test_exponent_flip_is_clamped() -> None:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 4))
    guard = ExponentRangeGuard(top_k_percent=100.0)
    guard.apply(model)
    x = torch.randn(2, 8)
    clean = model(x).clone()
    linear = model[0]
    # Exponent-bit flip → value blows up to ~1e38.
    flip_bit(linear.weight.data, flat_index=0, bit_position=30)
    assert linear.weight.detach().abs().max().item() > 1e30  # blown up
    out = model(x)  # forward pre-hook clamps it back into range
    assert torch.isfinite(out).all()
    report = guard.collect_report()
    assert report.n_corruptions_detected >= 1
    assert report.n_corruptions_recovered >= 1
    # Output is bounded and close to clean (clamp restores in-range value).
    assert (out - clean).abs().max().item() < clean.abs().max().item() * 10 + 1.0


def test_nan_weight_is_neutralised() -> None:
    torch.manual_seed(1)
    model = nn.Sequential(nn.Linear(6, 3))
    guard = ExponentRangeGuard(top_k_percent=100.0)
    guard.apply(model)
    model[0].weight.data.view(-1)[0] = float("nan")
    out = model(torch.randn(2, 6))
    assert torch.isfinite(out).all()
    assert guard.collect_report().n_corruptions_recovered >= 1


def test_overhead_is_tiny(tiny_transformer: nn.Module) -> None:
    guard = ExponentRangeGuard(top_k_percent=100.0)
    guard.apply(tiny_transformer)
    report = guard.collect_report()
    assert report.strategy_name == "exponent_range_guard"
    total_weight_bytes = sum(p.element_size() * p.numel() for p in tiny_transformer.parameters())
    assert report.overhead_bytes < 0.01 * total_weight_bytes


def test_detach_removes_hooks() -> None:
    model = nn.Sequential(nn.Linear(4, 2))
    guard = ExponentRangeGuard(top_k_percent=100.0)
    guard.apply(model)
    guard.detach()
    # After detach, a corrupted weight is NOT clamped on forward.
    flip_bit(model[0].weight.data, flat_index=0, bit_position=30)
    out = model(torch.randn(1, 4))
    assert not torch.isfinite(out).all() or out.abs().max().item() > 1e20


def test_reset_clears_counters() -> None:
    model = nn.Sequential(nn.Linear(4, 2))
    guard = ExponentRangeGuard(top_k_percent=100.0)
    guard.apply(model)
    model(torch.randn(1, 4))
    assert guard.collect_report().n_checks_performed > 0
    guard.reset()
    assert guard.collect_report().n_checks_performed == 0
