"""Tests for ``spacellm.protection.gradient`` (training-time SDC guard)."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spacellm._internal.bitops import flip_bit
from spacellm.protection import GradientGuard


def _model_with_grads() -> nn.Module:
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 4))
    x = torch.randn(4, 8)
    target = torch.randn(4, 4)
    loss = ((model(x) - target) ** 2).mean()
    loss.backward()
    return model


def test_gradient_guard_rejects_bad_params() -> None:
    with pytest.raises(ValueError, match="spike_factor"):
        GradientGuard(spike_factor=1.0)
    with pytest.raises(ValueError, match="ema_decay"):
        GradientGuard(ema_decay=1.0)


def test_clean_gradients_flagged_none() -> None:
    model = _model_with_grads()
    guard = GradientGuard()
    assert guard.check_gradients(model) == 0
    report = guard.collect_report()
    assert report.n_corruptions_detected == 0
    assert report.n_checks_performed > 0


def test_non_finite_gradient_detected_and_recovered() -> None:
    model = _model_with_grads()
    guard = GradientGuard(recover=True)
    guard.check_gradients(model)  # establish baseline
    # Force a NaN into a gradient (an exponent flip would do this on orbit).
    first = next(p for p in model.parameters() if p.grad is not None)
    first.grad.view(-1)[0] = float("nan")
    flagged = guard.check_gradients(model)
    assert flagged >= 1
    assert torch.isfinite(first.grad).all()  # sanitised
    report = guard.collect_report()
    assert report.n_corruptions_detected >= 1
    assert report.n_corruptions_recovered >= 1
    assert report.extra["n_non_finite"] >= 1


def test_magnitude_spike_detected() -> None:
    model = _model_with_grads()
    guard = GradientGuard(spike_factor=50.0, recover=True)
    # Seed the EMA over a few clean checks.
    for _ in range(3):
        guard.check_gradients(model)
    # Exponent-bit flip on one gradient element → huge finite value.
    target_param = next(p for p in model.parameters() if p.grad is not None)
    flip_bit(target_param.grad.data, flat_index=0, bit_position=30)
    flagged = guard.check_gradients(model)
    assert flagged >= 1
    assert guard.collect_report().extra["n_spikes"] >= 1


def test_detect_only_mode_does_not_mutate() -> None:
    model = _model_with_grads()
    guard = GradientGuard(recover=False)
    guard.check_gradients(model)
    first = next(p for p in model.parameters() if p.grad is not None)
    first.grad.view(-1)[0] = float("inf")
    before = first.grad.detach().clone()
    flagged = guard.check_gradients(model)
    assert flagged >= 1
    # Not sanitised because recovery is disabled.
    assert not torch.isfinite(before).all()
    assert (torch.isnan(first.grad) | torch.isinf(first.grad)).any()
    assert guard.collect_report().n_corruptions_recovered == 0


def test_optimizer_state_corruption_detected() -> None:
    model = _model_with_grads()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.step()  # populates exp_avg / exp_avg_sq
    guard = GradientGuard()
    assert guard.check_optimizer_state(optimizer) == 0  # clean baseline
    # Corrupt one optimizer-state tensor.
    state = optimizer.state[next(iter(model.parameters()))]
    state["exp_avg"].view(-1)[0] = float("nan")
    flagged = guard.check_optimizer_state(optimizer)
    assert flagged >= 1
    assert guard.collect_report().n_corruptions_detected >= 1


def test_apply_is_noop_on_graph() -> None:
    model = _model_with_grads()
    guard = GradientGuard()
    assert guard.apply(model) is model


def test_reset_clears_counters() -> None:
    model = _model_with_grads()
    guard = GradientGuard()
    guard.check_gradients(model)
    guard.reset()
    report = guard.collect_report()
    assert report.n_checks_performed == 0
    assert report.n_corruptions_detected == 0
