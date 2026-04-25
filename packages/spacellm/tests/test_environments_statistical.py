"""Tests for ``spacellm.environments.statistical``."""

from __future__ import annotations

import pytest
import torch

from spacellm.environments import StatisticalEnvironment
from spacellm.types import FaultCause, RadiationStep


def test_step_advances_time_and_returns_step_object() -> None:
    env = StatisticalEnvironment(rate_per_bit_per_s=0.0, seed=0)
    step = env.step(dt=1.0)
    assert isinstance(step, RadiationStep)
    assert env.t == pytest.approx(1.0)
    env.step(dt=0.5)
    assert env.t == pytest.approx(1.5)


def test_step_rejects_non_positive_dt() -> None:
    env = StatisticalEnvironment(seed=0)
    with pytest.raises(ValueError, match="dt"):
        env.step(dt=0.0)
    with pytest.raises(ValueError, match="dt"):
        env.step(dt=-1.0)


def test_zero_rate_produces_zero_faults() -> None:
    env = StatisticalEnvironment(rate_per_bit_per_s=0.0, seed=0)
    tensors = {"w": torch.zeros(64, dtype=torch.float32)}
    events = env.sample_faults(tensors, dt=1000.0)
    assert events == []


def test_high_rate_produces_faults_within_bounds() -> None:
    env = StatisticalEnvironment(rate_per_bit_per_s=1e-3, seed=42)
    tensors = {"w": torch.zeros(64, dtype=torch.float32)}
    events = env.sample_faults(tensors, dt=1.0)
    assert len(events) > 0
    for event in events:
        assert event.tensor_id == "w"
        assert 0 <= event.bit_index < 64 * 32
        assert 0 <= event.bit_position < 32
        assert event.cause is FaultCause.SYNTHETIC


def test_seed_yields_reproducible_event_stream() -> None:
    e1 = StatisticalEnvironment(rate_per_bit_per_s=1e-3, seed=7)
    e2 = StatisticalEnvironment(rate_per_bit_per_s=1e-3, seed=7)
    t1 = torch.zeros(64, dtype=torch.float32)
    t2 = torch.zeros(64, dtype=torch.float32)
    events_a = e1.sample_faults({"w": t1}, dt=1.0)
    events_b = e2.sample_faults({"w": t2}, dt=1.0)
    assert len(events_a) == len(events_b)
    assert [e.bit_index for e in events_a] == [e.bit_index for e in events_b]


def test_negative_rate_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        StatisticalEnvironment(rate_per_bit_per_s=-1.0)


def test_reset_zeroes_time() -> None:
    env = StatisticalEnvironment(seed=0)
    env.step(dt=10.0)
    env.reset()
    assert env.t == 0.0


def test_empty_tensor_produces_no_faults() -> None:
    env = StatisticalEnvironment(rate_per_bit_per_s=1.0, seed=0)
    tensors = {"empty": torch.empty(0, dtype=torch.float32)}
    assert env.sample_faults(tensors, dt=1.0) == []


def test_cause_attribution_is_preserved() -> None:
    env = StatisticalEnvironment(
        rate_per_bit_per_s=1e-2,
        seed=0,
        cause=FaultCause.SEU_PROTON,
    )
    tensors = {"w": torch.zeros(32, dtype=torch.float32)}
    events = env.sample_faults(tensors, dt=1.0)
    assert events  # rate is high enough to produce something
    assert all(e.cause is FaultCause.SEU_PROTON for e in events)
