"""Tests for ``spacellm.environments.mcu``."""

from __future__ import annotations

import pytest
import torch

from spacellm._internal.sampling import RandomState
from spacellm.environments import MCUDistribution, MCUEnvironment, StatisticalEnvironment
from spacellm.types import FaultCause


def test_distribution_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MCUDistribution(probabilities=())


def test_distribution_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        MCUDistribution(probabilities=(0.5, -0.5, 1.0))


def test_distribution_rejects_unnormalised() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        MCUDistribution(probabilities=(0.5, 0.4))


def test_distribution_max_and_expected_size() -> None:
    d = MCUDistribution(probabilities=(0.5, 0.3, 0.2))
    assert d.max_size == 3
    assert d.expected_size == pytest.approx(1.0 * 0.5 + 2.0 * 0.3 + 3.0 * 0.2)


def test_distribution_samples_within_range() -> None:
    rng = RandomState(seed=0)
    d = MCUDistribution(probabilities=(0.4, 0.3, 0.2, 0.1))
    sizes = [d.sample_size(rng) for _ in range(2000)]
    assert min(sizes) >= 1
    assert max(sizes) <= 4


def test_distribution_empirical_mean_matches_expectation() -> None:
    rng = RandomState(seed=42)
    d = MCUDistribution(probabilities=(0.4, 0.3, 0.2, 0.1))
    sizes = [d.sample_size(rng) for _ in range(5000)]
    empirical = sum(sizes) / len(sizes)
    assert empirical == pytest.approx(d.expected_size, rel=0.05)


@pytest.mark.parametrize(
    ("node_nm", "expected_max_size"),
    [(180.0, 1), (90.0, 1), (45.0, 4), (28.0, 4), (16.0, 4), (12.0, 5), (7.0, 5)],
)
def test_distribution_for_process_node_grows_with_shrinking_node(
    node_nm: float,
    expected_max_size: int,
) -> None:
    d = MCUDistribution.for_process_node(node_nm)
    assert d.max_size == expected_max_size


def test_distribution_for_process_node_increases_expected_size_at_smaller_nodes() -> None:
    big = MCUDistribution.for_process_node(90.0)
    small = MCUDistribution.for_process_node(7.0)
    assert big.expected_size < small.expected_size


def test_mcu_environment_passes_through_with_sbu_only_distribution() -> None:
    base = StatisticalEnvironment(rate_per_bit_per_s=1e-3, seed=0)
    sbu_only = MCUDistribution(probabilities=(1.0,))
    env = MCUEnvironment(base, sbu_only, seed=0)
    tensors = {"w": torch.zeros(64, dtype=torch.float32)}
    events = env.sample_faults(tensors, dt=1.0)
    # SBU-only distribution must produce exactly the base set, no extras.
    base_events = StatisticalEnvironment(rate_per_bit_per_s=1e-3, seed=0).sample_faults(
        {"w": torch.zeros(64, dtype=torch.float32)},
        dt=1.0,
    )
    assert len(events) == len(base_events)


def test_mcu_environment_amplifies_event_count() -> None:
    base = StatisticalEnvironment(rate_per_bit_per_s=1e-2, seed=42)
    distribution = MCUDistribution.for_process_node(7.0)  # heavy MCU
    env = MCUEnvironment(base, distribution, seed=0)
    tensors = {"w": torch.zeros(64, dtype=torch.float32)}
    events = env.sample_faults(tensors, dt=1.0)
    # With expected cluster size > 1.5, we should see >> base count.
    base_count = StatisticalEnvironment(rate_per_bit_per_s=1e-2, seed=42).sample_faults(
        {"w": torch.zeros(64, dtype=torch.float32)},
        dt=1.0,
    )
    assert len(events) > len(base_count)


def test_mcu_events_share_cause_with_seed_event() -> None:
    distribution = MCUDistribution(probabilities=(0.0, 0.0, 1.0))  # always size-3

    # Count base seeds first (same seed reproduces).
    base_for_count = StatisticalEnvironment(
        rate_per_bit_per_s=1e-2,
        seed=7,
        cause=FaultCause.SEU_HEAVY_ION,
    )
    base_events = base_for_count.sample_faults(
        {"w": torch.zeros(32, dtype=torch.float32)},
        dt=1.0,
    )
    n_seeds = len(base_events)
    assert n_seeds > 0

    # MCU-wrapped run should produce exactly 3× the base events.
    base = StatisticalEnvironment(
        rate_per_bit_per_s=1e-2,
        seed=7,
        cause=FaultCause.SEU_HEAVY_ION,
    )
    env = MCUEnvironment(base, distribution, seed=0)
    events = env.sample_faults({"w": torch.zeros(32, dtype=torch.float32)}, dt=1.0)
    assert len(events) == 3 * n_seeds
    assert all(e.cause is FaultCause.SEU_HEAVY_ION for e in events)


def test_mcu_environment_t_and_step_pass_through() -> None:
    base = StatisticalEnvironment(rate_per_bit_per_s=0.0, seed=0)
    env = MCUEnvironment(base, MCUDistribution.for_process_node(28.0), seed=0)
    assert env.t == 0.0
    step = env.step(dt=2.5)
    assert env.t == pytest.approx(2.5)
    assert step.source == "statistical"


def test_mcu_environment_reset_zeroes_time() -> None:
    base = StatisticalEnvironment(rate_per_bit_per_s=0.0, seed=0)
    env = MCUEnvironment(base, MCUDistribution.for_process_node(28.0), seed=0)
    env.step(dt=10.0)
    env.reset()
    assert env.t == 0.0
