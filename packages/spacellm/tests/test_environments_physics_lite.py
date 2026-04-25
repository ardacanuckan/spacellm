"""Tests for ``spacellm.environments.physics_lite``."""

from __future__ import annotations

import pytest
import torch

from spacellm.environments import LEO_ISS_NOMINAL, MARS_TRANSIT, PhysicsLiteEnvironment
from spacellm.environments.devices import POLARFIRE_LSRAM
from spacellm.types import FaultCause, RadiationStep


def test_constructs_for_real_device_and_leo() -> None:
    env = PhysicsLiteEnvironment(POLARFIRE_LSRAM, LEO_ISS_NOMINAL, seed=0)
    assert env.t == 0.0
    assert env.mission_seu_rate_per_bit_per_s > 0
    assert env.effective_seu_rate_per_bit_per_s == pytest.approx(
        env.mission_seu_rate_per_bit_per_s,
    )


def test_time_acceleration_scales_effective_rate() -> None:
    base = PhysicsLiteEnvironment(POLARFIRE_LSRAM, LEO_ISS_NOMINAL, seed=0)
    fast = PhysicsLiteEnvironment(
        POLARFIRE_LSRAM,
        LEO_ISS_NOMINAL,
        seed=0,
        time_acceleration=1e6,
    )
    assert fast.effective_seu_rate_per_bit_per_s == pytest.approx(
        base.mission_seu_rate_per_bit_per_s * 1e6,
    )


def test_step_returns_calibrated_radiation_step() -> None:
    env = PhysicsLiteEnvironment(POLARFIRE_LSRAM, MARS_TRANSIT, seed=0)
    step = env.step(dt=1.0)
    assert isinstance(step, RadiationStep)
    assert step.source == "physics_lite:MARS_TRANSIT"
    assert step.tid_rate_gy_per_s == MARS_TRANSIT.tid_rate_gy_per_s
    assert step.proton_flux == MARS_TRANSIT.proton_flux_above_10mev_per_cm2_per_s
    assert step.heavy_ion_flux > 0
    assert env.t == pytest.approx(1.0)


def test_step_rejects_non_positive_dt() -> None:
    env = PhysicsLiteEnvironment(POLARFIRE_LSRAM, LEO_ISS_NOMINAL, seed=0)
    with pytest.raises(ValueError, match="dt"):
        env.step(dt=0.0)
    with pytest.raises(ValueError, match="dt"):
        env.step(dt=-1.0)


def test_high_acceleration_produces_faults_with_correct_attribution() -> None:
    """Pick a tame acceleration so CI doesn't sit on millions of events."""
    env = PhysicsLiteEnvironment(
        POLARFIRE_LSRAM,
        LEO_ISS_NOMINAL,
        seed=42,
        time_acceleration=1e9,
    )
    tensors = {"w": torch.zeros(64, dtype=torch.float32)}
    events = env.sample_faults(tensors, dt=1.0)
    assert events
    assert all(e.cause is FaultCause.SEU_HEAVY_ION for e in events)


def test_seed_reproducibility() -> None:
    e1 = PhysicsLiteEnvironment(
        POLARFIRE_LSRAM,
        LEO_ISS_NOMINAL,
        seed=7,
        time_acceleration=1e9,
    )
    e2 = PhysicsLiteEnvironment(
        POLARFIRE_LSRAM,
        LEO_ISS_NOMINAL,
        seed=7,
        time_acceleration=1e9,
    )
    t1 = torch.zeros(32, dtype=torch.float32)
    t2 = torch.zeros(32, dtype=torch.float32)
    a = e1.sample_faults({"w": t1}, dt=1.0)
    b = e2.sample_faults({"w": t2}, dt=1.0)
    assert [e.bit_index for e in a] == [e.bit_index for e in b]


def test_reset_zeroes_time() -> None:
    env = PhysicsLiteEnvironment(POLARFIRE_LSRAM, LEO_ISS_NOMINAL, seed=0)
    env.step(dt=10.0)
    env.reset()
    assert env.t == 0.0


def test_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="bits_per_cell"):
        PhysicsLiteEnvironment(POLARFIRE_LSRAM, LEO_ISS_NOMINAL, bits_per_cell=0.0)
    with pytest.raises(ValueError, match="time_acceleration"):
        PhysicsLiteEnvironment(POLARFIRE_LSRAM, LEO_ISS_NOMINAL, time_acceleration=-1.0)


def test_bits_per_cell_divides_rate() -> None:
    env_one = PhysicsLiteEnvironment(POLARFIRE_LSRAM, LEO_ISS_NOMINAL, bits_per_cell=1.0)
    env_six = PhysicsLiteEnvironment(POLARFIRE_LSRAM, LEO_ISS_NOMINAL, bits_per_cell=6.0)
    assert env_six.mission_seu_rate_per_bit_per_s == pytest.approx(
        env_one.mission_seu_rate_per_bit_per_s / 6.0,
    )


def test_mars_transit_rate_exceeds_leo_iss_for_polarfire() -> None:
    """A device exposed to Mars transit GCR should see a higher SEU rate than at LEO."""
    leo = PhysicsLiteEnvironment(POLARFIRE_LSRAM, LEO_ISS_NOMINAL)
    mars = PhysicsLiteEnvironment(POLARFIRE_LSRAM, MARS_TRANSIT)
    assert mars.mission_seu_rate_per_bit_per_s > leo.mission_seu_rate_per_bit_per_s
