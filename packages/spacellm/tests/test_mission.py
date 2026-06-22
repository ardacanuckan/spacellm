"""Tests for ``spacellm.mission`` cost modelling."""

from __future__ import annotations

import math

import pytest

from spacellm.mission import MissionProfile, estimate_mission_cost


def _profile(**overrides: object) -> MissionProfile:
    base: dict[str, object] = {
        "seu_rate_per_bit_per_s": 3e-11,
        "n_parameters": 7_000_000_000,
        "step_time_s": 0.5,
        "power_w": 700.0,
        "checkpoint_cost_s": 60.0,
        "run_duration_s": 6 * 3600.0,
    }
    base.update(overrides)
    return MissionProfile(**base)  # type: ignore[arg-type]


def test_profile_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="n_parameters"):
        _profile(n_parameters=0)
    with pytest.raises(ValueError, match="power_w"):
        _profile(power_w=-1.0)
    with pytest.raises(ValueError, match="corrupting_fraction"):
        _profile(corrupting_fraction=1.5)


def test_n_bits_matches_width() -> None:
    p = _profile(n_parameters=1_000_000_000, bytes_per_parameter=2)
    assert p.n_bits == 1_000_000_000 * 2 * 8


def test_cost_report_basic_invariants() -> None:
    report = estimate_mission_cost(_profile())
    assert report.fault_rate_per_s > 0
    assert math.isfinite(report.mtbf_s)
    assert report.expected_faults_per_day > 0
    assert report.optimal_checkpoint_interval_s > 0
    assert report.wasted_energy_kwh_per_day > 0
    # Detection must save energy versus a silent whole-run re-fly.
    assert report.silent_redo_energy_kwh_per_run > 0
    assert report.detection_savings_kwh_per_run >= 0


def test_zero_rate_yields_no_faults() -> None:
    report = estimate_mission_cost(_profile(seu_rate_per_bit_per_s=0.0))
    assert report.fault_rate_per_s == 0.0
    assert math.isinf(report.mtbf_s)
    assert report.expected_faults_per_day == 0.0
    assert report.wasted_energy_kwh_per_day == 0.0
    assert report.detection_savings_kwh_per_run == 0.0


def test_young_optimal_interval_formula() -> None:
    p = _profile()
    report = estimate_mission_cost(p)
    lam = p.seu_rate_per_bit_per_s * p.n_bits * p.corrupting_fraction
    mtbf = 1.0 / lam
    expected = min(math.sqrt(2.0 * p.checkpoint_cost_s * mtbf), p.run_duration_s)
    assert report.optimal_checkpoint_interval_s == pytest.approx(expected, rel=1e-9)


def test_override_checkpoint_interval_is_used() -> None:
    report = estimate_mission_cost(_profile(), checkpoint_interval_s=120.0)
    assert report.used_checkpoint_interval_s == 120.0


def test_override_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="checkpoint_interval_s"):
        estimate_mission_cost(_profile(), checkpoint_interval_s=0.0)


def test_higher_rate_costs_more_energy() -> None:
    low = estimate_mission_cost(_profile(seu_rate_per_bit_per_s=1e-12))
    high = estimate_mission_cost(_profile(seu_rate_per_bit_per_s=1e-10))
    assert high.wasted_energy_kwh_per_day > low.wasted_energy_kwh_per_day


def test_summary_is_readable() -> None:
    text = estimate_mission_cost(_profile()).summary()
    assert "mission cost budget" in text
    assert "detection saves" in text
