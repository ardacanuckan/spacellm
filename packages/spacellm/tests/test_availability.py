"""Tests for ``spacellm.availability`` (eclipse + SEL FDIR model)."""

from __future__ import annotations

import math

import pytest

from spacellm.availability import AvailabilityProfile, simulate_availability


def _profile(**overrides: object) -> AvailabilityProfile:
    base: dict[str, object] = {
        "sel_rate_per_s": 1e-5,
        "reboot_time_s": 120.0,
        "orbit_period_s": 5560.0,
        "eclipse_fraction": 0.35,
        "mission_duration_s": 30 * 86400.0,
    }
    base.update(overrides)
    return AvailabilityProfile(**base)  # type: ignore[arg-type]


def test_profile_validates_inputs() -> None:
    with pytest.raises(ValueError, match="reboot_time_s"):
        _profile(reboot_time_s=0.0)
    with pytest.raises(ValueError, match="eclipse_fraction"):
        _profile(eclipse_fraction=1.0)
    with pytest.raises(ValueError, match="sel_rate_per_s"):
        _profile(sel_rate_per_s=-1.0)


def test_availability_in_unit_interval() -> None:
    report = simulate_availability(_profile(), seed=0)
    assert 0.0 <= report.availability <= 1.0
    assert report.uptime_s >= 0.0
    assert report.total_downtime_s >= report.eclipse_downtime_s


def test_eclipse_only_when_no_sel() -> None:
    report = simulate_availability(_profile(sel_rate_per_s=0.0), seed=1)
    assert report.n_sel_events == 0
    assert report.reboot_downtime_s == pytest.approx(0.0, abs=1e-6)
    # Availability ≈ 1 - eclipse_fraction.
    assert report.availability == pytest.approx(0.65, abs=0.02)


def test_no_eclipse_full_uptime_without_sel() -> None:
    report = simulate_availability(
        _profile(eclipse_fraction=0.0, sel_rate_per_s=0.0),
        seed=2,
    )
    assert report.n_eclipse_cycles == 0
    assert report.availability == pytest.approx(1.0, abs=1e-9)
    assert math.isinf(report.mtbf_s)


def test_higher_sel_rate_lowers_availability() -> None:
    low = simulate_availability(_profile(sel_rate_per_s=1e-6), seed=3)
    high = simulate_availability(_profile(sel_rate_per_s=1e-4), seed=3)
    assert high.availability <= low.availability
    assert high.n_effective_reboots >= low.n_effective_reboots


def test_reproducible_with_seed() -> None:
    a = simulate_availability(_profile(), seed=7)
    b = simulate_availability(_profile(), seed=7)
    assert a.availability == b.availability
    assert a.n_sel_events == b.n_sel_events


def test_mttr_equals_reboot_time() -> None:
    report = simulate_availability(_profile(reboot_time_s=90.0), seed=4)
    assert report.mttr_s == 90.0


def test_masked_plus_effective_equals_total_sel() -> None:
    report = simulate_availability(_profile(sel_rate_per_s=1e-4), seed=5)
    assert report.n_effective_reboots + report.n_masked_sel == report.n_sel_events


def test_summary_is_readable() -> None:
    text = simulate_availability(_profile(), seed=0).summary()
    assert "availability budget" in text
    assert "MTBF" in text
