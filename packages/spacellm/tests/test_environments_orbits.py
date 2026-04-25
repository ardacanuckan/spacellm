"""Tests for ``spacellm.environments.orbits``."""

from __future__ import annotations

import numpy as np
import pytest

from spacellm.environments.orbits import (
    GEO_QUIET,
    LEO_ISS_NOMINAL,
    MARS_TRANSIT,
    REFERENCE_ORBITS,
    OrbitProfile,
    gcr_like_spectrum,
)


@pytest.mark.parametrize(
    "profile",
    [LEO_ISS_NOMINAL, GEO_QUIET, MARS_TRANSIT],
    ids=lambda p: p.name,
)
def test_reference_orbit_grid_is_strictly_increasing(profile: OrbitProfile) -> None:
    grid = profile.let_grid_mev_cm2_per_mg
    assert grid.ndim == 1
    assert grid.size >= 2
    assert np.all(np.diff(grid) > 0)


@pytest.mark.parametrize(
    "profile",
    [LEO_ISS_NOMINAL, GEO_QUIET, MARS_TRANSIT],
    ids=lambda p: p.name,
)
def test_reference_orbit_flux_aligns_with_grid(profile: OrbitProfile) -> None:
    assert profile.flux_per_let_per_s.shape == profile.let_grid_mev_cm2_per_mg.shape
    assert np.all(profile.flux_per_let_per_s >= 0)


@pytest.mark.parametrize(
    "profile",
    [LEO_ISS_NOMINAL, GEO_QUIET, MARS_TRANSIT],
    ids=lambda p: p.name,
)
def test_reference_orbit_carries_citations(profile: OrbitProfile) -> None:
    assert len(profile.citations) >= 1


def test_reference_orbits_dict_contains_all_three() -> None:
    assert set(REFERENCE_ORBITS) == {"LEO_ISS_NOMINAL", "GEO_QUIET", "MARS_TRANSIT"}
    for name, profile in REFERENCE_ORBITS.items():
        assert profile.name == name


def test_orbit_profile_rejects_mismatched_arrays() -> None:
    with pytest.raises(ValueError, match="match in shape"):
        OrbitProfile(
            name="bad",
            description="",
            altitude_km=400.0,
            inclination_deg=0.0,
            let_grid_mev_cm2_per_mg=np.array([1.0, 2.0, 3.0]),
            flux_per_let_per_s=np.array([1.0, 2.0]),
            proton_flux_above_10mev_per_cm2_per_s=0.0,
            tid_rate_gy_per_s=0.0,
        )


def test_orbit_profile_rejects_non_increasing_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        OrbitProfile(
            name="bad",
            description="",
            altitude_km=400.0,
            inclination_deg=0.0,
            let_grid_mev_cm2_per_mg=np.array([3.0, 2.0, 1.0]),
            flux_per_let_per_s=np.array([1.0, 1.0, 1.0]),
            proton_flux_above_10mev_per_cm2_per_s=0.0,
            tid_rate_gy_per_s=0.0,
        )


def test_orbit_profile_rejects_invalid_saa_fraction() -> None:
    with pytest.raises(ValueError, match="saa_fraction"):
        OrbitProfile(
            name="bad",
            description="",
            altitude_km=400.0,
            inclination_deg=0.0,
            let_grid_mev_cm2_per_mg=np.array([1.0, 2.0]),
            flux_per_let_per_s=np.array([1.0, 1.0]),
            proton_flux_above_10mev_per_cm2_per_s=0.0,
            tid_rate_gy_per_s=0.0,
            saa_fraction=2.0,
        )


def test_gcr_spectrum_normalises_to_integral_flux() -> None:
    grid, flux = gcr_like_spectrum(integral_flux=2.5e-3)
    integral = float(np.trapezoid(flux, grid))
    assert integral == pytest.approx(2.5e-3, rel=1e-3)


def test_gcr_spectrum_is_monotone_decreasing() -> None:
    grid, flux = gcr_like_spectrum(integral_flux=1.0e-3)
    assert np.all(np.diff(flux) < 0)
    assert np.all(np.diff(grid) > 0)


def test_gcr_spectrum_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="integral_flux"):
        gcr_like_spectrum(integral_flux=-1.0)
    with pytest.raises(ValueError, match="let_min"):
        gcr_like_spectrum(integral_flux=1.0, let_min=10.0, let_max=5.0)
    with pytest.raises(ValueError, match="n_points"):
        gcr_like_spectrum(integral_flux=1.0, n_points=1)
    with pytest.raises(ValueError, match="decay_let"):
        gcr_like_spectrum(integral_flux=1.0, decay_let=0.0)


def test_dose_rates_are_in_a_sensible_order() -> None:
    """LEO ISS < GEO quiet < Mars transit (roughly, by integrated GCR exposure)."""
    assert LEO_ISS_NOMINAL.tid_rate_gy_per_s < MARS_TRANSIT.tid_rate_gy_per_s
