"""Tests for the Weibull calibration path and COTS analogue profiles."""

from __future__ import annotations

import numpy as np
import pytest

from spacellm.environments import cots
from spacellm.environments.devices import POLARFIRE_LSRAM
from spacellm.environments.physics import (
    cross_section_for,
    device_from_measurements,
    fit_weibull,
)


def _synthetic_points(
    device: object,
    let: np.ndarray,
) -> np.ndarray:
    return np.asarray(cross_section_for(device, let), dtype=np.float64)  # type: ignore[arg-type]


def test_fit_weibull_recovers_known_curve() -> None:
    # Generate points from a known device, then recover its parameters.
    let = np.linspace(0.5, 60.0, 25)
    sigma = _synthetic_points(POLARFIRE_LSRAM, let)
    sigma = np.maximum(sigma, 1e-12)  # fitter needs strictly-positive σ
    fit = fit_weibull(let, sigma)
    assert fit.n_points == 25
    assert fit.r_squared > 0.95
    # Recovered curve matches the source curve closely in log space.
    assert fit.rmse_log10 < 0.3


def test_fit_weibull_rejects_short_input() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        fit_weibull(np.array([1.0, 2.0]), np.array([1e-9, 2e-9]))


def test_fit_weibull_rejects_nonpositive_sigma() -> None:
    with pytest.raises(ValueError, match="strictly positive"):
        fit_weibull(np.array([1.0, 2.0, 3.0]), np.array([1e-9, 0.0, 2e-9]))


def test_device_from_measurements_builds_usable_model() -> None:
    let = np.linspace(0.5, 60.0, 20)
    sigma = np.maximum(_synthetic_points(POLARFIRE_LSRAM, let), 1e-12)
    device = device_from_measurements(
        "my_cots_part",
        let,
        sigma,
        process_node_nm=7.0,
        citation="My beam campaign, 2026.",
    )
    assert device.name == "my_cots_part"
    assert device.verified is True
    assert device.saturation_cross_section_cm2 > 0
    # The calibrated model reproduces the input within an order of magnitude.
    predicted = np.asarray(cross_section_for(device, let), dtype=np.float64)
    ratio = predicted[-1] / sigma[-1]
    assert 0.3 < ratio < 3.0


def test_cots_profiles_are_unverified_estimates() -> None:
    assert cots.COTS_DEVICES
    for device in cots.COTS_DEVICES.values():
        assert device.verified is False
        assert device.citations
        assert "ESTIMATE" in device.citations[0]


def test_cots_profiles_have_positive_weibull_params() -> None:
    for device in cots.COTS_DEVICES.values():
        assert device.saturation_cross_section_cm2 > 0
        assert device.onset_let_mev_cm2_per_mg > 0
        assert device.width > 0
        assert device.shape > 0
