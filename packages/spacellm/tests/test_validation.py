"""Tests for ``spacellm.validation``."""

from __future__ import annotations

import numpy as np
import pytest

from spacellm.environments.physics import cross_section_for
from spacellm.types import DeviceModel
from spacellm.validation import (
    ValidationResult,
    WeibullValidationData,
    validate_against_measurements,
)
from spacellm.validation.harness import DEFAULT_ECSS_FACTOR


@pytest.fixture
def reference_device() -> DeviceModel:
    return DeviceModel(
        name="reference_28nm",
        process_node_nm=28.0,
        sensitive_volume_um3=0.05,
        saturation_cross_section_cm2=1.0e-8,
        onset_let_mev_cm2_per_mg=0.5,
        width=20.0,
        shape=1.5,
    )


def _self_consistent_data(device: DeviceModel) -> WeibullValidationData:
    """Build a 'measurement set' that exactly matches the device's Weibull
    so the validator should report a near-perfect fit."""
    let_grid = np.array([1.0, 5.0, 10.0, 25.0, 50.0, 100.0])
    measured = np.asarray(cross_section_for(device, let_grid))
    return WeibullValidationData(
        device_name=device.name,
        let_points_mev_cm2_per_mg=let_grid,
        measured_cross_section_cm2=measured,
        citation="self-consistent test fixture",
    )


def test_data_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="match in shape"):
        WeibullValidationData(
            device_name="x",
            let_points_mev_cm2_per_mg=np.array([1.0, 2.0]),
            measured_cross_section_cm2=np.array([1e-9]),
        )


def test_data_rejects_non_increasing_let() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        WeibullValidationData(
            device_name="x",
            let_points_mev_cm2_per_mg=np.array([3.0, 2.0, 1.0]),
            measured_cross_section_cm2=np.array([1e-9, 1e-9, 1e-9]),
        )


def test_data_rejects_negative_cross_section() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        WeibullValidationData(
            device_name="x",
            let_points_mev_cm2_per_mg=np.array([1.0, 2.0]),
            measured_cross_section_cm2=np.array([1e-9, -1e-9]),
        )


def test_data_rejects_too_few_points() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        WeibullValidationData(
            device_name="x",
            let_points_mev_cm2_per_mg=np.array([1.0]),
            measured_cross_section_cm2=np.array([1e-9]),
        )


def test_self_consistent_fit_is_near_perfect(reference_device: DeviceModel) -> None:
    data = _self_consistent_data(reference_device)
    result = validate_against_measurements(reference_device, data)
    assert isinstance(result, ValidationResult)
    assert result.device_name == reference_device.name
    assert result.n_points == 6
    assert result.rmse_log10 < 1e-6
    assert result.r_squared > 0.999
    assert result.max_residual_factor < 1.0001
    assert result.passes_ecss_tolerance is True


def test_bad_fit_fails_ecss(reference_device: DeviceModel) -> None:
    """Inflate the measurements by 100× — the predicted curve should
    miss them by 100×, well above the 3× ECSS envelope."""
    let_grid = np.array([1.0, 5.0, 10.0, 25.0, 50.0])
    measured = np.asarray(cross_section_for(reference_device, let_grid)) * 100.0
    data = WeibullValidationData(
        device_name=reference_device.name,
        let_points_mev_cm2_per_mg=let_grid,
        measured_cross_section_cm2=measured,
    )
    result = validate_against_measurements(reference_device, data)
    assert result.max_residual_factor > 50.0
    assert result.passes_ecss_tolerance is False


def test_loose_tolerance_passes_a_2x_offset(reference_device: DeviceModel) -> None:
    let_grid = np.array([1.0, 5.0, 10.0, 25.0, 50.0])
    measured = np.asarray(cross_section_for(reference_device, let_grid)) * 1.8
    data = WeibullValidationData(
        device_name=reference_device.name,
        let_points_mev_cm2_per_mg=let_grid,
        measured_cross_section_cm2=measured,
    )
    result = validate_against_measurements(reference_device, data, ecss_factor=2.5)
    assert result.passes_ecss_tolerance is True

    tight = validate_against_measurements(reference_device, data, ecss_factor=1.5)
    assert tight.passes_ecss_tolerance is False


def test_default_ecss_factor_is_loose() -> None:
    """The default 3× envelope is the documented first-cut tolerance."""
    assert DEFAULT_ECSS_FACTOR == 3.0


def test_validate_rejects_invalid_ecss_factor(reference_device: DeviceModel) -> None:
    data = _self_consistent_data(reference_device)
    with pytest.raises(ValueError, match="ecss_factor"):
        validate_against_measurements(reference_device, data, ecss_factor=0.5)


def test_predicted_cross_section_returned_for_plotting(reference_device: DeviceModel) -> None:
    data = _self_consistent_data(reference_device)
    result = validate_against_measurements(reference_device, data)
    assert result.predicted_cross_section_cm2.shape == (6,)
    assert np.all(result.predicted_cross_section_cm2 >= 0)
    assert np.all(
        result.predicted_cross_section_cm2 <= reference_device.saturation_cross_section_cm2 + 1e-12
    )


def test_residual_factor_per_point_returned(reference_device: DeviceModel) -> None:
    data = _self_consistent_data(reference_device)
    result = validate_against_measurements(reference_device, data)
    assert result.residual_factor_per_point.shape == (6,)
    assert np.all(result.residual_factor_per_point >= 1.0 - 1e-9)
