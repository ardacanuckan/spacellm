"""Tests for ``spacellm.environments.physics``."""

from __future__ import annotations

import numpy as np
import pytest

from spacellm.environments import physics
from spacellm.environments.devices import POLARFIRE_LSRAM
from spacellm.environments.physics import (
    cross_section_for,
    irpp_seu_rate_per_cell,
    petersen_fom,
    weibull_cross_section,
)
from spacellm.types import DeviceModel


def test_weibull_zero_below_onset() -> None:
    sigma = weibull_cross_section(
        let=0.1,
        saturation_cross_section_cm2=1e-8,
        onset_let_mev_cm2_per_mg=1.0,
        width=20.0,
        shape=1.5,
    )
    assert sigma == 0.0


def test_weibull_at_onset_is_zero() -> None:
    sigma = weibull_cross_section(
        let=1.0,
        saturation_cross_section_cm2=1e-8,
        onset_let_mev_cm2_per_mg=1.0,
        width=20.0,
        shape=1.5,
    )
    assert sigma == 0.0


def test_weibull_approaches_saturation() -> None:
    sigma = weibull_cross_section(
        let=200.0,  # well above L0 + a few W
        saturation_cross_section_cm2=1e-8,
        onset_let_mev_cm2_per_mg=1.0,
        width=20.0,
        shape=1.5,
    )
    assert sigma == pytest.approx(1e-8, rel=1e-3)


def test_weibull_monotone_increasing_above_onset() -> None:
    grid = np.linspace(2.0, 100.0, 50)
    sigma = weibull_cross_section(
        let=grid,
        saturation_cross_section_cm2=1e-8,
        onset_let_mev_cm2_per_mg=1.0,
        width=20.0,
        shape=1.5,
    )
    assert np.all(np.diff(sigma) >= 0)


def test_weibull_array_shape_preserved() -> None:
    grid = np.linspace(0.0, 50.0, 8)
    sigma = weibull_cross_section(
        let=grid,
        saturation_cross_section_cm2=1e-8,
        onset_let_mev_cm2_per_mg=1.0,
        width=20.0,
        shape=1.5,
    )
    assert isinstance(sigma, np.ndarray)
    assert sigma.shape == grid.shape


def test_weibull_rejects_invalid_params() -> None:
    with pytest.raises(ValueError, match="saturation"):
        weibull_cross_section(
            let=10.0,
            saturation_cross_section_cm2=-1.0,
            onset_let_mev_cm2_per_mg=1.0,
            width=20.0,
            shape=1.5,
        )
    with pytest.raises(ValueError, match="width"):
        weibull_cross_section(
            let=10.0,
            saturation_cross_section_cm2=1e-8,
            onset_let_mev_cm2_per_mg=1.0,
            width=0.0,
            shape=1.5,
        )
    with pytest.raises(ValueError, match="shape"):
        weibull_cross_section(
            let=10.0,
            saturation_cross_section_cm2=1e-8,
            onset_let_mev_cm2_per_mg=1.0,
            width=20.0,
            shape=-1.0,
        )
    with pytest.raises(ValueError, match="onset"):
        weibull_cross_section(
            let=10.0,
            saturation_cross_section_cm2=1e-8,
            onset_let_mev_cm2_per_mg=-1.0,
            width=20.0,
            shape=1.5,
        )


def test_cross_section_for_polarfire_is_finite() -> None:
    grid = np.linspace(0.0, 100.0, 20)
    sigma = cross_section_for(POLARFIRE_LSRAM, grid)
    assert isinstance(sigma, np.ndarray)
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma >= 0)
    assert np.all(sigma <= POLARFIRE_LSRAM.saturation_cross_section_cm2 + 1e-12)


def test_irpp_zero_flux_returns_zero() -> None:
    let_grid = np.linspace(1.0, 100.0, 64)
    flux = np.zeros_like(let_grid)
    rate = irpp_seu_rate_per_cell(POLARFIRE_LSRAM, let_grid, flux)
    assert rate == 0.0


def test_irpp_high_flux_produces_positive_rate() -> None:
    let_grid = np.linspace(1.0, 100.0, 64)
    flux = np.ones_like(let_grid)  # 1 particle / cm² / s / (MeV·cm²/mg)
    rate = irpp_seu_rate_per_cell(POLARFIRE_LSRAM, let_grid, flux)
    assert rate > 0


def test_irpp_validates_grid() -> None:
    with pytest.raises(ValueError, match="match in shape"):
        irpp_seu_rate_per_cell(
            POLARFIRE_LSRAM,
            np.array([1.0, 2.0, 3.0]),
            np.array([1.0, 2.0]),
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        irpp_seu_rate_per_cell(
            POLARFIRE_LSRAM,
            np.array([3.0, 2.0, 1.0]),
            np.array([1.0, 1.0, 1.0]),
        )
    with pytest.raises(ValueError, match="at least two"):
        irpp_seu_rate_per_cell(
            POLARFIRE_LSRAM,
            np.array([1.0]),
            np.array([1.0]),
        )
    with pytest.raises(ValueError, match="1-D"):
        irpp_seu_rate_per_cell(
            POLARFIRE_LSRAM,
            np.array([[1.0, 2.0]]),
            np.array([[1.0, 1.0]]),
        )


def test_petersen_fom_positive_for_polarfire() -> None:
    fom = petersen_fom(POLARFIRE_LSRAM)
    assert fom > 0
    expected = POLARFIRE_LSRAM.saturation_cross_section_cm2 / (
        POLARFIRE_LSRAM.onset_let_mev_cm2_per_mg**2
    )
    assert fom == pytest.approx(expected)


def test_petersen_fom_undefined_for_zero_onset() -> None:
    bad = DeviceModel(
        name="bad",
        process_node_nm=28.0,
        sensitive_volume_um3=0.05,
        saturation_cross_section_cm2=1e-8,
        onset_let_mev_cm2_per_mg=0.0,
        width=20.0,
        shape=1.5,
    )
    with pytest.raises(ValueError, match="undefined"):
        petersen_fom(bad)


def test_physics_namespace_exported_from_environments() -> None:
    """The physics module must be reachable via ``spacellm.environments.physics``."""
    assert physics.weibull_cross_section is weibull_cross_section
