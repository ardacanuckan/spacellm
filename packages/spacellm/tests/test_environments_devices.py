"""Tests for ``spacellm.environments.devices``.

Every device in the catalogue is built from publicly-fetched primary-source
beam-test data (see ``packages/docs/docs/beam-test-data.md``). These tests
enforce that contract.
"""

from __future__ import annotations

import numpy as np
import pytest

from spacellm.environments.devices import (
    POLARFIRE_DFF_ALL_ONE,
    POLARFIRE_DFF_ALL_ZERO,
    POLARFIRE_DFF_CB_SLOW,
    POLARFIRE_DFF_CHECKERBOARD,
    POLARFIRE_LSRAM,
    POLARFIRE_USRAM,
    REFERENCE_DEVICES,
    RT_POLARFIRE,
)
from spacellm.environments.physics import (
    cross_section_for,
    irpp_seu_rate_per_cell,
    weibull_cross_section,
)
from spacellm.types import DeviceModel

ALL_DEVICES = [
    POLARFIRE_USRAM,
    POLARFIRE_LSRAM,
    POLARFIRE_DFF_ALL_ZERO,
    POLARFIRE_DFF_ALL_ONE,
    POLARFIRE_DFF_CHECKERBOARD,
    POLARFIRE_DFF_CB_SLOW,
]


@pytest.mark.parametrize("device", ALL_DEVICES, ids=lambda d: d.name)
def test_device_is_marked_verified(device: DeviceModel) -> None:
    assert device.verified is True


@pytest.mark.parametrize("device", ALL_DEVICES, ids=lambda d: d.name)
def test_device_has_full_citation(device: DeviceModel) -> None:
    assert len(device.citations) >= 1
    assert "Microchip" in device.citations[0]
    assert "2020-02-04" in device.citations[0]


@pytest.mark.parametrize("device", ALL_DEVICES, ids=lambda d: d.name)
def test_device_weibull_params_in_sane_ranges(device: DeviceModel) -> None:
    assert device.saturation_cross_section_cm2 > 0
    assert device.onset_let_mev_cm2_per_mg >= 0
    assert device.width > 0
    assert device.shape > 0


@pytest.mark.parametrize("device", ALL_DEVICES, ids=lambda d: d.name)
def test_device_cross_section_is_finite_and_bounded(device: DeviceModel) -> None:
    sigma = weibull_cross_section(
        let=50.0,
        saturation_cross_section_cm2=device.saturation_cross_section_cm2,
        onset_let_mev_cm2_per_mg=device.onset_let_mev_cm2_per_mg,
        width=device.width,
        shape=device.shape,
    )
    assert 0 <= sigma <= device.saturation_cross_section_cm2


@pytest.mark.parametrize("device", ALL_DEVICES, ids=lambda d: d.name)
def test_device_carries_sel_threshold(device: DeviceModel) -> None:
    assert device.sel_threshold_mev_cm2_per_mg is not None
    assert device.sel_threshold_mev_cm2_per_mg > 0


@pytest.mark.parametrize("device", ALL_DEVICES, ids=lambda d: d.name)
def test_device_carries_tid_threshold(device: DeviceModel) -> None:
    assert device.tid_failure_krad_si is not None
    assert device.tid_failure_krad_si >= 100.0


def test_default_polarfire_alias_points_to_lsram() -> None:
    assert RT_POLARFIRE is POLARFIRE_LSRAM


def test_dff_checkerboard_has_highest_sigma_sat() -> None:
    """Microchip's measurement: FF_CB has the highest σ_sat in the family."""
    assert POLARFIRE_DFF_CHECKERBOARD.saturation_cross_section_cm2 == max(
        d.saturation_cross_section_cm2 for d in ALL_DEVICES
    )


def test_reference_devices_dict_lookup_consistency() -> None:
    for name, device in REFERENCE_DEVICES.items():
        assert device.name == name


def test_irpp_returns_finite_under_uniform_flux() -> None:
    let = np.linspace(1.0, 100.0, 64)
    flat_flux = np.ones_like(let)
    rate = irpp_seu_rate_per_cell(POLARFIRE_LSRAM, let, flat_flux)
    assert rate > 0
    assert np.isfinite(rate)


def test_cross_section_sweeps_for_all_devices() -> None:
    let_grid = np.linspace(1.0, 60.0, 30)
    for device in ALL_DEVICES:
        sigma = np.asarray(cross_section_for(device, let_grid))
        # Weibull is monotone non-decreasing by construction.
        assert np.all(np.diff(sigma) >= -1e-15)
