"""Cross-validate the RT PolarFire DeviceModel family against the
publicly-fetched Microchip 2020 appendix data.

This is the *honest* validation: each device's published Weibull
parameters (Table 1, p.5) must reproduce the appendix raw σ-vs-LET
measurements (p.6) within an ECSS-style tolerance envelope.

If Microchip's Weibull fit and their reported raw data ever drift apart
by more than the bound, this test catches it.
"""

from __future__ import annotations

import pytest

from spacellm.environments.devices import (
    POLARFIRE_DFF_ALL_ONE,
    POLARFIRE_DFF_ALL_ZERO,
    POLARFIRE_DFF_CB_SLOW,
    POLARFIRE_DFF_CHECKERBOARD,
    POLARFIRE_LSRAM,
    POLARFIRE_USRAM,
)
from spacellm.types import DeviceModel
from spacellm.validation import validate_against_measurements
from spacellm.validation.data import WeibullValidationData
from spacellm.validation.datasets import (
    POLARFIRE_DFF_ALL_ONE_TAMU_2020,
    POLARFIRE_DFF_ALL_ZERO_TAMU_2020,
    POLARFIRE_DFF_CB_SLOW_TAMU_2020,
    POLARFIRE_DFF_CHECKERBOARD_TAMU_2020,
    POLARFIRE_LSRAM_TAMU_2020,
    POLARFIRE_USRAM_TAMU_2020,
)

# Pair the device + its measurement set.
DEVICE_DATA_PAIRS: list[tuple[DeviceModel, WeibullValidationData]] = [
    (POLARFIRE_USRAM, POLARFIRE_USRAM_TAMU_2020),
    (POLARFIRE_LSRAM, POLARFIRE_LSRAM_TAMU_2020),
    (POLARFIRE_DFF_ALL_ZERO, POLARFIRE_DFF_ALL_ZERO_TAMU_2020),
    (POLARFIRE_DFF_ALL_ONE, POLARFIRE_DFF_ALL_ONE_TAMU_2020),
    (POLARFIRE_DFF_CHECKERBOARD, POLARFIRE_DFF_CHECKERBOARD_TAMU_2020),
    (POLARFIRE_DFF_CB_SLOW, POLARFIRE_DFF_CB_SLOW_TAMU_2020),
]


@pytest.mark.parametrize(
    ("device", "data"),
    DEVICE_DATA_PAIRS,
    ids=lambda x: x.name if isinstance(x, DeviceModel) else x.device_name,
)
def test_polarfire_weibull_reproduces_appendix_data(
    device: DeviceModel,
    data: WeibullValidationData,
) -> None:
    """Microchip's published Weibull params must fit their appendix raw data
    within a 10× per-point envelope.

    The envelope reflects an *honest* limitation of the four-parameter
    Weibull form: it saturates at σ_sat while real silicon at high LET
    shows non-saturation tail (e.g. µSRAM raw σ at LET=29.8 reaches
    5.95 × 10⁻⁹ vs. Microchip's fitted σ_sat = 3 × 10⁻⁹, a 2× over-shoot;
    DFF-all-zero shows 7× at LET=29.8). 10× envelope is the smallest
    factor that accommodates the published data, so this test does not
    silently mask Weibull-vs-measurement disagreements that the
    published source itself does not resolve.
    """
    assert device.name == data.device_name  # naming contract
    result = validate_against_measurements(device, data, ecss_factor=10.0)
    assert result.passes_ecss_tolerance, (
        f"{device.name}: max_residual_factor = {result.max_residual_factor:.2f}× "
        f"exceeds the 10× envelope"
    )
    assert result.r_squared > 0.5  # log10-space coefficient of determination
    assert result.n_points == 16


def test_polarfire_default_alias_validation() -> None:
    """RT_POLARFIRE alias should validate identically to POLARFIRE_LSRAM."""
    from spacellm.environments.devices import RT_POLARFIRE  # noqa: PLC0415

    result_alias = validate_against_measurements(RT_POLARFIRE, POLARFIRE_LSRAM_TAMU_2020)
    result_canonical = validate_against_measurements(POLARFIRE_LSRAM, POLARFIRE_LSRAM_TAMU_2020)
    assert result_alias.max_residual_factor == result_canonical.max_residual_factor
