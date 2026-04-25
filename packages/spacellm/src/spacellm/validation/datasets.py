"""Public-source beam-test measurement datasets.

Every dataset in this module is a direct, publicly-fetched primary-source
table, see ``packages/docs/docs/beam-test-data.md`` for the extraction
record. These are the calibration data the validation harness uses to
score :class:`~spacellm.types.DeviceModel` cross-sections.

Add to this module ONLY when the values have been pulled from a publicly
fetched primary source. No estimates, no synthetic data, no plot
digitization, if a number is not in the source's machine-readable
text/table, it does not appear here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from spacellm.validation.data import WeibullValidationData

# ────────────────────────────────────────────────────────────────────
# Microchip RT PolarFire MPF300, Appendix raw σ-vs-LET (TAMU 2020)
#
# Direct table read from
# https://ww1.microchip.com/downloads/aemdocuments/documents/fpga/
# ProductDocuments/SupportingCollateral/
# rt_polarfire_radiation_test_report_2020-02-04.pdf
# Appendix p.6 (DFF table) and Appendix p.6 (µSRAM/LSRAM table). 16 LET
# points are common to all six cell-type variants.
# ────────────────────────────────────────────────────────────────────

_RT_POLARFIRE_CITATION = (
    "Wang et al. 2020-02-04, Microchip RT PolarFire Radiation Test Report, "
    "Appendix raw σ-vs-LET, p.6."
)

_RT_POLARFIRE_LET_GRID: NDArray[np.floating] = np.array(
    [
        0.9,
        1.9,
        2.2,
        2.8,
        6.3,
        6.4,
        7.6,
        7.8,
        9.7,
        9.8,
        24.1,
        24.3,
        29.4,
        29.7,
        29.8,
        30.37,
    ],
)


POLARFIRE_USRAM_TAMU_2020 = WeibullValidationData(
    device_name="microchip_rt_polarfire_usram",
    let_points_mev_cm2_per_mg=_RT_POLARFIRE_LET_GRID,
    measured_cross_section_cm2=np.array(
        [
            3.65e-10,
            4.42e-10,
            5.00e-10,
            5.22e-10,
            6.39e-10,
            9.12e-10,
            8.58e-10,
            9.08e-10,
            1.20e-9,
            1.30e-9,
            2.14e-9,
            2.06e-9,
            2.67e-9,
            2.60e-9,
            5.95e-9,
            5.36e-9,
        ],
    ),
    citation=_RT_POLARFIRE_CITATION + " µSRAM column.",
)

POLARFIRE_LSRAM_TAMU_2020 = WeibullValidationData(
    device_name="microchip_rt_polarfire_lsram",
    let_points_mev_cm2_per_mg=_RT_POLARFIRE_LET_GRID,
    measured_cross_section_cm2=np.array(
        [
            5.44e-10,
            7.86e-10,
            9.03e-10,
            1.13e-9,
            1.09e-9,
            1.78e-9,
            1.83e-9,
            2.19e-9,
            2.53e-9,
            2.65e-9,
            3.73e-9,
            3.75e-9,
            4.76e-9,
            4.86e-9,
            1.09e-8,
            7.94e-9,
        ],
    ),
    citation=_RT_POLARFIRE_CITATION + " LSRAM column.",
)

POLARFIRE_DFF_ALL_ZERO_TAMU_2020 = WeibullValidationData(
    device_name="microchip_rt_polarfire_dff_all_zero",
    let_points_mev_cm2_per_mg=_RT_POLARFIRE_LET_GRID,
    measured_cross_section_cm2=np.array(
        [
            4.28571e-10,
            4.25000e-10,
            3.42951e-10,
            6.72495e-10,
            1.14226e-9,
            1.90310e-9,
            1.73551e-9,
            2.15517e-9,
            2.21607e-9,
            2.72374e-9,
            2.86338e-9,
            3.56738e-9,
            4.05186e-9,
            3.46962e-9,
            1.05556e-8,
            7.88177e-9,
        ],
    ),
    citation=_RT_POLARFIRE_CITATION + " FF0 (all-0) column.",
)

POLARFIRE_DFF_ALL_ONE_TAMU_2020 = WeibullValidationData(
    device_name="microchip_rt_polarfire_dff_all_one",
    let_points_mev_cm2_per_mg=_RT_POLARFIRE_LET_GRID,
    measured_cross_section_cm2=np.array(
        [
            2.78571e-10,
            5.25000e-10,
            4.91563e-10,
            5.52407e-10,
            8.30737e-10,
            1.71132e-9,
            1.89552e-9,
            1.90965e-9,
            2.14681e-9,
            2.04280e-9,
            4.36324e-9,
            3.99207e-9,
            5.15166e-9,
            6.47037e-9,
            1.16667e-8,
            1.03448e-8,
        ],
    ),
    citation=_RT_POLARFIRE_CITATION + " FF1 (all-1) column.",
)

POLARFIRE_DFF_CHECKERBOARD_TAMU_2020 = WeibullValidationData(
    device_name="microchip_rt_polarfire_dff_checkerboard",
    let_points_mev_cm2_per_mg=_RT_POLARFIRE_LET_GRID,
    measured_cross_section_cm2=np.array(
        [
            6.14286e-10,
            7.75000e-10,
            1.10888e-9,
            1.39303e-9,
            2.54413e-9,
            4.48483e-9,
            4.33263e-9,
            5.23789e-9,
            6.50970e-9,
            8.31712e-9,
            3.02018e-8,
            3.08041e-8,
            3.99398e-8,
            4.38860e-8,
            1.00000e-7,
            8.02956e-8,
        ],
    ),
    citation=_RT_POLARFIRE_CITATION + " FF_CB column.",
)

POLARFIRE_DFF_CB_SLOW_TAMU_2020 = WeibullValidationData(
    device_name="microchip_rt_polarfire_dff_cb_slow",
    let_points_mev_cm2_per_mg=_RT_POLARFIRE_LET_GRID,
    measured_cross_section_cm2=np.array(
        [
            5.42857e-10,
            6.25000e-10,
            5.02995e-10,
            7.20530e-10,
            2.49221e-9,
            3.79146e-9,
            3.17562e-9,
            4.11938e-9,
            4.50139e-9,
            3.98833e-9,
            1.55440e-8,
            1.38732e-8,
            1.77124e-8,
            1.80045e-8,
            3.97222e-8,
            4.66749e-8,
        ],
    ),
    citation=_RT_POLARFIRE_CITATION + " FF_CB_slow column.",
)


REFERENCE_DATASETS: dict[str, WeibullValidationData] = {
    "microchip_rt_polarfire_usram": POLARFIRE_USRAM_TAMU_2020,
    "microchip_rt_polarfire_lsram": POLARFIRE_LSRAM_TAMU_2020,
    "microchip_rt_polarfire_dff_all_zero": POLARFIRE_DFF_ALL_ZERO_TAMU_2020,
    "microchip_rt_polarfire_dff_all_one": POLARFIRE_DFF_ALL_ONE_TAMU_2020,
    "microchip_rt_polarfire_dff_checkerboard": POLARFIRE_DFF_CHECKERBOARD_TAMU_2020,
    "microchip_rt_polarfire_dff_cb_slow": POLARFIRE_DFF_CB_SLOW_TAMU_2020,
}
"""``device_name → WeibullValidationData`` lookup, keyed identically to
:data:`spacellm.environments.devices.REFERENCE_DEVICES` so the validator
can pair Weibull params and measurements automatically."""


__all__ = [
    "POLARFIRE_DFF_ALL_ONE_TAMU_2020",
    "POLARFIRE_DFF_ALL_ZERO_TAMU_2020",
    "POLARFIRE_DFF_CB_SLOW_TAMU_2020",
    "POLARFIRE_DFF_CHECKERBOARD_TAMU_2020",
    "POLARFIRE_LSRAM_TAMU_2020",
    "POLARFIRE_USRAM_TAMU_2020",
    "REFERENCE_DATASETS",
]
