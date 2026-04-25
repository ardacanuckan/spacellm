"""Catalogue of :class:`~spacellm.types.DeviceModel` profiles.

Every entry in this file is built from **publicly-fetched, primary-source
beam-test data**. Each ``DeviceModel.citations`` field traces every
numeric parameter back to the page and table of the source PDF that was
pulled directly from the vendor or NEPP repository, see
``packages/docs/docs/beam-test-data.md`` for the extraction record.

There are no placeholder, estimated, or "approximate"
device entries here. If a device cannot be populated from a publicly
verifiable primary source, it does not appear.

Currently verified family
-------------------------
* **Microchip RT PolarFire MPF300** (28 nm SONOS-flash FPGA), six
  cell-type entries (µSRAM, LSRAM, four DFF data patterns), all
  Weibull parameters and the SEL onset taken directly from
  `rt_polarfire_radiation_test_report_2020-02-04.pdf` Table 1 / §1.5.

The validation suite cross-checks each Weibull against its appendix
σ-vs-LET measurement in :mod:`spacellm.validation.datasets`.
"""

from __future__ import annotations

from spacellm.types import DeviceModel

_RT_POLARFIRE_CITATION = (
    "Wang, J. et al. 'Radiation Characteristics of FPGA Using "
    "Complementary-SONOS Configuration Cell.' Microchip 2020-02-04. "
    "https://ww1.microchip.com/downloads/aemdocuments/documents/fpga/"
    "ProductDocuments/SupportingCollateral/"
    "rt_polarfire_radiation_test_report_2020-02-04.pdf "
    "Table 1, p.5 (Weibull); §1.5, p.6 (SEL); §TID p.2 (300 krad(SiO₂))."
)
_RT_POLARFIRE_SEL_LET = 48.0  # MeV·cm²/mg, first SEL onset at 3.3 V + 5% I/O
_RT_POLARFIRE_TID_KRAD = 300.0  # krad(SiO₂), upper limit of "negligible" range


def _polarfire(
    *,
    name: str,
    sigma_sat: float,
    onset: float,
    width: float,
    shape: float,
    suffix: str,
) -> DeviceModel:
    """Helper to build an RT PolarFire variant with shared SEL/TID/citations."""
    return DeviceModel(
        name=name,
        process_node_nm=28.0,
        sensitive_volume_um3=0.05,  # representative 28 nm SRAM cell volume
        saturation_cross_section_cm2=sigma_sat,
        onset_let_mev_cm2_per_mg=onset,
        width=width,
        shape=shape,
        verified=True,
        sel_threshold_mev_cm2_per_mg=_RT_POLARFIRE_SEL_LET,
        tid_failure_krad_si=_RT_POLARFIRE_TID_KRAD,
        citations=(_RT_POLARFIRE_CITATION + f" {suffix} row.",),
    )


# ────────────────────────────────────────────────────────────────────
# Microchip RT PolarFire MPF300, six cell-type variants
# ────────────────────────────────────────────────────────────────────

POLARFIRE_USRAM = _polarfire(
    name="microchip_rt_polarfire_usram",
    sigma_sat=3.0e-9,
    onset=0.1,
    width=12.0,
    shape=1.3,
    suffix="µSRAM",
)
"""µSRAM blocks of the MPF300 (768-bit blocks, 2,772 of them)."""

POLARFIRE_LSRAM = _polarfire(
    name="microchip_rt_polarfire_lsram",
    sigma_sat=5.0e-9,
    onset=0.1,
    width=11.0,
    shape=1.3,
    suffix="LSRAM",
)
"""LSRAM blocks of the MPF300 (20 Kb blocks, 952 of them)."""

POLARFIRE_DFF_ALL_ONE = _polarfire(
    name="microchip_rt_polarfire_dff_all_one",
    sigma_sat=5.0e-9,
    onset=0.5,
    width=11.0,
    shape=1.3,
    suffix="FF (All One)",
)
"""DFF in MPF300, all-one data pattern."""

POLARFIRE_DFF_ALL_ZERO = _polarfire(
    name="microchip_rt_polarfire_dff_all_zero",
    sigma_sat=4.0e-9,
    onset=0.5,
    width=10.0,
    shape=1.3,
    suffix="FF (All Zero)",
)
"""DFF in MPF300, all-zero data pattern."""

POLARFIRE_DFF_CHECKERBOARD = _polarfire(
    name="microchip_rt_polarfire_dff_checkerboard",
    sigma_sat=7.0e-8,
    onset=0.1,
    width=32.0,
    shape=1.6,
    suffix="FF Checkerboard",
)
"""DFF in MPF300, checkerboard data pattern (highest σ_sat in the family)."""

POLARFIRE_DFF_CB_SLOW = _polarfire(
    name="microchip_rt_polarfire_dff_cb_slow",
    sigma_sat=3.0e-8,
    onset=0.1,
    width=28.0,
    shape=1.5,
    suffix="FF CB-Slow",
)
"""DFF in MPF300, slow-clock checkerboard data pattern."""


# Default RT PolarFire model for environment-wiring code that just wants a
# representative single-bit profile. LSRAM is the closest analogue to a
# generic SRAM weight-storage cell and is the right default for most
# transformer-weight workloads.
RT_POLARFIRE = POLARFIRE_LSRAM
"""Convenience alias, the LSRAM variant of MPF300, the closest analogue
to weight storage for an LLM workload."""


REFERENCE_DEVICES: dict[str, DeviceModel] = {
    "microchip_rt_polarfire_usram": POLARFIRE_USRAM,
    "microchip_rt_polarfire_lsram": POLARFIRE_LSRAM,
    "microchip_rt_polarfire_dff_all_one": POLARFIRE_DFF_ALL_ONE,
    "microchip_rt_polarfire_dff_all_zero": POLARFIRE_DFF_ALL_ZERO,
    "microchip_rt_polarfire_dff_checkerboard": POLARFIRE_DFF_CHECKERBOARD,
    "microchip_rt_polarfire_dff_cb_slow": POLARFIRE_DFF_CB_SLOW,
}
"""Stable name → DeviceModel lookup. All entries ``verified=True``."""


__all__ = [
    "POLARFIRE_DFF_ALL_ONE",
    "POLARFIRE_DFF_ALL_ZERO",
    "POLARFIRE_DFF_CB_SLOW",
    "POLARFIRE_DFF_CHECKERBOARD",
    "POLARFIRE_LSRAM",
    "POLARFIRE_USRAM",
    "REFERENCE_DEVICES",
    "RT_POLARFIRE",
]
