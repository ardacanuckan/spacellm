"""Physics primitives for SEU rate computation.

This module is the math foundation for the v0.3 physics-lite
environment. It is dependency-light by design, pure NumPy, no SpacePy
/ IRBEM yet, so users can compute Weibull cross-sections and run IRPP
integrations against any LET spectrum they bring.

References
----------
* Pickel, J.C. & Blandford, J.T. (1980). "Cosmic-ray-induced errors in
  MOS devices." IEEE TNS 27(2).
* Petersen, E.L. (1996). "Approaches to proton single-event rate
  calculations." IEEE TNS 43(2): 496–504.
* The four-parameter Weibull cross-section model is the de-facto
  standard fit for heavy-ion SEU response; see for example
  NASA/TN-2008-215459 and the consolidated Weibull tables in NEPP
  reports.

Honesty caveat
--------------
For real Weibull parameters of a specific device, consult NEPP / NSREC
publications, vendor radiation test reports, or run a beam test. The
:data:`DEMO_SRAM` device shipped here is a *placeholder* with
order-of-magnitude reasonable values; do **not** use it for mission
analysis. the physics primer lists primary
sources for real device characterisation.
"""

from __future__ import annotations

from typing import overload

import numpy as np
from numpy.typing import NDArray

from spacellm.types import DeviceModel


@overload
def weibull_cross_section(
    let: float,
    *,
    saturation_cross_section_cm2: float,
    onset_let_mev_cm2_per_mg: float,
    width: float,
    shape: float,
) -> float: ...


@overload
def weibull_cross_section(
    let: NDArray[np.floating],
    *,
    saturation_cross_section_cm2: float,
    onset_let_mev_cm2_per_mg: float,
    width: float,
    shape: float,
) -> NDArray[np.floating]: ...


def weibull_cross_section(
    let: float | NDArray[np.floating],
    *,
    saturation_cross_section_cm2: float,
    onset_let_mev_cm2_per_mg: float,
    width: float,
    shape: float,
) -> float | NDArray[np.floating]:
    """Four-parameter Weibull SEU cross-section model.

    σ(L) = σ_sat · (1 − exp(−((L − L₀) / W)ˢ))   for L > L₀
         = 0                                       otherwise

    Args:
        let: Linear energy transfer in MeV·cm²·mg⁻¹. Scalar or array.
        saturation_cross_section_cm2: Asymptotic cross-section σ_sat in cm².
        onset_let_mev_cm2_per_mg: Onset LET threshold L₀.
        width: Width parameter W.
        shape: Shape parameter s.

    Returns:
        σ(L) in cm² with the same shape as the input.

    Raises:
        ValueError: If any of the Weibull params is non-positive.
    """
    if saturation_cross_section_cm2 <= 0:
        raise ValueError("saturation_cross_section_cm2 must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    if shape <= 0:
        raise ValueError("shape must be positive")
    if onset_let_mev_cm2_per_mg < 0:
        raise ValueError("onset_let_mev_cm2_per_mg must be non-negative")

    let_arr = np.asarray(let, dtype=np.float64)
    excess = np.maximum(let_arr - onset_let_mev_cm2_per_mg, 0.0)
    sigma = saturation_cross_section_cm2 * (1.0 - np.exp(-((excess / width) ** shape)))
    if np.ndim(let) == 0:
        return float(sigma)
    return sigma


def cross_section_for(
    device: DeviceModel,
    let: float | NDArray[np.floating],
) -> float | NDArray[np.floating]:
    """Apply ``device``'s Weibull params to one LET value or an array."""
    return weibull_cross_section(
        let,
        saturation_cross_section_cm2=device.saturation_cross_section_cm2,
        onset_let_mev_cm2_per_mg=device.onset_let_mev_cm2_per_mg,
        width=device.width,
        shape=device.shape,
    )


def irpp_seu_rate_per_cell(
    device: DeviceModel,
    let_grid: NDArray[np.floating],
    flux_per_let_per_s: NDArray[np.floating],
) -> float:
    """Integrated rectangular-parallelepiped SEU rate per sensitive cell.

    R = ∫ σ(L) · dF/dL · dL

    Numerically integrated by the trapezoidal rule on the supplied grid.
    The caller is responsible for choosing a grid fine enough to capture
    the cross-section's roll-on around L₀.

    Args:
        device: Source of Weibull cross-section parameters.
        let_grid: Strictly-increasing LET grid in MeV·cm²·mg⁻¹.
        flux_per_let_per_s: Differential flux dF/dL in
            particles · cm⁻² · s⁻¹ · (MeV·cm²/mg)⁻¹ sampled at
            ``let_grid``.

    Returns:
        SEU rate per cell in upsets · cell⁻¹ · s⁻¹.

    Raises:
        ValueError: If grids are mismatched, too short, or not increasing.
    """
    grid = np.asarray(let_grid, dtype=np.float64)
    flux = np.asarray(flux_per_let_per_s, dtype=np.float64)
    if grid.shape != flux.shape:
        raise ValueError(
            f"let_grid and flux_per_let_per_s must match in shape; "
            f"got {grid.shape} vs {flux.shape}",
        )
    if grid.ndim != 1:
        raise ValueError("let_grid must be 1-D")
    if grid.size < 2:
        raise ValueError("let_grid must have at least two samples")
    if not np.all(np.diff(grid) > 0):
        raise ValueError("let_grid must be strictly increasing")

    sigma = cross_section_for(device, grid)
    integrand = np.asarray(sigma) * flux
    return float(np.trapezoid(integrand, grid))


def petersen_fom(device: DeviceModel) -> float:
    """Petersen Figure of Merit (cm² · (MeV·cm²·mg⁻¹)⁻²).

    FoM = σ_sat / L₀²

    A higher FoM means more upsets per unit incident flux. Useful for
    rapid device comparisons; not a substitute for a full IRPP
    integration but a good sanity check.

    Raises:
        ValueError: If onset LET is zero (FoM is undefined).
    """
    if device.onset_let_mev_cm2_per_mg <= 0:
        raise ValueError(
            f"Petersen FoM is undefined for L₀ ≤ 0 (got {device.onset_let_mev_cm2_per_mg})",
        )
    return device.saturation_cross_section_cm2 / (device.onset_let_mev_cm2_per_mg**2)


__all__ = [
    "cross_section_for",
    "irpp_seu_rate_per_cell",
    "petersen_fom",
    "weibull_cross_section",
]
