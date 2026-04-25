"""Reference orbit profiles for physics-calibrated environments.

Each :class:`OrbitProfile` captures the LET spectrum, proton flux, and
TID rate for a representative orbit segment. Numeric dose rates trace
to primary sources listed in ``packages/docs/docs/physics-primer.md``;
the LET spectra are first-pass *parametric* approximations calibrated
to the cited integrals. v0.5 will replace these with real AE-9 / AP-9
/ CREME-MC outputs once those bridges are wired in.

Citation discipline
-------------------
Every numeric value here traces to a primary source recorded in the
profile's ``citations`` field. Where a cited paper supplies only a
dose rate (and not the full LET spectrum), the spectrum is built
parametrically by :func:`gcr_like_spectrum` and the profile's
``description`` says so explicitly. Mission-critical analyses must
substitute real spectra.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class OrbitProfile:
    """A characterised orbital radiation profile.

    Attributes:
        name: Stable identifier, e.g. ``"LEO_ISS_NOMINAL"``.
        description: Human-readable summary, including caveats about
            whether the LET spectrum is empirical or parametric.
        altitude_km: Approximate orbital altitude.
        inclination_deg: Approximate inclination (irrelevant for
            interplanetary segments, set to 0 there).
        let_grid_mev_cm2_per_mg: 1-D strictly-increasing LET grid.
        flux_per_let_per_s: ``dF/dL`` at each LET point in
            particles · cm⁻² · s⁻¹ · (MeV·cm²·mg⁻¹)⁻¹.
        proton_flux_above_10mev_per_cm2_per_s: Aggregate proton flux
            above 10 MeV. Used by proton-direct-ionisation estimates
            in v0.5.
        tid_rate_gy_per_s: Total ionising dose rate, in Gy(Si) · s⁻¹.
        saa_fraction: Fraction of the orbit spent inside the South
            Atlantic Anomaly (LEO only; 0 elsewhere).
        citations: Tuple of bibliographic strings backing the numbers.
    """

    name: str
    description: str
    altitude_km: float
    inclination_deg: float
    let_grid_mev_cm2_per_mg: NDArray[np.floating] = field(repr=False)
    flux_per_let_per_s: NDArray[np.floating] = field(repr=False)
    proton_flux_above_10mev_per_cm2_per_s: float
    tid_rate_gy_per_s: float
    saa_fraction: float = 0.0
    citations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.let_grid_mev_cm2_per_mg.shape != self.flux_per_let_per_s.shape:
            raise ValueError(
                f"LET grid and flux must match in shape; got "
                f"{self.let_grid_mev_cm2_per_mg.shape} vs {self.flux_per_let_per_s.shape}",
            )
        if self.let_grid_mev_cm2_per_mg.ndim != 1:
            raise ValueError("LET grid must be 1-D")
        if self.let_grid_mev_cm2_per_mg.size < 2:
            raise ValueError("LET grid must contain at least two samples")
        if not np.all(np.diff(self.let_grid_mev_cm2_per_mg) > 0):
            raise ValueError("LET grid must be strictly increasing")
        if not 0.0 <= self.saa_fraction <= 1.0:
            raise ValueError(f"saa_fraction must be in [0, 1], got {self.saa_fraction}")


def gcr_like_spectrum(
    *,
    integral_flux: float,
    let_min: float = 0.05,
    let_max: float = 150.0,
    n_points: int = 64,
    decay_let: float = 25.0,
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    """Build a parametric GCR-like LET spectrum.

    The shape is an exponential decay in LET with a configurable scale
    length; the result is rescaled so that
    ``∫ dF/dL · dL = integral_flux``. This is *not* a physically faithful
    GCR model, it is a smooth, monotone-in-rank stand-in suitable for
    illustrative profiles. Real CREME-MC / AE-9 spectra land in v0.5.

    Args:
        integral_flux: Target integral flux in particles · cm⁻² · s⁻¹
            (i.e. the area under the returned dF/dL curve).
        let_min: Minimum LET in MeV·cm²·mg⁻¹ (geometric lower edge).
        let_max: Maximum LET (geometric upper edge).
        n_points: Number of grid samples; geometric spacing.
        decay_let: Exponential scale length in MeV·cm²·mg⁻¹.

    Returns:
        ``(let_grid, dF_per_L)``, both 1-D arrays of length ``n_points``.
    """
    if integral_flux <= 0:
        raise ValueError("integral_flux must be positive")
    if let_min <= 0 or let_max <= let_min:
        raise ValueError("0 < let_min < let_max required")
    if n_points < 2:
        raise ValueError("n_points must be at least 2")
    if decay_let <= 0:
        raise ValueError("decay_let must be positive")

    grid = np.geomspace(let_min, let_max, n_points)
    raw = np.exp(-grid / decay_let)
    norm = float(np.trapezoid(raw, grid))
    return grid, raw * integral_flux / norm


# ────────────────────────────────────────────────────────────────────
# Reference profiles
# ────────────────────────────────────────────────────────────────────

# All TID rates below are scaled from cited µGy/day measurements:
#   1 µGy/day = 1e-6 / 86400 Gy/s ≈ 1.157e-11 Gy/s.

_LEO_ISS_GRID, _LEO_ISS_FLUX = gcr_like_spectrum(
    integral_flux=1.0e-3,  # particles · cm⁻² · s⁻¹ (parametric, calibrated to dose rate)
    decay_let=15.0,
)

LEO_ISS_NOMINAL = OrbitProfile(
    name="LEO_ISS_NOMINAL",
    description=(
        "ISS-like LEO at ~400 km, 51.6° inclination. GCR-dominated during quiet sun "
        "(no solar particle event). The LET spectrum is a v0.3 parametric "
        "approximation calibrated to the cited daily dose; replace with AE-9/AP-9 + "
        "CREME-MC for mission analysis."
    ),
    altitude_km=400.0,
    inclination_deg=51.6,
    let_grid_mev_cm2_per_mg=_LEO_ISS_GRID,
    flux_per_let_per_s=_LEO_ISS_FLUX,
    proton_flux_above_10mev_per_cm2_per_s=1.0,
    tid_rate_gy_per_s=39.7e-6 / 86400.0,  # ≈ 4.59e-10 Gy/s
    saa_fraction=0.15,
    citations=(
        "Narici, L. et al. (2015). 'Radiation environment on the International Space Station.' "
        "Scientific Reports.",
    ),
)


_GEO_GRID, _GEO_FLUX = gcr_like_spectrum(
    integral_flux=4.0e-3,  # higher than LEO ISS, no atmospheric/geomagnetic shielding
    decay_let=30.0,
)

GEO_QUIET = OrbitProfile(
    name="GEO_QUIET",
    description=(
        "Geosynchronous orbit, quiet sun. GCR-dominated; no trapped electrons in this "
        "first-pass spectrum. v0.5 will add the AE-9 trapped electron contribution. "
        "Dose rate is an order-of-magnitude estimate consistent with NASA/SP-2008-565."
    ),
    altitude_km=35786.0,
    inclination_deg=0.0,
    let_grid_mev_cm2_per_mg=_GEO_GRID,
    flux_per_let_per_s=_GEO_FLUX,
    proton_flux_above_10mev_per_cm2_per_s=4.0,
    tid_rate_gy_per_s=1.0e-6 / 86400.0 * 50.0,  # ~ 50 µGy/day order-of-magnitude
    saa_fraction=0.0,
    citations=("NASA/SP-2008-565 (radiation environment overview).",),
)


_MARS_GRID, _MARS_FLUX = gcr_like_spectrum(
    integral_flux=8.0e-3,
    decay_let=40.0,  # heavier-ion tail relative to LEO
)

MARS_TRANSIT = OrbitProfile(
    name="MARS_TRANSIT",
    description=(
        "Earth–Mars transit cruise, ~210 day mission, solar minimum. LET tail is "
        "weighted toward heavy ions (Z>1). Dose rate is from MSL/RAD; the LET "
        "spectrum is parametric, replaceable with HZETRN / CREME outputs."
    ),
    altitude_km=0.0,  # interplanetary
    inclination_deg=0.0,
    let_grid_mev_cm2_per_mg=_MARS_GRID,
    flux_per_let_per_s=_MARS_FLUX,
    proton_flux_above_10mev_per_cm2_per_s=10.0,
    tid_rate_gy_per_s=458e-6 / 86400.0,  # ≈ 5.30e-9 Gy/s
    saa_fraction=0.0,
    citations=(
        "Zeitlin, C. et al. (2013). 'Measurements of energetic particle radiation in transit "
        "to Mars on the Mars Science Laboratory.' Science.",
    ),
)


REFERENCE_ORBITS: dict[str, OrbitProfile] = {
    "LEO_ISS_NOMINAL": LEO_ISS_NOMINAL,
    "GEO_QUIET": GEO_QUIET,
    "MARS_TRANSIT": MARS_TRANSIT,
}
"""Stable name → profile lookup, useful for CLI / config-driven runs."""


__all__ = [
    "GEO_QUIET",
    "LEO_ISS_NOMINAL",
    "MARS_TRANSIT",
    "REFERENCE_ORBITS",
    "OrbitProfile",
    "gcr_like_spectrum",
]
