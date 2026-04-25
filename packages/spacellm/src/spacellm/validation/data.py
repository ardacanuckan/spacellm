"""Measurement-data containers for validation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class WeibullValidationData:
    """Measured (LET, cross-section) data points for one device.

    Built from a primary beam-test paper. Used by
    :func:`spacellm.validation.harness.validate_against_measurements` to
    score a :class:`~spacellm.types.DeviceModel`'s predicted Weibull
    curve against the actual beam-measured points.

    Attributes:
        device_name: Name of the device the points belong to. Match it
            to the corresponding ``DeviceModel.name``.
        let_points_mev_cm2_per_mg: 1-D strictly-increasing LET values.
        measured_cross_section_cm2: 1-D measured σ at each LET point.
        measurement_uncertainty: 1-D 1-σ measurement uncertainties on
            ``measured_cross_section_cm2``. If absent the harness uses
            a flat 30 % default per radiation test convention.
        citation: Bibliographic string the data was extracted from.
    """

    device_name: str
    let_points_mev_cm2_per_mg: NDArray[np.floating] = field(repr=False)
    measured_cross_section_cm2: NDArray[np.floating] = field(repr=False)
    measurement_uncertainty: NDArray[np.floating] | None = field(default=None, repr=False)
    citation: str = ""

    def __post_init__(self) -> None:
        if self.let_points_mev_cm2_per_mg.shape != self.measured_cross_section_cm2.shape:
            raise ValueError(
                f"LET and σ arrays must match in shape; got "
                f"{self.let_points_mev_cm2_per_mg.shape} vs "
                f"{self.measured_cross_section_cm2.shape}",
            )
        if self.let_points_mev_cm2_per_mg.ndim != 1:
            raise ValueError("LET array must be 1-D")
        if self.let_points_mev_cm2_per_mg.size < 2:
            raise ValueError("need at least 2 measurement points")
        if not np.all(np.diff(self.let_points_mev_cm2_per_mg) > 0):
            raise ValueError("LET points must be strictly increasing")
        if np.any(self.measured_cross_section_cm2 < 0):
            raise ValueError("measured cross-sections must be non-negative")
        if self.measurement_uncertainty is not None:
            if self.measurement_uncertainty.shape != self.let_points_mev_cm2_per_mg.shape:
                raise ValueError("uncertainty array must match LET shape")
            if np.any(self.measurement_uncertainty < 0):
                raise ValueError("uncertainties must be non-negative")


__all__ = ["WeibullValidationData"]
