"""Compare ``DeviceModel`` predictions against beam-test measurements."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from spacellm.environments.physics import cross_section_for
from spacellm.types import DeviceModel
from spacellm.validation.data import WeibullValidationData

# A practical first-cut of the ECSS-Q-ST-60-15C "one-sided 90 % tolerance"
# expectation: the predicted cross-section should sit within 3× of every
# measurement (i.e. the residual factor, the larger of measured/predicted
# and predicted/measured, should be ≤ this bound). v0.5 will swap this
# for the formal one-sided 90 % tolerance limit on log10(σ) residuals.
DEFAULT_ECSS_FACTOR: float = 3.0


@dataclass(slots=True)
class ValidationResult:
    """Outcome of comparing a ``DeviceModel`` against measurements.

    Attributes:
        device_name: Name of the device under test.
        n_points: Number of measurement points compared.
        rmse_log10: Root-mean-square error in ``log10(σ)`` space ,
            standard for cross-sections that span orders of magnitude.
        r_squared: Coefficient of determination on ``log10(σ)`` ,
            1.0 means perfect fit; 0.0 or below means the model fits
            no better than the measurement mean.
        max_residual_factor: Maximum of ``max(measured/predicted,
            predicted/measured)`` across all points. A tight upper bound
            on how far the model is from the data.
        passes_ecss_tolerance: ``True`` if ``max_residual_factor`` is
            within the ECSS-style envelope (defaults to 3×).
        residual_factor_per_point: 1-D array of per-point residual
            factors, same length as the measurement.
        predicted_cross_section_cm2: 1-D array of predicted σ at each
            LET point, useful for plotting alongside the measurements.
    """

    device_name: str
    n_points: int
    rmse_log10: float
    r_squared: float
    max_residual_factor: float
    passes_ecss_tolerance: bool
    residual_factor_per_point: NDArray[np.floating] = field(repr=False)
    predicted_cross_section_cm2: NDArray[np.floating] = field(repr=False)


def _safe_log10(x: NDArray[np.floating]) -> NDArray[np.floating]:
    floor = 1e-30
    clamped: NDArray[np.floating] = np.maximum(x, floor)
    result: NDArray[np.floating] = np.log10(clamped)
    return result


def validate_against_measurements(
    device: DeviceModel,
    data: WeibullValidationData,
    *,
    ecss_factor: float = DEFAULT_ECSS_FACTOR,
) -> ValidationResult:
    """Score a ``DeviceModel`` against measured beam-test points.

    The model's analytical Weibull curve is evaluated at every LET in
    ``data`` and compared to the measured σ. Residuals are reported in
    log10 space (where σ varies over orders of magnitude) plus an
    explicit per-point residual factor for the ECSS-style tolerance
    check.

    Args:
        device: The model under test.
        data: Beam-test measurements pulled from a primary source.
        ecss_factor: Tolerance envelope. ``passes_ecss_tolerance`` is
            ``True`` iff the largest per-point residual factor is
            ≤ this value. Defaults to 3.0 (loose first-cut). v0.5 will
            replace this with the formal ECSS-Q-ST-60-15C statistical
            test.

    Returns:
        A :class:`ValidationResult` recording RMSE, R², max residual
        factor, and the ECSS pass flag.
    """
    if ecss_factor < 1.0:
        raise ValueError(f"ecss_factor must be ≥ 1.0, got {ecss_factor}")

    let = np.asarray(data.let_points_mev_cm2_per_mg, dtype=np.float64)
    measured = np.asarray(data.measured_cross_section_cm2, dtype=np.float64)
    predicted = np.asarray(cross_section_for(device, let), dtype=np.float64)

    log_meas = _safe_log10(measured)
    log_pred = _safe_log10(predicted)
    residuals = log_pred - log_meas
    rmse_log10 = float(np.sqrt(np.mean(residuals**2)))

    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((log_meas - log_meas.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Per-point residual factor: max(meas/pred, pred/meas), guarded against zero
    eps = 1e-30
    factor_pred_over_meas = predicted / np.maximum(measured, eps)
    factor_meas_over_pred = measured / np.maximum(predicted, eps)
    residual_factor_per_point = np.maximum(factor_pred_over_meas, factor_meas_over_pred)
    max_residual_factor = float(np.max(residual_factor_per_point))
    passes = max_residual_factor <= ecss_factor

    return ValidationResult(
        device_name=device.name,
        n_points=int(let.size),
        rmse_log10=rmse_log10,
        r_squared=r_squared,
        max_residual_factor=max_residual_factor,
        passes_ecss_tolerance=passes,
        residual_factor_per_point=residual_factor_per_point,
        predicted_cross_section_cm2=predicted,
    )


__all__ = [
    "DEFAULT_ECSS_FACTOR",
    "ValidationResult",
    "validate_against_measurements",
]
