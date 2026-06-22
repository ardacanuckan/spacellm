"""Validation harness, compare DeviceModel predictions against
published beam-test measurements.

The radiation-effects community accepts a software simulator as
"trustworthy" when its predicted Weibull cross-section reproduces
measured beam-test points within an established tolerance band
(commonly ECSS-Q-ST-60-15C Rev.1, March 2025). This module provides
the data-class plumbing and the comparison harness so a DeviceModel
in ``spacellm.environments.devices`` can be calibrated and refit
against its primary source.

This module does **not** ship a curated database of measurements ,
that would mean redistributing third-party data tables. It provides
the API; users bring the (LET, σ) points from the cited NEPP / NSREC
PDF and run the validator.
"""

from __future__ import annotations

from spacellm.validation import datasets
from spacellm.validation.data import WeibullValidationData
from spacellm.validation.harness import ValidationResult, validate_against_measurements
from spacellm.validation.reconcile import (
    ReconciliationResult,
    reconcile_events,
    reconcile_seu_rate,
)

__all__ = [
    "ReconciliationResult",
    "ValidationResult",
    "WeibullValidationData",
    "datasets",
    "reconcile_events",
    "reconcile_seu_rate",
    "validate_against_measurements",
]
