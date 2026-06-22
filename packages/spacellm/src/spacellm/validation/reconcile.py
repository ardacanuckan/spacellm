"""Reconcile predicted SEU rates against on-orbit telemetry.

Closes the predict/fly/reconcile loop:

    predict (ground)  →  fly  →  reconcile (post-flight)

The ground-side simulator predicts an SEU rate; the spacecraft logs actual
upset events into the same :class:`~spacellm.types.FaultEvent` / ``RunDB``
schema. :func:`reconcile_seu_rate` tests whether the observed upset count
is statistically consistent with the prediction. A consistent result
flight-validates the model for that orbit/device; an inconsistent one
flags a discrepancy to investigate.

Statistics
----------
Upsets over a fixed exposure are a Poisson process with mean
``μ = rate · n_bits · exposure_s``. Given ``k`` observed events:

* ``chi_square = (k − μ)² / μ`` is the one-bin Pearson statistic.
* The two-sided Poisson p-value is ``min(1, 2·min(P(X≤k), P(X≥k)))``,
  evaluated exactly by summation for small means and via the normal
  approximation ``erfc(|z|/√2)`` for large ones, no SciPy dependency.
* ``within_envelope`` is the operator-friendly check: the observed/expected
  ratio sits inside a factor band (default 3×), the same spirit as the
  ECSS envelope used for cross-section validation.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

from spacellm.types import FaultEvent

_EXACT_MAX_K = 1000
_EXACT_MAX_MU = 1000.0


@dataclass(frozen=True, slots=True)
class ReconciliationResult:
    """Outcome of comparing predicted vs observed on-orbit upsets.

    Attributes:
        expected_events: Model-predicted mean upset count ``μ``.
        observed_events: Actual on-orbit upset count ``k``.
        ratio: ``observed / expected`` (``inf`` if expected is 0 and
            something was observed).
        chi_square: One-bin Pearson statistic ``(k − μ)² / μ``.
        p_value: Two-sided Poisson p-value.
        within_envelope: ``True`` if the ratio is inside the factor band.
        consistent: ``True`` if ``p_value ≥ alpha`` (the headline verdict).
        envelope_factor: The factor band used.
        alpha: The significance threshold used.
    """

    expected_events: float
    observed_events: int
    ratio: float
    chi_square: float
    p_value: float
    within_envelope: bool
    consistent: bool
    envelope_factor: float
    alpha: float

    def summary(self) -> str:
        verdict = "CONSISTENT" if self.consistent else "DIVERGENT"
        return (
            f"SpaceLLM reconciliation: {verdict}\n"
            f"  expected upsets : {self.expected_events:.2f}\n"
            f"  observed upsets : {self.observed_events}\n"
            f"  ratio           : {self.ratio:.3f} "
            f"(envelope {1 / self.envelope_factor:.2f}–{self.envelope_factor:.2f}×: "
            f"{'in' if self.within_envelope else 'out'})\n"
            f"  chi-square      : {self.chi_square:.3f}\n"
            f"  p-value         : {self.p_value:.4f} (alpha {self.alpha})"
        )


def _poisson_log_pmf(i: int, mu: float) -> float:
    return -mu + i * math.log(mu) - math.lgamma(i + 1)


def _poisson_cdf(k: int, mu: float) -> float:
    """``P(X ≤ k)`` for ``X ~ Poisson(mu)``, exact summation."""
    total = 0.0
    for i in range(k + 1):
        total += math.exp(_poisson_log_pmf(i, mu))
    return min(total, 1.0)


def _two_sided_p_value(k: int, mu: float) -> float:
    if mu <= 0.0:
        return 1.0 if k == 0 else 0.0
    if k <= _EXACT_MAX_K and mu <= _EXACT_MAX_MU:
        p_le = _poisson_cdf(k, mu)
        p_ge = 1.0 - (_poisson_cdf(k - 1, mu) if k > 0 else 0.0)
        return min(1.0, 2.0 * min(p_le, p_ge))
    # Normal approximation for large counts.
    z = (k - mu) / math.sqrt(mu)
    return math.erfc(abs(z) / math.sqrt(2.0))


def reconcile_seu_rate(
    predicted_rate_per_bit_per_s: float,
    *,
    n_bits: int,
    exposure_s: float,
    observed_events: int,
    envelope_factor: float = 3.0,
    alpha: float = 0.05,
) -> ReconciliationResult:
    """Test observed on-orbit upset counts against a predicted SEU rate.

    Args:
        predicted_rate_per_bit_per_s: Ground-side predicted SEU rate (e.g.
            ``PhysicsLiteEnvironment.mission_seu_rate_per_bit_per_s``).
        n_bits: Number of bits in the monitored state.
        exposure_s: Mission exposure window the observation covers.
        observed_events: Actual upset count logged on orbit.
        envelope_factor: Factor band for ``within_envelope`` (default 3×).
        alpha: Significance level for the ``consistent`` verdict.

    Returns:
        A :class:`ReconciliationResult`.

    Raises:
        ValueError: On negative rate / bits / exposure / counts, or
            ``envelope_factor < 1`` or ``alpha`` outside ``(0, 1)``.
    """
    if predicted_rate_per_bit_per_s < 0:
        raise ValueError("predicted_rate_per_bit_per_s must be non-negative")
    if n_bits <= 0:
        raise ValueError("n_bits must be positive")
    if exposure_s <= 0:
        raise ValueError("exposure_s must be positive")
    if observed_events < 0:
        raise ValueError("observed_events must be non-negative")
    if envelope_factor < 1.0:
        raise ValueError("envelope_factor must be ≥ 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    mu = predicted_rate_per_bit_per_s * n_bits * exposure_s
    k = observed_events

    if mu > 0.0:
        ratio = k / mu
        chi_square = (k - mu) ** 2 / mu
    else:
        ratio = math.inf if k > 0 else 1.0
        chi_square = math.inf if k > 0 else 0.0

    p_value = _two_sided_p_value(k, mu)
    within_envelope = (mu > 0.0) and (1.0 / envelope_factor <= ratio <= envelope_factor)
    consistent = p_value >= alpha

    return ReconciliationResult(
        expected_events=mu,
        observed_events=k,
        ratio=ratio,
        chi_square=chi_square,
        p_value=p_value,
        within_envelope=within_envelope,
        consistent=consistent,
        envelope_factor=envelope_factor,
        alpha=alpha,
    )


def reconcile_events(
    predicted_rate_per_bit_per_s: float,
    *,
    n_bits: int,
    exposure_s: float,
    observed: Iterable[FaultEvent],
    envelope_factor: float = 3.0,
    alpha: float = 0.05,
) -> ReconciliationResult:
    """Like :func:`reconcile_seu_rate`, counting a stream of logged events.

    Convenience wrapper for on-orbit telemetry already loaded as
    :class:`~spacellm.types.FaultEvent` objects (e.g. from
    ``RunDB.fault_events_for``).
    """
    return reconcile_seu_rate(
        predicted_rate_per_bit_per_s,
        n_bits=n_bits,
        exposure_s=exposure_s,
        observed_events=sum(1 for _ in observed),
        envelope_factor=envelope_factor,
        alpha=alpha,
    )


__all__ = ["ReconciliationResult", "reconcile_events", "reconcile_seu_rate"]
