"""Tests for ``spacellm.validation.reconcile`` (post-flight reconciliation)."""

from __future__ import annotations

import math

import pytest

from spacellm.types import FaultCause, FaultEvent
from spacellm.validation import reconcile_events, reconcile_seu_rate


def test_consistent_when_observed_matches_prediction() -> None:
    # μ = 1e-9 * 1e9 * 100 = 100 expected; observe exactly 100.
    result = reconcile_seu_rate(
        1e-9,
        n_bits=1_000_000_000,
        exposure_s=100.0,
        observed_events=100,
    )
    assert result.expected_events == pytest.approx(100.0)
    assert result.consistent is True
    assert result.within_envelope is True
    assert result.p_value > 0.05


def test_divergent_when_observed_far_off() -> None:
    # Expect ~100, observe 10× too many.
    result = reconcile_seu_rate(
        1e-9,
        n_bits=1_000_000_000,
        exposure_s=100.0,
        observed_events=1000,
    )
    assert result.consistent is False
    assert result.within_envelope is False
    assert result.p_value < 0.05
    assert result.ratio == pytest.approx(10.0, rel=1e-6)


def test_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n_bits"):
        reconcile_seu_rate(1e-9, n_bits=0, exposure_s=1.0, observed_events=0)
    with pytest.raises(ValueError, match="exposure_s"):
        reconcile_seu_rate(1e-9, n_bits=1, exposure_s=0.0, observed_events=0)
    with pytest.raises(ValueError, match="alpha"):
        reconcile_seu_rate(1e-9, n_bits=1, exposure_s=1.0, observed_events=0, alpha=1.0)
    with pytest.raises(ValueError, match="envelope_factor"):
        reconcile_seu_rate(1e-9, n_bits=1, exposure_s=1.0, observed_events=0, envelope_factor=0.5)


def test_zero_prediction_zero_observed_is_consistent() -> None:
    result = reconcile_seu_rate(0.0, n_bits=1000, exposure_s=10.0, observed_events=0)
    assert result.expected_events == 0.0
    assert result.consistent is True
    assert result.chi_square == 0.0


def test_zero_prediction_with_observed_diverges() -> None:
    result = reconcile_seu_rate(0.0, n_bits=1000, exposure_s=10.0, observed_events=5)
    assert math.isinf(result.ratio)
    assert math.isinf(result.chi_square)
    assert result.consistent is False


def test_large_mean_uses_normal_approximation() -> None:
    # μ huge → exact summation path is skipped; should still produce a
    # finite, sensible p-value and a consistent verdict for an on-mean obs.
    result = reconcile_seu_rate(
        1e-3,
        n_bits=10_000_000,
        exposure_s=1000.0,
        observed_events=10_000_000,
    )
    assert 0.0 <= result.p_value <= 1.0
    assert result.consistent is True


def test_reconcile_events_counts_stream() -> None:
    events = [
        FaultEvent(
            t=float(i),
            tensor_id="w",
            bit_index=i,
            bit_position=0,
            before=0,
            after=1,
            cause=FaultCause.SEU_HEAVY_ION,
        )
        for i in range(100)
    ]
    result = reconcile_events(
        1e-9,
        n_bits=1_000_000_000,
        exposure_s=100.0,
        observed=events,
    )
    assert result.observed_events == 100
    assert result.consistent is True


def test_summary_is_readable() -> None:
    result = reconcile_seu_rate(1e-9, n_bits=1_000_000_000, exposure_s=100.0, observed_events=100)
    text = result.summary()
    assert "reconciliation" in text
    assert "p-value" in text
