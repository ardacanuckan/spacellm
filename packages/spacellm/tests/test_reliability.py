"""Tests for ``spacellm.reliability`` (defense-layer composition)."""

from __future__ import annotations

import math

import pytest

from spacellm.reliability import ReliabilityStack, compose_residual_rate


def _stack(**overrides: object) -> ReliabilityStack:
    base: dict[str, object] = {
        "raw_seu_rate_per_bit_per_s": 1e-11,
        "n_bits": 16_000_000_000,
        "ecc": "secded",
        "word_data_bits": 64,
        "scrub_period_s": 1.0,
        "abft_coverage": 0.99,
        "mcu_double_fraction": 0.0,
    }
    base.update(overrides)
    return ReliabilityStack(**base)  # type: ignore[arg-type]


def test_validates_inputs() -> None:
    with pytest.raises(ValueError, match="n_bits"):
        _stack(n_bits=0)
    with pytest.raises(ValueError, match="ecc"):
        _stack(ecc="hamming")
    with pytest.raises(ValueError, match="abft_coverage"):
        _stack(abft_coverage=1.5)
    with pytest.raises(ValueError, match="scrub_period_s"):
        _stack(scrub_period_s=0.0)


def test_no_protection_residual_equals_raw() -> None:
    report = compose_residual_rate(_stack(ecc="none", abft_coverage=0.0))
    assert report.ecc_escape_prob == 1.0
    assert report.abft_escape_prob == 1.0
    assert report.residual_silent_rate_per_s == pytest.approx(report.raw_rate_per_s)
    assert report.improvement_factor == pytest.approx(1.0)


def test_each_layer_reduces_residual() -> None:
    raw = compose_residual_rate(_stack(ecc="none", abft_coverage=0.0))
    ecc_only = compose_residual_rate(_stack(ecc="secded", abft_coverage=0.0))
    both = compose_residual_rate(_stack(ecc="secded", abft_coverage=0.99))
    assert ecc_only.residual_silent_rate_per_s < raw.residual_silent_rate_per_s
    assert both.residual_silent_rate_per_s < ecc_only.residual_silent_rate_per_s
    assert both.improvement_factor > ecc_only.improvement_factor > 1.0


def test_shorter_scrub_period_helps() -> None:
    slow = compose_residual_rate(_stack(scrub_period_s=10.0, abft_coverage=0.0))
    fast = compose_residual_rate(_stack(scrub_period_s=0.1, abft_coverage=0.0))
    assert fast.residual_silent_rate_per_s < slow.residual_silent_rate_per_s


def test_mcu_double_fraction_floors_ecc_escape() -> None:
    # An immediate multi-cell double defeats SECDED regardless of scrubbing.
    report = compose_residual_rate(
        _stack(mcu_double_fraction=0.3, scrub_period_s=1e-6, abft_coverage=0.0)
    )
    assert report.ecc_escape_prob >= 0.3


def test_perfect_abft_zeroes_residual() -> None:
    report = compose_residual_rate(_stack(abft_coverage=1.0))
    assert report.residual_silent_rate_per_s == 0.0
    assert math.isinf(report.improvement_factor)
    assert math.isinf(report.mean_time_between_silent_s)


def test_summary_is_readable() -> None:
    text = compose_residual_rate(_stack()).summary()
    assert "residual silent-error budget" in text
    assert "improvement" in text
