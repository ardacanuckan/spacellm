"""Tests for ``spacellm.types``."""

from __future__ import annotations

import dataclasses
from datetime import datetime

import pytest

from spacellm.types import (
    DeviceModel,
    FaultCause,
    FaultEvent,
    RadiationStep,
    RunRecord,
)


def test_fault_event_is_frozen() -> None:
    e = FaultEvent(t=0.0, tensor_id="x", bit_index=1, bit_position=0, before=0, after=1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.t = 1.0  # type: ignore[misc]


def test_fault_cause_round_trips_as_string() -> None:
    # StrEnum: str(member) is the value, which is what JSON/SQLite consumers want.
    assert FaultCause.SEU_PROTON.value == "seu_proton"
    assert FaultCause("seu_heavy_ion") is FaultCause.SEU_HEAVY_ION
    assert str(FaultCause.SYNTHETIC) == "synthetic"
    assert FaultCause.SYNTHETIC.name == "SYNTHETIC"


def test_fault_event_default_cause_is_synthetic() -> None:
    e = FaultEvent(t=0.0, tensor_id="w", bit_index=0, bit_position=0, before=0, after=1)
    assert e.cause is FaultCause.SYNTHETIC


def test_radiation_step_has_safe_defaults() -> None:
    step = RadiationStep(
        t=0.0,
        proton_flux=0.0,
        heavy_ion_flux=0.0,
        seu_rate_per_bit_per_s=0.0,
        tid_rate_gy_per_s=0.0,
    )
    assert step.inside_saa is False
    assert step.source == "synthetic"


def test_run_record_generates_unique_ids() -> None:
    a = RunRecord()
    b = RunRecord()
    assert a.run_id != b.run_id
    assert len(a.run_id) == 32


def test_run_record_started_at_is_utc() -> None:
    r = RunRecord()
    assert isinstance(r.started_at, datetime)
    assert r.started_at.tzinfo is not None


def test_device_model_minimal_fields() -> None:
    d = DeviceModel(
        name="test",
        process_node_nm=7.0,
        sensitive_volume_um3=1.0,
        saturation_cross_section_cm2=1e-7,
        onset_let_mev_cm2_per_mg=1.0,
    )
    assert d.shielding_g_cm2 == 0.0
    assert d.width == 30.0
    assert d.shape == 1.5
