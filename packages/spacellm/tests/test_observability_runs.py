"""Tests for ``spacellm.observability.runs``."""

from __future__ import annotations

from pathlib import Path

from spacellm.observability import Run, RunDB
from spacellm.types import FaultCause, FaultEvent, RunRecord


def test_run_db_round_trip_in_memory() -> None:
    with RunDB(":memory:") as db:
        record = RunRecord(name="abl", model_name="tiny", model_sha="abc")
        db.upsert_run(record)
        fetched = db.get_run(record.run_id)
        assert fetched is not None
        assert fetched.name == "abl"
        assert fetched.model_sha == "abc"


def test_run_db_persists_to_file(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    rid: str
    with RunDB(db_path) as db:
        record = RunRecord(name="durable")
        db.upsert_run(record)
        rid = record.run_id

    with RunDB(db_path) as reopened:
        fetched = reopened.get_run(rid)
    assert fetched is not None
    assert fetched.name == "durable"


def test_run_db_list_recent_first() -> None:
    with RunDB(":memory:") as db:
        a = RunRecord(name="a")
        b = RunRecord(name="b")
        db.upsert_run(a)
        db.upsert_run(b)
        rows = db.list_runs()
    names = {r.name for r in rows}
    assert {"a", "b"} <= names


def test_run_db_inserts_fault_events() -> None:
    with RunDB(":memory:") as db:
        record = RunRecord(name="fault-test")
        db.upsert_run(record)
        events = [
            FaultEvent(
                t=float(i),
                tensor_id=f"weights.{i}",
                bit_index=i,
                bit_position=i % 32,
                before=0,
                after=1 << (i % 32),
                cause=FaultCause.SEU_PROTON,
            )
            for i in range(5)
        ]
        n_inserted = db.insert_fault_events(record.run_id, events)
        assert n_inserted == 5
        fetched = db.fault_events_for(record.run_id)
        assert len(fetched) == 5
        assert fetched[2].tensor_id == "weights.2"
        assert fetched[2].cause is FaultCause.SEU_PROTON


def test_run_context_manager_persists_on_exit(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    with RunDB(db_path) as db:
        with Run(name="ctx", db=db) as run:
            run.log_metric("loss", 1.5)
            run.log_fault(
                FaultEvent(
                    t=0.1,
                    tensor_id="w",
                    bit_index=0,
                    bit_position=0,
                    before=0,
                    after=1,
                ),
            )
            run.log_recovery(1)
            rid = run.run_id
        fetched = db.get_run(rid)
    assert fetched is not None
    assert fetched.metrics["loss"] == 1.5
    assert fetched.n_faults_injected == 1
    assert fetched.n_faults_recovered == 1
    assert fetched.ended_at is not None


def test_run_db_get_returns_none_on_unknown_id() -> None:
    with RunDB(":memory:") as db:
        assert db.get_run("not-a-real-id") is None


def test_insert_fault_events_zero_returns_zero() -> None:
    with RunDB(":memory:") as db:
        record = RunRecord(name="empty-faults")
        db.upsert_run(record)
        assert db.insert_fault_events(record.run_id, []) == 0
