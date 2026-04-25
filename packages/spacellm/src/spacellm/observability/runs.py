"""SQLite-backed persistence for ``RunRecord`` and ``FaultEvent``."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from spacellm.types import FaultCause, FaultEvent, RunRecord

_RUN_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    name                TEXT,
    model_name          TEXT,
    model_sha           TEXT,
    environment_spec    TEXT,
    protection_spec     TEXT,
    started_at          TEXT,
    ended_at            TEXT,
    metrics             TEXT,
    artifact_uris       TEXT,
    n_faults_injected   INTEGER DEFAULT 0,
    n_faults_recovered  INTEGER DEFAULT 0,
    notes               TEXT
)
"""

_FAULT_TABLE_DDL = """
CREATE TABLE IF NOT EXISTS fault_events (
    run_id        TEXT NOT NULL,
    t             REAL NOT NULL,
    tensor_id     TEXT NOT NULL,
    bit_index     INTEGER NOT NULL,
    bit_position  INTEGER NOT NULL,
    before        INTEGER NOT NULL,
    after         INTEGER NOT NULL,
    cause         TEXT NOT NULL,
    FOREIGN KEY(run_id) REFERENCES runs(run_id) ON DELETE CASCADE
)
"""

_FAULT_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS idx_fault_events_run_id ON fault_events(run_id)
"""


def _to_iso(ts: datetime | None) -> str | None:
    return ts.isoformat() if ts is not None else None


def _from_iso(text: str | None) -> datetime | None:
    return datetime.fromisoformat(text) if text else None


class RunDB:
    """A single-file SQLite store for runs and their fault events.

    Designed for local development and CI. Production deployments will
    swap in PostgreSQL via the same API in v0.5.

    Args:
        path: SQLite file path. ``":memory:"`` is supported for tests.
    """

    def __init__(self, path: str | Path = "runs.db") -> None:
        self.path = path if path == ":memory:" else Path(path)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute(_RUN_TABLE_DDL)
        self._conn.execute(_FAULT_TABLE_DDL)
        self._conn.execute(_FAULT_INDEX_DDL)
        self._conn.commit()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        self._conn.close()

    def upsert_run(self, run: RunRecord) -> None:
        """Insert or update one run, keyed by ``run_id``."""
        self._conn.execute(
            """
            INSERT INTO runs (
                run_id, name, model_name, model_sha,
                environment_spec, protection_spec,
                started_at, ended_at, metrics, artifact_uris,
                n_faults_injected, n_faults_recovered, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                name=excluded.name,
                model_name=excluded.model_name,
                model_sha=excluded.model_sha,
                environment_spec=excluded.environment_spec,
                protection_spec=excluded.protection_spec,
                started_at=excluded.started_at,
                ended_at=excluded.ended_at,
                metrics=excluded.metrics,
                artifact_uris=excluded.artifact_uris,
                n_faults_injected=excluded.n_faults_injected,
                n_faults_recovered=excluded.n_faults_recovered,
                notes=excluded.notes
            """,
            (
                run.run_id,
                run.name,
                run.model_name,
                run.model_sha,
                json.dumps(run.environment_spec),
                json.dumps(run.protection_spec),
                _to_iso(run.started_at),
                _to_iso(run.ended_at),
                json.dumps(run.metrics),
                json.dumps(run.artifact_uris),
                run.n_faults_injected,
                run.n_faults_recovered,
                run.notes,
            ),
        )
        self._conn.commit()

    def insert_fault_events(self, run_id: str, events: Iterable[FaultEvent]) -> int:
        """Bulk insert fault events for ``run_id``. Returns inserted count."""
        rows = [
            (
                run_id,
                e.t,
                e.tensor_id,
                e.bit_index,
                e.bit_position,
                e.before,
                e.after,
                e.cause.value,
            )
            for e in events
        ]
        if not rows:
            return 0
        self._conn.executemany(
            "INSERT INTO fault_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        return len(rows)

    def get_run(self, run_id: str) -> RunRecord | None:
        """Fetch one run by id, or ``None`` if absent."""
        cur = self._conn.execute(
            "SELECT name, model_name, model_sha, environment_spec, protection_spec, "
            "started_at, ended_at, metrics, artifact_uris, "
            "n_faults_injected, n_faults_recovered, notes "
            "FROM runs WHERE run_id = ?",
            (run_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        (
            name,
            model_name,
            model_sha,
            environment_spec,
            protection_spec,
            started_at,
            ended_at,
            metrics,
            artifact_uris,
            n_faults_injected,
            n_faults_recovered,
            notes,
        ) = row
        return RunRecord(
            run_id=run_id,
            name=name,
            model_name=model_name,
            model_sha=model_sha,
            environment_spec=json.loads(environment_spec or "{}"),
            protection_spec=json.loads(protection_spec or "{}"),
            started_at=_from_iso(started_at) or datetime.now(UTC),
            ended_at=_from_iso(ended_at),
            metrics=json.loads(metrics or "{}"),
            artifact_uris=json.loads(artifact_uris or "[]"),
            n_faults_injected=n_faults_injected or 0,
            n_faults_recovered=n_faults_recovered or 0,
            notes=notes or "",
        )

    def list_runs(self, *, limit: int = 100) -> list[RunRecord]:
        """Most-recent-first list of runs."""
        cur = self._conn.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        run_ids = [r[0] for r in cur.fetchall()]
        return [r for r in (self.get_run(rid) for rid in run_ids) if r is not None]

    def render_run_html(self, run_id: str, *, event_cap: int = 500) -> str | None:
        """Render the named run as a self-contained HTML document.

        Returns ``None`` if the run does not exist; otherwise the full HTML
        document including the run's metrics and (up to ``event_cap``)
        fault events.
        """
        from spacellm.observability.report import render_run_html  # noqa: PLC0415

        record = self.get_run(run_id)
        if record is None:
            return None
        events = self.fault_events_for(run_id)
        return render_run_html(record, events, event_cap=event_cap)

    def fault_events_for(self, run_id: str) -> list[FaultEvent]:
        """All fault events recorded for one run, in insertion order."""
        cur = self._conn.execute(
            "SELECT t, tensor_id, bit_index, bit_position, before, after, cause "
            "FROM fault_events WHERE run_id = ? ORDER BY rowid",
            (run_id,),
        )
        return [
            FaultEvent(
                t=row[0],
                tensor_id=row[1],
                bit_index=row[2],
                bit_position=row[3],
                before=row[4],
                after=row[5],
                cause=FaultCause(row[6]),
            )
            for row in cur.fetchall()
        ]


class Run:
    """Context-managed lifecycle for one run.

    Lazily flushes fault events to the DB on exit; metrics and counters
    are persisted on entry and on exit so a crashed run still leaves a
    record visible to ``RunDB.list_runs()``.

    Example
    -------

    .. code-block:: python

        from spacellm.observability import Run, RunDB

        db = RunDB("runs.db")
        with Run(name="ablation", db=db) as run:
            run.log_metric("perplexity", 7.42)
            run.log_fault(event)
    """

    def __init__(
        self,
        name: str = "",
        *,
        db: RunDB | None = None,
        record: RunRecord | None = None,
    ) -> None:
        self.record = record or RunRecord(name=name)
        self.db = db or RunDB()
        self._fault_buffer: list[FaultEvent] = []

    @property
    def run_id(self) -> str:
        return self.record.run_id

    def log_metric(self, key: str, value: float) -> None:
        self.record.metrics[key] = float(value)

    def log_fault(self, event: FaultEvent) -> None:
        self._fault_buffer.append(event)
        self.record.n_faults_injected += 1

    def log_recovery(self, n: int = 1) -> None:
        self.record.n_faults_recovered += int(n)

    def __enter__(self) -> Self:
        self.db.upsert_run(self.record)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.record.ended_at = datetime.now(UTC)
        if self._fault_buffer:
            self.db.insert_fault_events(self.record.run_id, self._fault_buffer)
            self._fault_buffer.clear()
        self.db.upsert_run(self.record)


__all__ = ["Run", "RunDB"]
