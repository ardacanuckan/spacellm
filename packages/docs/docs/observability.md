# Observability

v0.1 ships SQLite-backed persistence for `RunRecord` and `FaultEvent`. The Prometheus exporters, OpenTelemetry instrumentation, and Phoenix integration land in v0.5.

## RunDB

```python
from spacellm.observability import RunDB
from spacellm.types import RunRecord

with RunDB("runs.db") as db:
    record = RunRecord(name="ablation", model_name="phi-3-mini", model_sha="abc123")
    db.upsert_run(record)

    # Bulk-insert fault events.
    db.insert_fault_events(record.run_id, events)

    # Read back.
    rows = db.list_runs(limit=20)
    fetched = db.get_run(record.run_id)
    fault_events = db.fault_events_for(record.run_id)
```

* **`":memory:"`** is supported for tests and ephemeral runs.
* **Foreign keys** are enforced, deleting a run cascades to its fault events.
* **JSON columns** for `environment_spec`, `protection_spec`, `metrics`, `artifact_uris`. Round-trip via `json.loads` / `json.dumps`.

## Run context manager

The recommended idiom for instrumented runs:

```python
from spacellm.observability import Run, RunDB

with RunDB("runs.db") as db, Run(name="ablation-1", db=db) as run:
    run.log_metric("perplexity", 7.42)
    for event in events:
        run.log_fault(event)
    run.log_recovery(5)
```

The `RunRecord` is upserted on entry (so a crash leaves a partial record) and again on exit (with `ended_at` populated). Fault events flush to the DB on exit.

## CLI

```bash
spacellm runs                   # list 20 most recent
spacellm runs --db custom.db -n 100
spacellm show abc12345          # 8-char prefix is enough
```

Output is rendered with Rich, terminal-friendly tables.

## Forward path to v0.5+

The OSS observability stack v0.5 ships with:

* **Prometheus** exporter under `spacellm.observability.metrics`, bit-flip rate, perplexity over time, accuracy degradation curve, protection overhead, false-positive / false-negative detection rates.
* **OpenTelemetry** instrumentation following the GenAI semantic conventions (ratified by 2026).
* **Phoenix** by Arize for trace inspection, best OSS LLM observability UI today.
* **MLflow** for run-level tracking; Parquet artifact upload to local FS or S3.
* **Grafana dashboards** shipped as JSON in `observability/dashboards/`.
