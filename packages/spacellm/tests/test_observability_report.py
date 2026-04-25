"""Tests for ``spacellm.observability.report`` and the HTML hooks on ``ProfileReport`` / ``RunDB``."""

from __future__ import annotations

from pathlib import Path

from spacellm.observability import Run, RunDB, render_profile_html, render_run_html
from spacellm.profiling import sensitivity_profile, static_profile
from spacellm.profiling.report import LayerSensitivity, ProfileReport
from spacellm.types import FaultCause, FaultEvent, RunRecord


def _has_html_skeleton(html: str) -> bool:
    return all(token in html for token in ("<!DOCTYPE html>", "<html", "</html>"))


def test_render_profile_html_static_minimal(tiny_transformer):
    profile = static_profile(tiny_transformer)
    rendered = render_profile_html(profile)
    assert _has_html_skeleton(rendered)
    assert "Layers" in rendered
    # No dynamic metrics on a static profile
    assert "Mean Δ" not in rendered


def test_render_profile_html_dynamic_includes_perturbation(tiny_transformer, tiny_input_ids):
    def fwd(model):
        return model(tiny_input_ids)

    profile = sensitivity_profile(tiny_transformer, fwd, n_flips_per_layer=1, seed=0)
    rendered = profile.to_html(title="Phi-3 sensitivity")
    assert _has_html_skeleton(rendered)
    assert "Phi-3 sensitivity" in rendered
    assert "Mean Δ" in rendered
    assert "Max Δ" in rendered


def test_profile_report_save_html(tiny_transformer, tmp_path: Path):
    profile = static_profile(tiny_transformer)
    out = profile.save_html(tmp_path / "profile.html")
    assert out.exists()
    text = out.read_text()
    assert _has_html_skeleton(text)
    # Make sure layer names are escaped properly (no raw '<' from the model class)
    assert "<script" not in text


def test_render_profile_escapes_user_content(tiny_transformer):
    profile = static_profile(tiny_transformer)
    profile.layers.append(
        LayerSensitivity(
            layer_name="<script>alert('x')</script>",
            module_class="EvilModule",
            n_parameters=1,
            n_bits=32,
        ),
    )
    rendered = render_profile_html(profile)
    assert "<script>alert" not in rendered
    assert "&lt;script&gt;" in rendered


def test_render_run_html_with_events_and_metrics():
    record = RunRecord(name="abl-1", model_name="phi-3-mini")
    record.metrics["perplexity"] = 7.42
    record.metrics["accuracy"] = 0.84
    record.n_faults_injected = 5
    record.n_faults_recovered = 4
    events = [
        FaultEvent(
            t=float(i),
            tensor_id=f"layers.{i}.weight",
            bit_index=i,
            bit_position=i % 32,
            before=0,
            after=1 << (i % 32),
            cause=FaultCause.SEU_HEAVY_ION,
        )
        for i in range(3)
    ]
    rendered = render_run_html(record, events)
    assert _has_html_skeleton(rendered)
    assert "perplexity" in rendered
    assert "accuracy" in rendered
    assert "layers.1.weight" in rendered
    assert "seu_heavy_ion" in rendered


def test_render_run_html_truncates_event_table():
    record = RunRecord(name="big")
    events = [
        FaultEvent(
            t=float(i),
            tensor_id="w",
            bit_index=i,
            bit_position=i % 32,
            before=0,
            after=1,
        )
        for i in range(700)
    ]
    rendered = render_run_html(record, events, event_cap=100)
    assert "Showing first 100" in rendered
    assert "of 700 events" in rendered


def test_runs_db_render_run_html_round_trip(tmp_path: Path):
    db_path = tmp_path / "runs.db"
    rid: str
    with RunDB(db_path) as db, Run(name="ctx", db=db) as run:
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
        rid = run.run_id

    with RunDB(db_path) as db:
        rendered = db.render_run_html(rid)
    assert rendered is not None
    assert _has_html_skeleton(rendered)
    assert "loss" in rendered
    assert "ctx" in rendered


def test_runs_db_render_run_html_returns_none_for_unknown_id():
    with RunDB(":memory:") as db:
        assert db.render_run_html("not-a-real-id") is None


def test_render_profile_html_handles_root_layer():
    """Profile that includes the root module name='' should still escape sensibly."""
    profile = ProfileReport(
        layers=[
            LayerSensitivity(
                layer_name="",
                module_class="Root",
                n_parameters=0,
                n_bits=0,
            ),
        ],
    )
    rendered = render_profile_html(profile)
    assert _has_html_skeleton(rendered)
    assert "&lt;root&gt;" in rendered
