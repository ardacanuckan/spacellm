"""Tests for ``spacellm.cli``."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from spacellm import __version__
from spacellm.cli import app
from spacellm.observability import Run, RunDB

runner = CliRunner()


def test_version_command_prints_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_runs_command_warns_when_db_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no-such.db"
    result = runner.invoke(app, ["runs", "--db", str(missing)])
    assert result.exit_code == 0
    assert "No run database" in result.stdout


def test_runs_command_lists_recent(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    with RunDB(db_path) as db, Run(name="alpha", db=db):
        pass
    result = runner.invoke(app, ["runs", "--db", str(db_path)])
    assert result.exit_code == 0
    assert "alpha" in result.stdout


def test_show_command_resolves_prefix(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    with RunDB(db_path) as db, Run(name="beta", db=db) as run:
        rid = run.run_id
    result = runner.invoke(app, ["show", rid[:8], "--db", str(db_path)])
    assert result.exit_code == 0
    assert "beta" in result.stdout


def test_show_command_errors_on_unknown_run(tmp_path: Path) -> None:
    db_path = tmp_path / "runs.db"
    with RunDB(db_path):
        pass
    result = runner.invoke(app, ["show", "ffffffff", "--db", str(db_path)])
    assert result.exit_code == 1
