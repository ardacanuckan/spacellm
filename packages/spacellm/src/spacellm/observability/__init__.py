"""Run tracking and telemetry persistence.

v0.1 ships a single SQLite-backed :class:`RunDB` plus a :class:`Run`
context manager. The Prometheus exporters, OpenTelemetry instrumentation,
and Phoenix integration land in v0.5, see ``CHANGELOG.md``.
"""

from __future__ import annotations

from spacellm.observability.report import render_profile_html, render_run_html
from spacellm.observability.runs import Run, RunDB

__all__ = ["Run", "RunDB", "render_profile_html", "render_run_html"]
