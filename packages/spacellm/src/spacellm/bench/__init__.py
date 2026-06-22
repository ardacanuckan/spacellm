"""Benchmarking primitives, the foundation for SpaceBench v0 (v0.6).

v0.3 ships:

- :func:`spacellm.bench.silent_error_rate`, the unique radhard KPI from
  the bench module.
- :func:`spacellm.bench.bench_protection`, wraps a model + environment
  + forward function in the canonical "inject, observe, measure" loop
  and returns a :class:`BenchResult`.

The full SpaceBench leaderboard infrastructure (HF Spaces submission,
dose profiles, multi-task harness) lands in v0.6.
"""

from __future__ import annotations

from spacellm.bench.harness import BenchResult, bench_protection
from spacellm.bench.metrics import silent_error_rate
from spacellm.bench.quantization import (
    NumberFormatSensitivity,
    compare_number_formats,
    format_sensitivity,
)

__all__ = [
    "BenchResult",
    "NumberFormatSensitivity",
    "bench_protection",
    "compare_number_formats",
    "format_sensitivity",
    "silent_error_rate",
]
