"""Benchmark harness, the canonical inject/observe/measure loop."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from spacellm.bench.metrics import silent_error_rate
from spacellm.environments.base import Environment


@dataclass(slots=True)
class BenchResult:
    """Outcome of a :func:`bench_protection` run.

    Attributes:
        n_steps: Number of fault-injection steps that ran.
        n_faults_injected: Total bit-flips applied across the run.
        silent_error_rate_max: Maximum silent-error rate observed across
            all steps.
        silent_error_rate_final: Silent-error rate measured on the final
            forward.
        final_max_abs_diff: Largest absolute difference between the
            final output and the reference. ``inf`` if the final output
            contains any NaNs.
        walltime_s: Wall-clock seconds the harness spent.
        nan_at_end: ``True`` if the final output contains any NaNs.
    """

    n_steps: int
    n_faults_injected: int
    silent_error_rate_max: float
    silent_error_rate_final: float
    final_max_abs_diff: float
    walltime_s: float
    nan_at_end: bool


def bench_protection(
    model: nn.Module,
    forward_fn: Callable[[nn.Module], torch.Tensor],
    reference: torch.Tensor,
    environment: Environment,
    *,
    n_steps: int = 10,
    dt_s: float = 1.0,
    silent_error_threshold: float = 1e-3,
) -> BenchResult:
    """Run a fault-injection benchmark.

    Loop:

    .. code-block:: text

        for step in range(n_steps):
            events = environment.sample_faults(model.parameters, dt_s)
            environment.step(dt_s)
            out = forward_fn(model)
            ser = silent_error_rate(out, reference, threshold)

    The environment is responsible for actually flipping bits in the
    parameter tensors, :class:`StatisticalEnvironment` and
    :class:`PhysicsLiteEnvironment` both do this in
    :meth:`sample_faults`.

    Args:
        model: The model under test (typically pre-hardened by
            :func:`spacellm.harden`).
        forward_fn: Callable ``model -> output_tensor`` for a fixed
            input. The user fixes the input.
        reference: Trusted output; typically ``forward_fn(clean_model)``.
        environment: Fault source. The same instance is stepped each
            iteration.
        n_steps: Number of inject/observe/measure cycles.
        dt_s: Window length per step, in seconds (mission time).
        silent_error_threshold: Element-wise absolute deviation that
            counts as a silent error.

    Returns:
        A :class:`BenchResult` with aggregated metrics.

    Raises:
        ValueError: If ``n_steps`` < 1 or ``dt_s`` ≤ 0.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be ≥ 1, got {n_steps}")
    if dt_s <= 0:
        raise ValueError(f"dt_s must be positive, got {dt_s}")

    start = time.perf_counter()
    n_faults = 0
    max_ser = 0.0

    with torch.no_grad():
        for _ in range(n_steps):
            tensors = {
                name: param.data for name, param in model.named_parameters() if param.numel() > 0
            }
            events = environment.sample_faults(tensors, dt_s)
            n_faults += len(events)
            environment.step(dt_s)
            out = forward_fn(model)
            ser = silent_error_rate(out, reference, threshold=silent_error_threshold)
            max_ser = max(max_ser, ser)

        final = forward_fn(model)

    nan_at_end = bool(torch.isnan(final).any().item())
    final_diff = float("inf") if nan_at_end else float((final - reference).abs().max().item())

    return BenchResult(
        n_steps=n_steps,
        n_faults_injected=n_faults,
        silent_error_rate_max=max_ser,
        silent_error_rate_final=silent_error_rate(
            final,
            reference,
            threshold=silent_error_threshold,
        ),
        final_max_abs_diff=final_diff,
        walltime_s=time.perf_counter() - start,
        nan_at_end=nan_at_end,
    )


__all__ = ["BenchResult", "bench_protection"]
