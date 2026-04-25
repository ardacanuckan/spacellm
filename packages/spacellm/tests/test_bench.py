"""Tests for ``spacellm.bench``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

import spacellm as sl
from spacellm.bench import BenchResult, bench_protection, silent_error_rate
from spacellm.environments import StatisticalEnvironment


def test_silent_error_rate_zero_for_identical() -> None:
    a = torch.randn(8, 8)
    assert silent_error_rate(a, a) == 0.0


def test_silent_error_rate_one_for_huge_deviation() -> None:
    a = torch.zeros(4, 4)
    b = torch.ones(4, 4) * 1e6
    assert silent_error_rate(a, b, threshold=1e-3) == pytest.approx(1.0)


def test_silent_error_rate_partial() -> None:
    a = torch.zeros(10)
    b = torch.zeros(10)
    b[3:7] = 1.0  # 4 of 10 elements differ
    assert silent_error_rate(a, b, threshold=0.5) == pytest.approx(0.4)


def test_silent_error_rate_nan_counts_as_error() -> None:
    a = torch.zeros(4)
    b = torch.tensor([0.0, float("nan"), 0.0, 0.0])
    assert silent_error_rate(a, b) == pytest.approx(0.25)


def test_silent_error_rate_shape_mismatch_rejected() -> None:
    a = torch.zeros(4)
    b = torch.zeros(5)
    with pytest.raises(ValueError, match="match in shape"):
        silent_error_rate(a, b)


def test_silent_error_rate_empty_rejected() -> None:
    a = torch.empty(0)
    b = torch.empty(0)
    with pytest.raises(ValueError, match="empty"):
        silent_error_rate(a, b)


def test_bench_protection_clean_environment(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    """A zero-rate environment must produce zero faults and zero silent errors."""
    reference = tiny_transformer(tiny_input_ids).clone()

    def forward(model: nn.Module) -> torch.Tensor:
        return model(tiny_input_ids)

    env = StatisticalEnvironment(rate_per_bit_per_s=0.0, seed=0)
    result = bench_protection(
        tiny_transformer,
        forward,
        reference,
        env,
        n_steps=3,
        dt_s=1.0,
    )
    assert isinstance(result, BenchResult)
    assert result.n_steps == 3
    assert result.n_faults_injected == 0
    assert result.silent_error_rate_max == 0.0
    assert result.silent_error_rate_final == 0.0
    assert result.final_max_abs_diff < 1e-5  # floating-point noise OK
    assert not result.nan_at_end


def test_bench_protection_high_rate_breaks_unprotected_model(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    """At a punishing rate, an unprotected model should diverge from baseline."""
    reference = tiny_transformer(tiny_input_ids).clone()

    def forward(model: nn.Module) -> torch.Tensor:
        return model(tiny_input_ids)

    env = StatisticalEnvironment(rate_per_bit_per_s=1e-5, seed=0)
    result = bench_protection(
        tiny_transformer,
        forward,
        reference,
        env,
        n_steps=3,
        dt_s=1.0,
    )
    assert result.n_faults_injected > 0
    assert result.silent_error_rate_max > 0


def test_bench_protection_rejects_invalid_args(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    reference = tiny_transformer(tiny_input_ids).clone()
    env = StatisticalEnvironment(rate_per_bit_per_s=0.0, seed=0)

    def forward(model: nn.Module) -> torch.Tensor:
        return model(tiny_input_ids)

    with pytest.raises(ValueError, match="n_steps"):
        bench_protection(
            tiny_transformer,
            forward,
            reference,
            env,
            n_steps=0,
            dt_s=1.0,
        )
    with pytest.raises(ValueError, match="dt_s"):
        bench_protection(
            tiny_transformer,
            forward,
            reference,
            env,
            n_steps=1,
            dt_s=0.0,
        )


def test_bench_protection_walltime_recorded(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    reference = tiny_transformer(tiny_input_ids).clone()

    def forward(model: nn.Module) -> torch.Tensor:
        return model(tiny_input_ids)

    env = StatisticalEnvironment(rate_per_bit_per_s=0.0, seed=0)
    result = bench_protection(
        tiny_transformer,
        forward,
        reference,
        env,
        n_steps=2,
        dt_s=1.0,
    )
    assert result.walltime_s > 0


def test_bench_module_reachable_from_top_level() -> None:
    assert sl.bench.silent_error_rate is silent_error_rate
    assert sl.bench.bench_protection is bench_protection
