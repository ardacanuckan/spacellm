"""Tests for ``spacellm.runtime.harden``."""

from __future__ import annotations

import torch
from torch import nn

import spacellm as sl
from spacellm.nn import TMRLinear


def test_harden_returns_handle_with_strategies(tiny_transformer: nn.Module) -> None:
    strategies = [sl.protection.SelectiveTMR(top_k_percent=20.0)]
    hardened = sl.harden(tiny_transformer, strategies=strategies)
    assert isinstance(hardened, sl.HardenedModel)
    assert len(hardened.strategies) == 1
    assert hardened.environment is None


def test_harden_calls_apply_on_each_strategy(tiny_transformer: nn.Module) -> None:
    sl.harden(
        tiny_transformer,
        strategies=[sl.protection.SelectiveTMR(top_k_percent=100.0)],
    )
    n_tmr = sum(1 for m in tiny_transformer.modules() if isinstance(m, TMRLinear))
    assert n_tmr > 0


def test_harden_call_delegates_to_model(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    hardened = sl.harden(
        tiny_transformer,
        strategies=[sl.protection.SelectiveTMR(top_k_percent=20.0)],
    )
    out_via_handle = hardened(tiny_input_ids)
    out_via_model = hardened.model(tiny_input_ids)
    assert torch.equal(out_via_handle, out_via_model)


def test_harden_collect_reports(tiny_transformer: nn.Module) -> None:
    hardened = sl.harden(
        tiny_transformer,
        strategies=[
            sl.protection.SelectiveTMR(top_k_percent=20.0),
            sl.protection.AttentionChecksum(),
        ],
    )
    reports = hardened.collect_reports()
    assert len(reports) == 2
    assert {r.strategy_name for r in reports} == {"selective_tmr", "attention_checksum"}


def test_harden_with_environment_attaches_it(tiny_transformer: nn.Module) -> None:
    env = sl.environments.StatisticalEnvironment(rate_per_bit_per_s=0.0, seed=0)
    hardened = sl.harden(
        tiny_transformer,
        strategies=[sl.protection.SelectiveTMR(top_k_percent=20.0)],
        environment=env,
    )
    assert hardened.environment is env


def test_reset_telemetry_clears_counters(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    strategy = sl.protection.AttentionChecksum()
    hardened = sl.harden(tiny_transformer, strategies=[strategy])
    hardened(tiny_input_ids)
    assert strategy.collect_report().n_checks_performed > 0
    hardened.reset_telemetry()
    assert strategy.collect_report().n_checks_performed == 0
