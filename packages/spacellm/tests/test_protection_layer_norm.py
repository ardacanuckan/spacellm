"""Tests for ``spacellm.protection.layer_norm`` and ``spacellm.nn.layer_norm``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from spacellm._internal.bitops import flip_bit
from spacellm.nn import TMRLayerNorm
from spacellm.protection import LayerNormTMR


def test_tmr_layernorm_matches_original() -> None:
    torch.manual_seed(0)
    original = nn.LayerNorm(8)
    original.weight.data.uniform_(0.5, 1.5)
    original.bias.data.uniform_(-0.1, 0.1)
    wrapped = TMRLayerNorm(original)
    x = torch.randn(2, 4, 8)
    assert torch.allclose(original(x), wrapped(x))


def test_tmr_layernorm_masks_replica_corruption() -> None:
    torch.manual_seed(0)
    original = nn.LayerNorm(8)
    original.weight.data.uniform_(0.5, 1.5)
    wrapped = TMRLayerNorm(original)
    x = torch.randn(2, 4, 8)
    clean = wrapped(x).clone()
    flip_bit(wrapped.weight_a.data, flat_index=0, bit_position=30)  # exponent bit
    assert torch.allclose(wrapped(x), clean)


def test_tmr_layernorm_rejects_non_norm() -> None:
    with pytest.raises(TypeError):
        TMRLayerNorm(nn.Linear(4, 4))


def test_tmr_layernorm_rejects_non_affine() -> None:
    ln = nn.LayerNorm(4, elementwise_affine=False)
    with pytest.raises(ValueError, match="elementwise_affine"):
        TMRLayerNorm(ln)


def test_tmr_rmsnorm_runs() -> None:
    if not hasattr(nn, "RMSNorm"):
        pytest.skip("PyTorch < 2.4 — no nn.RMSNorm")
    rms = nn.RMSNorm(8)
    rms.weight.data.uniform_(0.5, 1.5)
    wrapped = TMRLayerNorm(rms)
    x = torch.randn(2, 4, 8)
    assert torch.allclose(rms(x), wrapped(x))


def test_layer_norm_tmr_strategy_wraps_norms_in_transformer(
    tiny_transformer: nn.Module,
) -> None:
    n_norms_before = sum(1 for m in tiny_transformer.modules() if isinstance(m, nn.LayerNorm))
    assert n_norms_before > 0  # encoder layers each have 2 LayerNorms
    strategy = LayerNormTMR()
    strategy.apply(tiny_transformer)
    n_tmr = sum(1 for m in tiny_transformer.modules() if isinstance(m, TMRLayerNorm))
    assert n_tmr == n_norms_before
    assert len(strategy.wrapped_paths) == n_tmr


def test_layer_norm_tmr_does_not_break_forward(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    clean_out = tiny_transformer(tiny_input_ids).clone()
    LayerNormTMR().apply(tiny_transformer)
    hardened_out = tiny_transformer(tiny_input_ids)
    assert torch.allclose(clean_out, hardened_out, atol=1e-6)


def test_layer_norm_tmr_no_double_wrap(tiny_transformer: nn.Module) -> None:
    LayerNormTMR().apply(tiny_transformer)
    n_after_first = sum(1 for m in tiny_transformer.modules() if isinstance(m, TMRLayerNorm))
    second = LayerNormTMR()
    second.apply(tiny_transformer)
    assert len(second.wrapped_paths) == 0
    n_after_second = sum(1 for m in tiny_transformer.modules() if isinstance(m, TMRLayerNorm))
    assert n_after_first == n_after_second


def test_layer_norm_tmr_report_has_overhead(tiny_transformer: nn.Module) -> None:
    strategy = LayerNormTMR()
    strategy.apply(tiny_transformer)
    report = strategy.collect_report()
    assert report.strategy_name == "layernorm_tmr"
    assert report.overhead_bytes > 0
    assert report.extra["n_wrapped_modules"] > 0
