"""Tests for ``spacellm.profiling``."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from spacellm.profiling import (
    LayerSensitivity,
    ProfileReport,
    sensitivity_profile,
    static_profile,
)


def test_static_profile_lists_every_module(tiny_transformer: nn.Module) -> None:
    report = static_profile(tiny_transformer)
    assert all(isinstance(layer, LayerSensitivity) for layer in report.layers)
    # At minimum: embed, encoder, encoder.layers.0, layer-internals, head
    assert len(report.layers) >= 5
    assert any(layer.layer_name == "embed" for layer in report.layers)
    assert any(layer.layer_name == "head" for layer in report.layers)


def test_static_profile_counts_bits(tiny_transformer: nn.Module) -> None:
    report = static_profile(tiny_transformer)
    head = next(layer for layer in report.layers if layer.layer_name == "head")
    assert head.module_class == "Linear"
    assert head.n_parameters > 0
    assert head.n_bits == head.n_parameters * 32  # default fp32


def test_static_profile_root_excluded_by_default(tiny_transformer: nn.Module) -> None:
    report = static_profile(tiny_transformer)
    assert all(layer.layer_name != "" for layer in report.layers)


def test_static_profile_root_included_when_asked(tiny_transformer: nn.Module) -> None:
    report = static_profile(tiny_transformer, include_root=True)
    assert any(layer.layer_name == "" for layer in report.layers)


def test_top_k_orders_by_attribute(tiny_transformer: nn.Module) -> None:
    report = static_profile(tiny_transformer)
    top3 = report.top_k(3, by="n_parameters")
    assert len(top3) <= 3
    # Strictly non-increasing
    counts = [layer.n_parameters for layer in top3]
    assert counts == sorted(counts, reverse=True)


def test_profile_report_save_json(tiny_transformer: nn.Module, tmp_path: Path) -> None:
    report = static_profile(tiny_transformer)
    out = report.save_json(tmp_path / "profile.json")
    assert out.exists()
    assert "layers" in out.read_text()


def test_sensitivity_profile_runs_without_error(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    def forward(model: nn.Module) -> torch.Tensor:
        return model(tiny_input_ids)

    report = sensitivity_profile(
        tiny_transformer,
        forward,
        n_flips_per_layer=2,
        seed=0,
    )
    assert isinstance(report, ProfileReport)
    # At least the head layer should have parameters and so receive a perturbation score.
    head = next(layer for layer in report.layers if layer.layer_name == "head")
    assert head.mean_perturbation is not None
    assert head.max_perturbation is not None
    assert head.max_perturbation >= head.mean_perturbation


def test_sensitivity_profile_restores_weights(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    """The profile must leave the model in its original state."""

    def forward(model: nn.Module) -> torch.Tensor:
        return model(tiny_input_ids)

    snapshot = {k: v.clone() for k, v in tiny_transformer.state_dict().items()}
    sensitivity_profile(tiny_transformer, forward, n_flips_per_layer=2, seed=0)
    after = tiny_transformer.state_dict()
    for k, v in snapshot.items():
        assert torch.equal(v, after[k]), f"layer '{k}' was not restored"
