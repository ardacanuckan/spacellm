"""Tests for ``spacellm.profiling.state_dict``."""

from __future__ import annotations

import torch

from spacellm.profiling import state_dict_profile


def test_state_dict_profile_counts_params() -> None:
    sd = {
        "embed.weight": torch.zeros(100, 16),
        "head.weight": torch.zeros(16, 100),
        "head.bias": torch.zeros(100, dtype=torch.float16),
    }
    profile = state_dict_profile(sd)
    assert len(profile.layers) == 3
    by_name = {layer.layer_name: layer for layer in profile.layers}
    assert by_name["embed.weight"].n_parameters == 100 * 16
    assert by_name["embed.weight"].n_bits == 100 * 16 * 32
    assert by_name["head.bias"].n_bits == 100 * 16  # fp16 → 16 bits
    assert by_name["head.bias"].module_class == "float16"


def test_state_dict_profile_top_k_orders_by_param_count() -> None:
    sd = {
        "small": torch.zeros(8),
        "big": torch.zeros(1024, 1024),
        "medium": torch.zeros(64, 64),
    }
    profile = state_dict_profile(sd)
    top = profile.top_k(2)
    assert [layer.layer_name for layer in top] == ["big", "medium"]


def test_state_dict_profile_empty_dict() -> None:
    profile = state_dict_profile({})
    assert profile.layers == []
