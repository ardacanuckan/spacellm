"""Tests for ``spacellm.protection.attention``."""

from __future__ import annotations

import torch
from torch import nn

from spacellm._internal.bitops import flip_bit
from spacellm.protection import AttentionChecksum


def test_attention_checksum_finds_attention_modules(
    tiny_transformer: nn.Module,
) -> None:
    strategy = AttentionChecksum()
    strategy.apply(tiny_transformer)
    report = strategy.collect_report()
    # TinyTransformer has 2 layers, each with a self_attn — we expect at least 2.
    assert report.extra["n_protected_modules"] >= 2.0


def test_attention_checksum_reports_zero_detections_when_clean(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    strategy = AttentionChecksum()
    strategy.apply(tiny_transformer)
    tiny_transformer(tiny_input_ids)
    report = strategy.collect_report()
    assert report.n_corruptions_detected == 0
    assert report.n_checks_performed > 0


def test_attention_checksum_detects_exponent_bit_flip(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    strategy = AttentionChecksum()
    strategy.apply(tiny_transformer)
    # First forward to populate baseline check counters.
    tiny_transformer(tiny_input_ids)
    # Locate an attention parameter and corrupt its top exponent bit.
    target_param: torch.nn.Parameter | None = None
    for name, p in tiny_transformer.named_parameters():
        if "self_attn" in name and p.numel() > 0:
            target_param = p
            break
    assert target_param is not None
    # Bit 30 of fp32 = exponent MSB-1 — guaranteed to shift magnitude dramatically.
    flip_bit(target_param.data, flat_index=0, bit_position=30)
    tiny_transformer(tiny_input_ids)
    report = strategy.collect_report()
    assert report.n_corruptions_detected >= 1


def test_attention_checksum_detach_removes_hooks(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    strategy = AttentionChecksum()
    strategy.apply(tiny_transformer)
    strategy.detach()
    pre_detach_checks = strategy.collect_report().n_checks_performed
    tiny_transformer(tiny_input_ids)
    post_detach_checks = strategy.collect_report().n_checks_performed
    assert post_detach_checks == pre_detach_checks  # no new checks ran


def test_attention_checksum_reset_clears_counters(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    strategy = AttentionChecksum()
    strategy.apply(tiny_transformer)
    tiny_transformer(tiny_input_ids)
    assert strategy.collect_report().n_checks_performed > 0
    strategy.reset()
    assert strategy.collect_report().n_checks_performed == 0
    assert strategy.collect_report().n_corruptions_detected == 0
