"""Tests for ``spacellm.protection.embedding``."""

from __future__ import annotations

import torch
from torch import nn

from spacellm._internal.bitops import flip_bit
from spacellm.protection import EmbeddingChecksum


def test_embedding_checksum_finds_embeddings(tiny_transformer: nn.Module) -> None:
    strategy = EmbeddingChecksum()
    strategy.apply(tiny_transformer)
    report = strategy.collect_report()
    # TinyTransformer has exactly one nn.Embedding (`.embed`).
    assert report.extra["n_protected_modules"] == 1.0


def test_embedding_checksum_clean_run_no_detections(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    strategy = EmbeddingChecksum()
    strategy.apply(tiny_transformer)
    tiny_transformer(tiny_input_ids)
    report = strategy.collect_report()
    assert report.n_corruptions_detected == 0
    assert report.n_checks_performed > 0


def test_embedding_checksum_detects_exponent_flip(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    strategy = EmbeddingChecksum()
    strategy.apply(tiny_transformer)
    tiny_transformer(tiny_input_ids)  # baseline check
    embed_weight = tiny_transformer.embed.weight
    flip_bit(embed_weight.data, flat_index=0, bit_position=30)  # exponent bit
    tiny_transformer(tiny_input_ids)
    report = strategy.collect_report()
    assert report.n_corruptions_detected >= 1


def test_embedding_checksum_detach_stops_checking(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    strategy = EmbeddingChecksum()
    strategy.apply(tiny_transformer)
    strategy.detach()
    pre = strategy.collect_report().n_checks_performed
    tiny_transformer(tiny_input_ids)
    post = strategy.collect_report().n_checks_performed
    assert pre == post


def test_embedding_checksum_reset_clears_counters(
    tiny_transformer: nn.Module,
    tiny_input_ids: torch.Tensor,
) -> None:
    strategy = EmbeddingChecksum()
    strategy.apply(tiny_transformer)
    tiny_transformer(tiny_input_ids)
    assert strategy.collect_report().n_checks_performed > 0
    strategy.reset()
    assert strategy.collect_report().n_checks_performed == 0


def test_embedding_checksum_skipped_on_models_without_embedding() -> None:
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))
    strategy = EmbeddingChecksum()
    strategy.apply(model)
    report = strategy.collect_report()
    assert report.extra["n_protected_modules"] == 0.0
