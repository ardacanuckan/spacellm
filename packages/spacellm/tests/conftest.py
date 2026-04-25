"""Shared pytest fixtures for SpaceLLM tests.

Tests are designed to run on a developer laptop or in CI without GPU access
and without downloading anything from HuggingFace Hub. The synthetic
``tiny_transformer`` fixture is enough exercise for protection strategies and
fault-injection plumbing.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn


@pytest.fixture
def small_fp32_tensor() -> torch.Tensor:
    """A 64-element FP32 tensor with deterministic content for bit-ops tests."""
    return torch.zeros(64, dtype=torch.float32)


@pytest.fixture
def small_int8_tensor() -> torch.Tensor:
    """A 64-element INT8 tensor for byte-granular fault tests."""
    return torch.zeros(64, dtype=torch.int8)


@pytest.fixture
def tiny_transformer() -> nn.Module:
    """A small encoder-only transformer.

    Two layers, two heads, embedding dimension 16, vocabulary 100. Small enough
    to forward in microseconds and exercise all attention/FFN/embedding code
    paths a real LLM would touch.
    """
    torch.manual_seed(0)

    class TinyTransformer(nn.Module):
        def __init__(
            self, dim: int = 16, heads: int = 2, layers: int = 2, vocab: int = 100
        ) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                batch_first=True,
                dim_feedforward=dim * 4,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.head = nn.Linear(dim, vocab)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.embed(x)
            x = self.encoder(x)
            return self.head(x)

    return TinyTransformer().eval()


@pytest.fixture
def tiny_input_ids() -> torch.Tensor:
    """A small batch of token IDs compatible with ``tiny_transformer``."""
    torch.manual_seed(0)
    return torch.randint(0, 100, (2, 8))
