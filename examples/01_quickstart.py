"""Tiny end-to-end demo, build a transformer, harden it, inject flips, observe.

Run with:

    uv run python examples/01_quickstart.py

The script uses a synthetic 2-layer transformer (no Hugging Face Hub
download), so it runs on a CPU laptop in under a second.
"""

from __future__ import annotations

import torch
from torch import nn

import spacellm as sl


class TinyTransformer(nn.Module):
    def __init__(self, dim: int = 32, heads: int = 4, layers: int = 2, vocab: int = 200) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, batch_first=True, dim_feedforward=dim * 4,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.head = nn.Linear(dim, vocab)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(self.embed(x)))


def main() -> None:
    torch.manual_seed(0)
    model = TinyTransformer().eval()
    inputs = torch.randint(0, 200, (1, 16))

    print("baseline forward...")
    baseline = model(inputs)

    print("\nhardening with TMR (top-20 %) + attention checksum...")
    hardened = sl.harden(
        model,
        strategies=[
            sl.protection.SelectiveTMR(top_k_percent=20.0),
            sl.protection.AttentionChecksum(),
        ],
    )
    out = hardened(inputs)
    print(f"  hardened output shape: {tuple(out.shape)}")
    print(f"  baseline ↔ hardened max abs diff: {(out - baseline).abs().max().item():.2e}")

    print("\nattaching a 1e-3 bit-flip-per-bit-per-second environment...")
    env = sl.environments.StatisticalEnvironment(rate_per_bit_per_s=1e-3, seed=0)
    weights_to_perturb = {
        name: param.data
        for name, param in model.named_parameters()
        if param.numel() > 0
    }
    events = env.sample_faults(weights_to_perturb, dt=1.0)
    print(f"  injected {len(events)} bit-flips into model weights")

    print("\npost-injection telemetry:")
    out_after = hardened(inputs)
    for report in hardened.collect_reports():
        print(f"  {report}")
    print(f"  baseline ↔ post-injection max abs diff: {(out_after - baseline).abs().max().item():.2e}")


if __name__ == "__main__":
    main()
