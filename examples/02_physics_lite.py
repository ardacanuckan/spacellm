"""Physics-calibrated bit-flip injection on a tiny transformer.

Demonstrates the v0.3 :class:`PhysicsLiteEnvironment` end-to-end against
a publicly-verified device profile:

1. Build a synthetic transformer.
2. Combine the verified Microchip RT PolarFire LSRAM device with a
   reference orbit (Weibull params straight from the Microchip 2020
   beam-test PDF).
3. Compare the mission-realistic SEU rate (per bit per second) for
   three different orbits.
4. Run an accelerated injection and report the protection telemetry.

Run with:

    uv run python examples/02_physics_lite.py
"""

from __future__ import annotations

import torch
from torch import nn

import spacellm as sl
from spacellm.environments import GEO_QUIET, LEO_ISS_NOMINAL, MARS_TRANSIT
from spacellm.environments.devices import POLARFIRE_LSRAM


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
    baseline = model(inputs)

    print("mission-realistic SEU rate per bit per second (Microchip RT PolarFire LSRAM):")
    for orbit in (LEO_ISS_NOMINAL, GEO_QUIET, MARS_TRANSIT):
        env = sl.environments.PhysicsLiteEnvironment(POLARFIRE_LSRAM, orbit)
        print(
            f"  {orbit.name:20s}  {env.mission_seu_rate_per_bit_per_s:.3e}  "
            f"(TID {orbit.tid_rate_gy_per_s:.2e} Gy/s)",
        )

    print("\nharden + accelerated MARS_TRANSIT injection...")
    sl.harden(
        model,
        strategies=[
            sl.protection.SelectiveTMR(top_k_percent=20.0),
            sl.protection.AttentionChecksum(),
            sl.protection.LayerNormTMR(),
        ],
    )
    env = sl.environments.PhysicsLiteEnvironment(
        POLARFIRE_LSRAM, MARS_TRANSIT, seed=0, time_acceleration=1e10,
    )
    weights = {n: p.data for n, p in model.named_parameters() if p.numel() > 0}
    events = env.sample_faults(weights, dt=1.0)
    print(
        f"  injected {len(events)} bit-flips in 1.0 s of accelerated time "
        f"(physical equivalent ≈ {1.0 / env.time_acceleration:.2e} s)",
    )

    out = model(inputs)
    print(f"  baseline ↔ post-injection max abs diff: {(out - baseline).abs().max().item():.2e}")


if __name__ == "__main__":
    main()
