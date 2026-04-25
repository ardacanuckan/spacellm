# `spacellm` Python framework

The Python core of SpaceLLM. PyTorch-style namespace, Apache-2.0.

```bash
pip install spacellm        # core (torch + numpy)
pip install spacellm[hf]    # + huggingface stack
pip install spacellm[all]   # + profiling + visualization extras
```

## 60-second example

```python
import torch
import spacellm as sl

env = sl.environments.StatisticalEnvironment(rate_per_bit_per_s=1e-6, seed=0)

weights = {"w": torch.randn(1024)}
env.step(dt=60.0)
events = env.sample_faults(weights, dt=60.0)
print(f"injected {len(events)} bit-flips in 60 s")
```

Apply protection (five composable strategies ship today):

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct").eval()

hardened = sl.harden(
    model,
    strategies=[
        sl.protection.SelectiveTMR(top_k_percent=5),
        sl.protection.AttentionChecksum(),
        sl.protection.LayerNormTMR(),
        sl.protection.EmbeddingChecksum(),
        sl.protection.KVCacheParity(),
    ],
)
```

See the repository root [`README.md`](../../README.md) for the full project context, and [`packages/docs`](../docs) for the documentation site.

## Layout

```
src/spacellm/
├── _internal/        bit-level utilities, RNG (private API)
├── environments/     fault sources, orbits, devices, MCU
├── nn/               PyTorch-style hardened modules (TMRLinear, TMRLayerNorm)
├── protection/       composable strategies (TMR, ABFT, parity)
├── runtime/          spacellm.harden() and HardenedModel
├── profiling/        static + sensitivity layer reports
├── observability/    SQLite-backed run DB + HTML reports
├── bench/            silent_error_rate, bench_protection, BenchResult
├── validation/       Weibull validation harness with ECSS envelope
└── cli.py            spacellm CLI
```

## Status

`0.3.0.dev0`. Pre-1.0; API may change. See `CHANGELOG.md` in the repo root.
