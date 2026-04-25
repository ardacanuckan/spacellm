# Architecture

## Module decomposition

```
spacellm/
├── types.py             RadiationStep, FaultEvent, RunRecord, DeviceModel
├── runtime/             sl.harden(), HardenedModel, ProtectionReport aggregator
├── environments/
│   ├── physics.py       weibull_cross_section, irpp_seu_rate_per_cell, petersen_fom
│   ├── orbits.py        OrbitProfile + LEO_ISS_NOMINAL, GEO_QUIET, MARS_TRANSIT
│   ├── devices.py       6 verified RT PolarFire DeviceModel entries
│   ├── physics_lite.py  PhysicsLiteEnvironment(device, orbit)
│   ├── mcu.py           MCUDistribution, MCUEnvironment cluster amplifier
│   └── statistical.py   StatisticalEnvironment for tests and tutorials
├── protection/
│   ├── tmr.py           SelectiveTMR (median vote on top-k Linear modules)
│   ├── attention.py     AttentionChecksum (Frobenius fingerprint)
│   ├── kv_cache.py      KVCacheParity (XOR row-parity)
│   ├── layer_norm.py    LayerNormTMR (TMR for nn.LayerNorm / nn.RMSNorm)
│   └── embedding.py     EmbeddingChecksum (token + position fingerprint)
├── nn/                  TMRLinear, TMRLayerNorm
├── _internal/bitops.py  flip_bit primitive (XOR via Tensor.view)
├── profiling/           static + sensitivity-driven layer reports
├── observability/       SQLite RunDB, Run context manager, HTML reports
├── bench/               silent_error_rate, bench_protection, BenchResult
├── validation/          WeibullValidationData, validate_against_measurements
└── cli.py               spacellm version | runs | show | profile
```

Public API stability follows the `__all__` declarations in each `__init__.py`. Anything under `spacellm._internal` is implementation detail and may change without semver constraints.

## Stack choices

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | tested on 3.11, 3.12, 3.13 in CI |
| ML framework | PyTorch 2.4+ | forward hooks and Tensor reinterpretation drive the protection layer |
| Build / package | hatchling, uv | reproducible, fast |
| Lint / type | ruff, mypy --strict | both clean across the source tree |
| Test runner | pytest | 217 / 217 passing |
| Run DB | SQLite | zero-config, ships in stdlib |
| Docs | MkDocs Material | builds with `--strict` |
| Web | Next.js 15 + Tailwind 4 | static export, no SSR runtime |

## Shipped surface (v0.3)

The current line ships everything the methodology preprint at `papers/spacellm-methodology-preprint.md` describes:

* `spacellm.types` data classes for `DeviceModel`, `OrbitProfile`, `RunRecord`, `FaultEvent`.
* The calibrated environment (`StatisticalEnvironment`, `PhysicsLiteEnvironment`, `MCUEnvironment`) with three primary-source orbit profiles and six RT PolarFire device profiles.
* All five protection strategies: `SelectiveTMR`, `AttentionChecksum`, `KVCacheParity`, `LayerNormTMR`, `EmbeddingChecksum`.
* `harden(model, strategies, environment)` runtime entry point and `HardenedModel` handle.
* Profiling, observability (SQLite RunDB), bench harness, validation harness, and CLI.

## Not shipped

The framework does not yet provide a fault-aware autograd path, a checkpoint scrubber, federated training across constellations, or a model-hub. These are tracked on the project roadmap but are not committed to a calendar.

Long-form physics context lives in the [physics primer](physics-primer.md). Beam-test extraction provenance lives in [beam-test-data.md](beam-test-data.md). The end-to-end orbital training runbook lives in [orbital-training.md](orbital-training.md).
