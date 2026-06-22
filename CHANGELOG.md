# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.0] - 2026-06-22

### Added
- `ABFTProtection` strategy and `spacellm.nn.ABFTLinear`: algorithm-based fault tolerance for the matrix multiply itself. Row/column weight checksums detect and correct a single corrupted weight element; an output checksum detects transient compute faults and replays. Two checksum vectors per layer instead of a 3× replica.
- Weight calibration path: `spacellm.environments.physics.fit_weibull` (dependency-free four-parameter Weibull fit) and `device_from_measurements` to build a `DeviceModel` from your own beam-test points.
- `spacellm.environments.cots`: process-node analogue `DeviceModel` estimates for COTS accelerators (Jetson Orin, Coral Edge TPU, 5 nm-class datacentre GPU), all `verified=False`.
- `spacellm.mission`: `estimate_mission_cost` turns an SEU rate into expected wasted energy per day, rollback count, and the Young-optimal checkpoint interval.
- `spacellm.availability`: `simulate_availability` discrete-event model of eclipse duty-cycle plus single-event latch-up, reporting availability, MTBF, and MTTR.
- `spacellm.reliability`: `compose_residual_rate` composes SECDED ECC, memory scrubbing, and compute ABFT into a residual silent-error rate with a per-layer improvement breakdown.
- `GradientGuard` protection strategy: detects (and optionally sanitises) silent corruption in gradients and optimizer state during training.
- `ExponentRangeGuard` protection strategy: clamps weights to their trusted value range, neutralising exponent-flip blow-ups for two scalars per tensor.
- `spacellm.validation.reconcile_seu_rate` / `reconcile_events`: post-flight reconciliation of predicted vs observed on-orbit upsets via a two-sided Poisson test.
- `spacellm.bench.compare_number_formats`: bit-flip sensitivity sweep across FP32/FP16/BF16/INT8.

### Changed
- Test suite expanded to 316 cases; mypy `--strict` and ruff remain clean across the source tree.

## [0.3.0-dev] - 2026-04

### Added
- Calibrated radiation environment: Weibull cross-section, IRPP integral, MCU clustering.
- Three reference orbit profiles with primary-source citations: `LEO_ISS_NOMINAL`, `GEO_QUIET`, `MARS_TRANSIT`.
- Six RT PolarFire `DeviceModel` entries (USRAM, LSRAM, four DFF data patterns), every Weibull parameter cited to the Microchip 2020 radiation report.
- Five composable protection strategies: `SelectiveTMR`, `AttentionChecksum`, `KVCacheParity`, `LayerNormTMR`, `EmbeddingChecksum`.
- `validate_against_measurements` Weibull validation harness with ECSS-Q-ST-60-15C-style envelope.
- `bench_protection`, `silent_error_rate`, `BenchResult` evaluation primitives.
- SQLite-backed `RunDB` with `Run` context manager and HTML report renderer.
- Static and sensitivity-driven layer profiling.
- `spacellm` CLI: `version`, `runs`, `show`, `profile`.
- Methodology preprint at `papers/spacellm-methodology-preprint.md` (12 pages IMRAD, 15 references).
- Three multi-panel evaluation dashboards in `docs/assets/`, regenerated nightly by `.github/workflows/bench.yml`.

## [0.1.0-dev]

### Added
- Monorepo skeleton with Apache-2.0 license, root tooling, and contribution guides.
- Python framework `packages/spacellm` with PyTorch-style namespace.
- Next.js 15 web app `packages/web` with App Router and Tailwind 4.
- MkDocs Material documentation site `packages/docs`.
- GitHub Actions CI for Python and web.

[Unreleased]: https://github.com/ardacanuckan/spacellm/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/ardacanuckan/spacellm/releases/tag/v0.4.0
[0.3.0-dev]: https://github.com/ardacanuckan/spacellm/releases/tag/v0.3.0-dev
[0.1.0-dev]: https://github.com/ardacanuckan/spacellm/releases/tag/v0.1.0-dev
