# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/ardacanuckan/spacellm/compare/HEAD...HEAD
[0.3.0-dev]: https://github.com/ardacanuckan/spacellm/releases/tag/v0.3.0-dev
[0.1.0-dev]: https://github.com/ardacanuckan/spacellm/releases/tag/v0.1.0-dev
