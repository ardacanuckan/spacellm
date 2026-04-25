# Contributing to SpaceLLM

Thank you for considering a contribution. SpaceLLM is open and Apache-2.0; we welcome issues, PRs, and discussion.

## Quick links

- Architecture: [`packages/docs/docs/architecture.md`](packages/docs/docs/architecture.md)
- Roadmap: [`CHANGELOG.md`](CHANGELOG.md)
- Open questions: [`CONTRIBUTING.md`](CONTRIBUTING.md)

## Local setup (one command)

```bash
make install   # installs Python deps (uv) and JS deps (pnpm)
make test      # runs all tests
make lint      # ruff + biome
make typecheck # mypy + tsc
```

Prerequisites:

- **Python 3.13+**, pinned in `.python-version`. We recommend [`uv`](https://github.com/astral-sh/uv) for environment management.
- **Node 22 LTS**, pinned in `.nvmrc`. We use **pnpm 10** (enabled via `corepack`).
- **macOS, Linux, or WSL2.** Native Windows is not supported for development.

## Repository layout

```
packages/spacellm/   Python framework (PyTorch-style namespace)
packages/web/        Next.js 15 app (App Router, Tailwind 4, shadcn/ui)
packages/docs/       MkDocs Material site
examples/            Runnable Python examples
benchmarks/          Benchmark scripts (SpaceBench v0)
```

## Development workflow

1. Fork → branch from `main` (`feat/<short-name>`, `fix/<short-name>`, etc.).
2. Run `make install` once.
3. Implement + add tests. We require tests for non-trivial code paths.
4. `make test && make lint && make typecheck` must pass locally.
5. Commit using **Conventional Commits** (see below).
6. Push and open a PR. Fill in the PR template.

## Conventional Commits

Format: `<type>(<scope>): <subject>`

Types: `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `ci`, `perf`, `build`, `style`.

Common scopes: `spacellm`, `web`, `docs`, `bench`, `monorepo`, `ci`.

Examples:

```
feat(spacellm): add SelectiveTMR protection strategy
fix(spacellm): correct KV-cache parity check for batched inference
docs: clarify v0.7 training roadmap
ci: cache uv installs for faster CI
```

The first line is ≤ 72 characters. A blank line then a body is optional but recommended for non-trivial changes.

## Coding standards

### Python

- Type hints **everywhere**. We run `mypy --strict` on `packages/spacellm/src`.
- Format with `ruff format` (Black-compatible). Lint with `ruff check`.
- Public APIs documented with docstrings (NumPy or Google style, pick one and stay consistent within a file).
- No relative imports beyond a single `.` level.
- We follow **PEP 8** + **PEP 257** + **PEP 484**.

### TypeScript / Next.js

- Strict TypeScript (`"strict": true`).
- Format and lint with **Biome** (`biome check`).
- Server Components by default; mark client islands explicitly with `"use client"`.
- Tailwind 4 utilities preferred; component primitives via shadcn/ui.

## Testing

- Python: `pytest` with the test layout under `packages/spacellm/tests/`. Use small synthetic models (no HF Hub downloads) for CI.
- Web: `vitest` for unit tests. End-to-end tests are out of scope until v0.5.
- Aim for ≥ 80% coverage on the `protection/` and `environments/` modules, they're the heart of the project.

## Reporting bugs and security issues

- Functional bugs: open a GitHub issue with the `bug` label, a minimal reproducer, and your environment (`spacellm.__version__`, Python version, OS).
- Security: see [`SECURITY.md`](SECURITY.md). Please do not file public issues for vulnerabilities.

## Scientific accuracy

SpaceLLM is a research-grade tool. Every flux number, LET threshold, dose rate, or SEU cross-section that ends up in code or docs **must cite a primary source**. See `packages/docs/docs/physics-primer.md` for the established baseline. If you find an unsupported number, file an issue or fix it directly.

## Code of Conduct

By participating you agree to abide by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
