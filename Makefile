.PHONY: help install install-py install-js test test-py test-web lint lint-py lint-web format format-py format-web typecheck typecheck-py typecheck-web clean docs-serve docs-build

help:
	@echo "SpaceLLM monorepo — common targets"
	@echo ""
	@echo "  make install         install Python (uv) and JS (pnpm) deps"
	@echo "  make test            run all tests (Python + web)"
	@echo "  make lint            run all linters"
	@echo "  make format          run all formatters"
	@echo "  make typecheck       run mypy + tsc"
	@echo "  make docs-serve      preview docs locally"
	@echo "  make docs-build      build static docs site"
	@echo "  make clean           remove caches and build artifacts"
	@echo ""
	@echo "Per-stack targets are also available: install-py, install-js,"
	@echo "test-py, test-web, lint-py, lint-web, format-py, format-web,"
	@echo "typecheck-py, typecheck-web."

install: install-py install-js

install-py:
	uv sync --all-extras --all-packages

install-js:
	@if ! command -v pnpm >/dev/null 2>&1; then \
		echo "→ enabling pnpm via corepack"; \
		corepack enable && corepack prepare pnpm@10.5.0 --activate; \
	fi
	pnpm install

test: test-py test-web

test-py:
	uv run pytest

test-web:
	pnpm --filter web test --run

lint: lint-py lint-web

lint-py:
	uv run ruff check .

lint-web:
	pnpm --filter web lint

format: format-py format-web

format-py:
	uv run ruff format .
	uv run ruff check --fix .

format-web:
	pnpm --filter web format

typecheck: typecheck-py typecheck-web

typecheck-py:
	uv run mypy packages/spacellm/src

typecheck-web:
	pnpm --filter web typecheck

docs-serve:
	uv run --package spacellm-docs mkdocs serve -f packages/docs/mkdocs.yml

docs-build:
	uv run --package spacellm-docs mkdocs build -f packages/docs/mkdocs.yml

clean:
	rm -rf .venv node_modules \
		packages/web/.next packages/web/node_modules packages/web/dist \
		packages/docs/site \
		**/.ruff_cache **/.mypy_cache **/.pytest_cache \
		**/__pycache__ **/*.egg-info **/build **/dist
