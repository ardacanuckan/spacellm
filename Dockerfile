# SpaceLLM reproducibility image.
#
# Goal: anyone — including a future on-orbit-data-center customer who
# wants to verify the bench numbers themselves — can pull this image,
# run the test suite, and reproduce the headline charts in
# `docs/assets/`. No mystery local state.
#
#   docker build -t spacellm:dev .
#   docker run --rm spacellm:dev pytest packages/spacellm -q
#   docker run --rm -v $PWD/out:/out spacellm:dev \
#       python benchmarks/qwen_eval.py --output-dir /out
#
# The image is intentionally based on the official Astral uv image so
# the lockfile resolution matches CI byte-for-byte.

FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH

WORKDIR /workspace

# Install only the dependency manifests first so Docker can cache the
# (slow) torch resolve across iterations on source code.
COPY pyproject.toml uv.lock ./
COPY packages/spacellm/pyproject.toml ./packages/spacellm/pyproject.toml
COPY packages/docs/pyproject.toml ./packages/docs/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --all-packages --no-install-workspace

COPY . .

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --all-extras --all-packages

# Default command: a one-line health check that imports the package
# and prints the version. Override with whatever you actually want
# to run (pytest, examples/04_qwen_protected.py, etc.).
CMD ["python", "-c", "import spacellm; print(f'spacellm {spacellm.__version__} ready')"]
