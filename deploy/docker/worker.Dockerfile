# ml4paleo v2 job worker (CPU).
#
# Build from the repository root:
#   docker build -f deploy/docker/worker.Dockerfile -t ml4paleo-worker .
#
# On a machine with an NVIDIA GPU, the same image sees the GPU when the
# container is started with GPU access (the compose `gpu` profile does this).

FROM ghcr.io/astral-sh/uv:0.12.23 AS uv

FROM python:3.12-slim-bookworm

# Upgrade first so the image picks up Debian security fixes released after the
# base image was built.
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

ENV UV_NO_DEV=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install third-party dependencies first, for better layer caching. uv and its
# download cache are mounted only for these steps, so neither ships in the
# image.
COPY pyproject.toml uv.lock ./
COPY server/pyproject.toml server/
COPY worker/pyproject.toml worker/
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --package ml4paleo-worker --no-install-workspace

COPY ml4paleo ml4paleo
COPY worker worker
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --package ml4paleo-worker --no-editable

# Workers parse untrusted uploads, so they never run as root.
RUN useradd --system --uid 10002 --no-create-home ml4paleo-worker
USER ml4paleo-worker
ENV PATH="/app/.venv/bin:$PATH"

CMD ["ml4paleo-worker"]
