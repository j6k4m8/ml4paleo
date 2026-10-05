# ml4paleo v2 API server.
#
# Build from the repository root:
#   docker build -f deploy/docker/server.Dockerfile -t ml4paleo-server .

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
    uv sync --locked --package ml4paleo-server --no-install-workspace

COPY ml4paleo ml4paleo
COPY server server
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --package ml4paleo-server --no-editable

RUN useradd --system --uid 10001 --no-create-home ml4paleo
USER ml4paleo
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["ml4paleo-server", "serve", "--host", "0.0.0.0", "--port", "8000"]
