# ml4paleo v2 API server.
#
# Build from the repository root:
#   docker build -f deploy/docker/server.Dockerfile -t ml4paleo-server .

FROM ghcr.io/astral-sh/uv:0.12.23 AS uv

# Neuroglancer, built from a pinned release (v2.41.2), served at /neuroglancer/.
# Only the built static files reach the final image.
FROM node:22-bookworm-slim AS neuroglancer
ARG NEUROGLANCER_COMMIT=e13f1f4c62918f2ea07b12f2116bdcb6767b1499
# Its test tooling would otherwise download browsers the build doesn't need.
ENV PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
RUN git init -q \
    && git remote add origin https://github.com/google/neuroglancer.git \
    && git fetch -q --depth 1 origin "$NEUROGLANCER_COMMIT" \
    && git checkout -q FETCH_HEAD
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund \
    && npm run build -- --no-typecheck --no-lint \
    && test -f dist/client/index.html

# The web app, built to static files.
FROM node:24-bookworm-slim AS web
WORKDIR /src/web
COPY web/package.json web/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm \
    npm ci --no-audit --no-fund
COPY web ./
RUN npm run build && test -f build/index.html

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

COPY --from=neuroglancer /src/dist/client /app/neuroglancer
COPY --from=web /src/web/build /app/web

RUN useradd --system --uid 10001 --no-create-home ml4paleo
USER ml4paleo
ENV PATH="/app/.venv/bin:$PATH" \
    M4P_NEUROGLANCER_DIR=/app/neuroglancer \
    M4P_WEB_DIR=/app/web

EXPOSE 8000
CMD ["ml4paleo-server", "serve", "--host", "0.0.0.0", "--port", "8000"]
