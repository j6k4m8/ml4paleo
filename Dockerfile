# ml4paleo Base Image
#
# Build a single image that can run the Flask web app plus the background job
# runners used by docker-compose.

FROM ghcr.io/astral-sh/uv:0.12.23 AS uv

FROM python:3.12-slim-bookworm

LABEL maintainer="Jordan Matelsky <ml4paleo@matelsky.com>"
LABEL description="ml4paleo: A web application for paleontological image segmentation."

# Keep the system packages needed by scientific Python wheels that may need
# local compilation. uv is mounted from the official image only for the
# install steps below, so it never ships in the runtime image.
# Upgrade first so the image picks up Debian security fixes released after the
# base image was built.
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends gcc g++ zlib1g-dev libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_NO_DEV=1
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /ml4paleo

# Install third-party dependencies before copying the whole repo to improve
# Docker layer reuse when application code changes.
COPY pyproject.toml uv.lock /ml4paleo/
COPY server/pyproject.toml /ml4paleo/server/
COPY worker/pyproject.toml /ml4paleo/worker/
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    uv sync --locked --extra v1 --no-install-workspace

# Copy the application source and install the project itself.
COPY . /ml4paleo
RUN --mount=from=uv,source=/uv,target=/bin/uv \
    uv sync --locked --extra v1 \
    && uv pip install --python .venv/bin/python gunicorn

# Expose the synced environment to the runtime entrypoints used in compose.
ENV PATH="/ml4paleo/.venv/bin:$PATH"

WORKDIR /ml4paleo/webapp
RUN mkdir -p volume

CMD ["gunicorn", "--bind", ":5000", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "main:app", "--timeout", "300"]
