# ml4paleo Base Image
#
# Build a single image that can run the Flask web app plus the background job
# runners used by docker-compose.

FROM python:3.12-slim-bookworm

LABEL maintainer="Jordan Matelsky <ml4paleo@matelsky.com>"
LABEL description="ml4paleo: A web application for paleontological image segmentation."

# Install uv from the official image and keep the system packages needed by
# scientific Python wheels that may need local compilation.
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /uvx /bin/

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ zlib1g-dev libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

ENV UV_NO_DEV=1
ENV UV_PYTHON_DOWNLOADS=never

WORKDIR /ml4paleo

# Install third-party dependencies before copying the whole repo to improve
# Docker layer reuse when application code changes.
COPY pyproject.toml uv.lock /ml4paleo/
RUN uv sync --locked --group dicom --no-install-project

# Copy the application source and install the project itself.
COPY . /ml4paleo
RUN uv sync --locked --group dicom \
    && uv pip install --python .venv/bin/python gunicorn

# Expose the synced environment to the runtime entrypoints used in compose.
ENV PATH="/ml4paleo/.venv/bin:$PATH"

WORKDIR /ml4paleo/webapp
RUN mkdir -p volume

CMD ["gunicorn", "--bind", ":5000", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info", "main:app", "--timeout", "300"]
