# ml4paleo Base Image

# This Dockerfile is used to build an image containing basic stuff to run a web
# application version of ml4paleo.
# This should be built from the top-level directory of the ml4paleo repository
# which is available at https://github.com/j6k4m8/ml4paleo.

FROM python:3.11
LABEL maintainer="Jordan Matelsky <ml4paleo@matelsky.com>"
LABEL description="ml4paleo: A web application for paleontological image segmentation."

# Install dependencies (poetry etc) and copy in the code. We need curl to install poetry.
RUN apt-get update && apt-get install -y curl
RUN curl -sSL https://install.python-poetry.org | python3 -
# Add poetry to the path:
ENV PATH="${PATH}:/root/.local/bin"

# Copy in the code:
COPY . /ml4paleo
WORKDIR /ml4paleo

# Install the library.
RUN poetry install

WORKDIR /ml4paleo/webapp
RUN mkdir -p volume

# Set the entrypoint to the command passed in.
ENTRYPOINT ["/bin/bash", "-c"]