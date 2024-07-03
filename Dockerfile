# ml4paleo Base Image

# This Dockerfile is used to build an image containing basic stuff to run a web
# application version of ml4paleo.
# This should be built from the top-level directory of the ml4paleo repository
# which is available at https://github.com/j6k4m8/ml4paleo.

FROM python:3.12-slim-bookworm
LABEL maintainer="Jordan Matelsky <ml4paleo@matelsky.com>"
LABEL description="ml4paleo: A web application for paleontological image segmentation."

# Install dependencies (poetry etc) and copy in the code. We need curl to install poetry.
RUN apt-get update
RUN apt-get install -y curl gcc g++ zlib1g-dev libjpeg-dev
RUN curl -sSL https://install.python-poetry.org | python3 -
# Add poetry to the path:
ENV PATH="${PATH}:/root/.local/bin"

# Copy in JUST the poetry files, and install the dependencies.
WORKDIR /ml4paleo
RUN poetry config virtualenvs.create false

# Copy in the code:
COPY . /ml4paleo
WORKDIR /ml4paleo

# Install the library.
RUN poetry install
RUN poetry run pip install scikit-image scikit-learn
RUN poetry run pip install zmesh

# Install the library.
RUN pip install -e ./

WORKDIR /ml4paleo/webapp
RUN mkdir -p volume

RUN pip install gunicorn

# Set the entrypoint to the command passed in.
ENTRYPOINT ["/bin/sh", "-c"]
