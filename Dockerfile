# ml4paleo Base Image

# This Dockerfile is used to build an image containing basic stuff to run a web
# application version of ml4paleo.
# This should be built from the top-level directory of the ml4paleo repository
# which is available at https://github.com/j6k4m8/ml4paleo.

FROM python:3.11-alpine
LABEL maintainer="Jordan Matelsky <ml4paleo@matelsky.com>"
LABEL description="ml4paleo: A web application for paleontological image segmentation."

# Install dependencies (poetry etc) and copy in the code. We need curl to install poetry.
# Install curl and gcc
RUN apk add --no-cache curl gcc musl-dev linux-headers make automake gcc g++
# RUN curl -sSL https://install.python-poetry.org | python3 -
# Add poetry to the path:
ENV PATH="${PATH}:/root/.local/bin"

# Copy in JUST the poetry files, and install the dependencies.
COPY requirements.txt /ml4paleo/
WORKDIR /ml4paleo
RUN apk add py3-scipy
# RUN poetry config virtualenvs.create false
# RUN poetry install
RUN pip install -r requirements.txt

# Copy in the code:
COPY . /ml4paleo
WORKDIR /ml4paleo

# Install the library.
# RUN poetry install

# Install the library.
RUN pip install -e ./

WORKDIR /ml4paleo/webapp
RUN mkdir -p volume

RUN pip install gunicorn

# Set the entrypoint to the command passed in.
ENTRYPOINT ["/bin/sh", "-c"]