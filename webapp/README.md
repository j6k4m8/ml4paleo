# ml4paleo Web Application

<img width="865" alt="image" src="https://user-images.githubusercontent.com/693511/219508273-79bf2ec0-6b6d-456b-9110-9e4bd4304612.png">

This directory contains the code for the web application companion to the `ml4paleo` Python package. In general, all of the capabilities of the web application are available in the Python package, but the web application provides a no-code interface for users who are not familiar with Python.

See the bottom of this document for a list of routes.

## Technical Overview

The web application is built with Flask. The routes of the web app are defined in [`webapp/main.py`](webapp/main.py). This calls out to the `ml4paleo` package for the actual processing, though compute-intensive jobs are run in three separate processes to avoid blocking the web server.

### Job Runners

There are three separate processes that run in the background to handle compute-intensive jobs. These are:

#### `conversionrunner.py`

This process handles image conversion jobs. It is responsible for converting images from the user's upload to the Zarr format used by the segmentation model. For more information, see the header documentation in [`conversionrunner.py`](conversionrunner.py).

This job is mostly IO-bound, so it is not a bottleneck for the other processes.

#### `segmentrunner.py`

This process handles segmentation jobs. It is responsible for running the segmentation model on the user's data. For more information, see the header documentation in [`segmentrunner.py`](segmentrunner.py).

This job is mostly compute-bound, so it is a bottleneck for the other processes. It is also the most computationally intensive of the three processes. Depending on the segmentation model used, this process can take a very long time to complete, and may saturate available CPU or GPU resources. If the server is encountering performance issues, it may be necessary to run this process on a separate machine.

#### `meshrunner.py`

This process handles mesh generation jobs. It is responsible for generating meshes from the segmentation results. For more information, see the header documentation in [`meshrunner.py`](meshrunner.py).

This job is mostly memory-bound. If the server is swapping or OOMing, this job may be responsible.

### Installation

The `ml4paleo` library should be installed before installing the web application. See the [main README](../README.md) for instructions on how to install the library.

### Usage

The web application can be run **from the top directory** with the following command:

```bash
poetry run python3 ./main.py
```

This will start the web server on port 5000. The web application can be accessed at `http://localhost:5000`.

Note that for production use, it is recommended to run the web application with a production uWSGI tool, and behind a reverse proxy such as Nginx.

To run the workers, each of the three worker scripts should be run in a separate shell. For example:

```bash
# Shell 1:
poetry run python3 ./conversionrunner.py
# Shell 2:
poetry run python3 ./segmentrunner.py
# Shell 3:
poetry run python3 ./meshrunner.py
```

## Routes

Note that these routes are organized by their "job," not by their HTTP method or ontology (i.e., API vs UI).

### Upload Routes

#### `GET` `/`

Render the home page.

#### `POST` `/api/job/new`

Create a new job. This is the first step in the workflow. This route accepts a JSON body.

#### `POST` `/api/upload`

Upload a file to the server. This is used for uploading images to the server for processing. This route accepts a file upload.

#### `POST` `/api/job/status`

Get the status of a job. This route accepts a JSON body.

#### `GET` `/api/job/<job_id>/status`

Get the status of a job. No body is required.

#### `POST` `/api/job/status/upload-complete`

Notify the server that an upload is complete. This route accepts a JSON body.

#### `GET` `/job/<job_id>`

Render the job page. This is the main page for a job.

### Annotation Routes

#### `GET` `/api/annotate/<job_id>/images/next`

Get the next image to annotate. This is randomly selected.

#### `POST` `/api/annotate/<job_id>/data/submit`

Submit the annotation data for a job. This route accepts a JSON body with base64-encoded PNG images of the image and its user-generated mask.

#### `POST` `/api/annotate/<job_id>/data/predict`

Requests the server to generate a prediction for the next image to annotate. This route accepts a JSON body with base64-encoded PNG images of the image.

#### `GET` `/job/annotate/<job_id>`

Render the annotation page. This is the application for annotating images.

### Segmentation Routes

#### `POST` `/api/job/<job_id>/retrain`

Trigger a retraining of the segmentation model.

#### `POST` `/api/job/<job_id>/start`

Start a segmentation job.

#### `POST` `/api/job/<job_id>/mesh`

Queue a mesh generation job.

### Visualization Routes

#### `GET` `/api/job/<job_id>/zarr/<path>`

Get a file from the Zarr store for a job.

#### `GET` `/api/job/<job_id>/segmentation/<seg_id>/zarr/<path>`

Get a file from the Zarr store for a segmentation.

#### `GET` `/api/job/<job_id>/segmentation/<seg_id>/obj/<path>`

Get an OBJ mesh file for a segmentation ID.

### Download Routes

#### `GET` `/job/<job_id>/download`

Render the download page. This is the user-facing page for downloading the results of a job.

#### `GET` `/api/job/<job_id>/segmentation/<seg_id>/download/zarr`

Download a zip archive of the Zarr store for a segmentation.

#### `GET` `/api/job/<job_id>/segmentation/<seg_id>/download/png`

Download a zip archive of the PNG image stack for a segmentation.

#### `GET` `/api/job/<job_id>/meshes/<seg_id>/download`

Download a zip archive of the combined STL mesh files for a segmentation.
