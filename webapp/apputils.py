"""
This file stores utilities and helper methods that are commonly used by other
files in the web application directory. Methods SHOULD go here if they are
used by more than one file in the web application directory. Methods SHOULD
NOT go here if they are only used by one file in the web application directory
or if they would be useful to other parts of the application (and should
therefore live in the ml4paleo package).

"""

import json
from typing import Optional

from flask import request
from job import UploadJob
from config import CONFIG
import pathlib
import numpy as np
from PIL import Image


def get_latest_segmentation_id(job: UploadJob) -> Optional[str]:
    """
    Get the latest segmentation for the job, or None if it doesn't exist.

    Segmentation is stored as a zarr file in the segmented directory, and it
    is named after the job ID and the time at which it was generated. These
    segmentation files MAY be broken or partial if the segmentation process
    was interrupted, so the latest segmentation may not always be the one
    that should be used.

    Arguments:
        job (UploadJob): The job for which to get the latest segmentation.

    Returns:
        str: The name of the latest segmentation file, like "1234.zarr".
        None: If no segmentation file exists for the job.

    """
    zarr_path = pathlib.Path(CONFIG.segmented_directory) / job.id
    if not zarr_path.exists():
        return None
    # Get the latest segmentation (the last one in the list)
    segmentation_path = sorted(zarr_path.glob("*.zarr"))[-1]
    return segmentation_path.name


def get_latest_segmentation_model(job: UploadJob) -> Optional[pathlib.Path]:
    """
    Get the latest segmentation model for the job.

    Segmentation models are stored as pickles in the models directory, and
    they are named after the job ID and the time at which they were generated.
    These line up with the segmentation filenames in the segmented directory.

    There is no such thing as a broken or partial segmentation model, so the
    latest model is always the one that should be used, assuming the training
    data hasn't gotten worse.

    The only time this method should return None is if the segmentation model
    directory doesn't exist.

    Arguments:
        job (UploadJob): The job for which to get the latest model.

    Returns:
        pathlib.Path: The path to the latest segmentation model.
        None: if no model has been made yet.

    """
    zarr_path = pathlib.Path(CONFIG.model_directory) / job.id
    if not zarr_path.exists():
        return None
    # Get the latest segmentation (the last one in the list)
    segmentation_path = sorted(zarr_path.glob("*.model"))[-1]
    return segmentation_path


def create_neuroglancer_link(job: UploadJob, return_state: bool = False):
    """
    Create a neuroglancer link for the images.

    Arguments:
        job (UploadJob): The job for which to create the link.
        return_state (bool): If True, return the neuroglancer state as well.

    Returns:
        str: The neuroglancer link
        dict: The neuroglancer state, if `return_state` is True.
    """
    # Check for segmentation:

    protocol = request.url.split(":")[0]
    jsondata = {
        "layers": [
            {
                "type": "image",
                "source": f"zarr://{protocol}://{request.host}/api/job/{job.id}/zarr/",
                "tab": "source",
                "name": "zarr",
            }
        ]
    }
    zarr_path = pathlib.Path(CONFIG.segmented_directory) / job.id
    if zarr_path.exists():
        # Get the latest segmentation (the last one in the list)
        seg_id = get_latest_segmentation_id(job)

        # Create the neuroglancer layer:
        seg_layer = {
            "type": "segmentation",
            "source": f"zarr://http://{request.host}/api/job/{job.id}/segmentation/{seg_id}/zarr/",
            "tab": "source",
            "name": f"segmentation {seg_id}",
        }
        jsondata["layers"].append(seg_layer)

    mesh_path = pathlib.Path(CONFIG.meshed_directory) / job.id
    if mesh_path.exists():
        # Get the last (sorted) directory in the meshed directory
        mesh_seg_id = sorted(mesh_path.glob("*"))[-1].name

        mesh_layer = {
            "type": "mesh",
            "source": {
                "url": f"obj://http://{request.host}/api/job/{job.id}/segmentation/{mesh_seg_id}/obj/255.combined.obj",
                "transform": {
                    "matrix": [[0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
                    "outputDimensions": {
                        "d0": [1, "m"],
                        "d1": [1, "m"],
                        "d2": [1, "m"],
                    },
                    "inputDimensions": {"x": [1, "m"], "y": [1, "m"], "z": [1, "m"]},
                },
            },
            "tab": "source",
            "name": f"mesh",
        }
        jsondata["layers"].append(mesh_layer)
    jsondump = json.dumps(jsondata)

    if return_state:
        return f"https://neuroglancer.bossdb.io/#!{jsondump}", jsondata
    return f"https://neuroglancer.bossdb.io/#!{jsondump}"


def get_png_filmstrip(vol: np.ndarray):
    """
    Create an image filmstrip from a volume.

    Note that this assumes that the volume is in ZYX order. The filmstrip will
    concatenate the layers in the Y direction, so the width will be the same
    as the width of the volume, and the height will be the height of the volume
    times the number of layers.

    Arguments:
        vol (np.ndarray): The volume to create the filmstrip from.

    Returns:
        PIL.Image: The filmstrip as an Image

    """
    # Get the number of layers
    num_layers = vol.shape[0]
    # Get the width and height of each layer
    width, height = vol.shape[1:]
    # Create a new image
    filmstrip = Image.fromarray(np.zeros((height * num_layers, width), dtype=vol.dtype))
    # Loop through the layers
    for i in range(num_layers):
        # Get the layer
        layer = vol[i]
        # Get the start and end of the layer in the filmstrip
        start = i * height
        end = (i + 1) * height
        # Paste the layer into the filmstrip
        filmstrip.paste(Image.fromarray(layer), (0, start, width, end))
    return filmstrip
