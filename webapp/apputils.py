"""
This file stores utilities and helper methods that are commonly used by other
files in the web application directory. Methods SHOULD go here if they are
used by more than one file in the web application directory. Methods SHOULD
NOT go here if they are only used by one file in the web application directory
or if they would be useful to other parts of the application (and should
therefore live in the ml4paleo package).

"""

import json
from typing import Any, Optional, Union

from flask import request
from job import UploadJob
from config import CONFIG
import pathlib
import numpy as np
from PIL import Image
from ml4paleo.volume_providers import ZarrVolumeProvider


def _annotation_directory(job_or_id: Union[UploadJob, str]) -> pathlib.Path:
    """
    Return the directory that stores annotation files for a job.
    """
    job_id = job_or_id.id if isinstance(job_or_id, UploadJob) else str(job_or_id)
    return pathlib.Path(CONFIG.training_directory) / job_id


def get_annotation_pairs(
    job_or_id: Union[UploadJob, str],
) -> list[tuple[pathlib.Path, pathlib.Path, Optional[pathlib.Path]]]:
    """
    Return saved annotation image/mask pairs, plus optional metadata files.
    """
    annotation_dir = _annotation_directory(job_or_id)
    if not annotation_dir.exists():
        return []

    pairs = []
    for img_path in sorted(annotation_dir.glob(f"{CONFIG.training_img_prefix}*.png")):
        timestamp = img_path.stem.removeprefix(CONFIG.training_img_prefix)
        seg_path = annotation_dir / f"{CONFIG.training_seg_prefix}{timestamp}.png"
        if not seg_path.exists():
            continue
        meta_path = annotation_dir / f"{CONFIG.training_meta_prefix}{timestamp}.json"
        pairs.append((img_path, seg_path, meta_path if meta_path.exists() else None))
    return pairs


def count_annotation_samples(job_or_id: Union[UploadJob, str]) -> int:
    """
    Count complete annotation samples for a job.
    """
    return len(get_annotation_pairs(job_or_id))


def load_annotation_sample_metadata(
    meta_path: Optional[pathlib.Path],
) -> Optional[dict[str, Any]]:
    """
    Load one saved annotation metadata JSON file if it exists.
    """
    if meta_path is None or not meta_path.exists():
        return None

    with meta_path.open("r") as f:
        metadata = json.load(f)

    return metadata if isinstance(metadata, dict) else None


def load_annotation_source_slice(
    job_or_id: Union[UploadJob, str],
    sample_metadata: dict[str, Any],
) -> np.ndarray:
    """
    Reconstruct the raw annotated XY slice from saved cutout metadata.

    The returned slice uses the native volume XY orientation used by the batch
    segmenter. Browser-displayed annotation PNGs are the transpose of this.
    """
    requested_x, requested_y, _requested_z = tuple(
        int(v) for v in sample_metadata["requested_shape_xyz"]
    )
    actual_x, actual_y, actual_z = tuple(
        int(v) for v in sample_metadata["cutout_shape_xyz"]
    )
    x_start, y_start, z_start = tuple(
        int(v) for v in sample_metadata["cutout_origin_xyz"]
    )
    x_pad_before, y_pad_before, z_pad_before = tuple(
        int(v) for v in sample_metadata["padding_before_xyz"]
    )
    annotated_local_z_index = int(sample_metadata["annotated_local_z_index"])
    source_local_z_index = annotated_local_z_index - z_pad_before

    job_id = job_or_id.id if isinstance(job_or_id, UploadJob) else str(job_or_id)
    volume_provider = ZarrVolumeProvider(pathlib.Path(CONFIG.chunked_directory) / job_id)
    source_slice = np.zeros((requested_x, requested_y), dtype=volume_provider.dtype)

    if 0 <= source_local_z_index < actual_z:
        raw_slice = np.asarray(
            volume_provider[
                x_start : x_start + actual_x,
                y_start : y_start + actual_y,
                z_start + source_local_z_index,
            ]
        )
        source_slice[
            x_pad_before : x_pad_before + actual_x,
            y_pad_before : y_pad_before + actual_y,
        ] = raw_slice

    return source_slice


def get_mesh_directory(job: UploadJob, seg_id: str) -> pathlib.Path:
    """
    Return the directory that stores mesh artifacts for a segmentation.
    """
    return pathlib.Path(CONFIG.meshed_directory) / job.id / seg_id


def get_mesh_files(
    job: UploadJob,
    seg_id: str,
    pattern: str,
) -> list[pathlib.Path]:
    """
    Return sorted mesh artifact files for a segmentation.
    """
    mesh_path = get_mesh_directory(job, seg_id)
    if not mesh_path.exists():
        return []
    return sorted(path for path in mesh_path.glob(pattern) if path.is_file())


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


def get_latest_mesh_id(job: UploadJob) -> Optional[str]:
    """
    Get the latest meshed segmentation ID for the job, or None if absent.

    Meshes are stored under the meshed directory as one subdirectory per
    segmentation timestamp/ID.
    """
    mesh_path = pathlib.Path(CONFIG.meshed_directory) / job.id
    if not mesh_path.exists():
        return None

    mesh_dirs = sorted(path for path in mesh_path.iterdir() if path.is_dir())
    for mesh_dir in reversed(mesh_dirs):
        if any(mesh_dir.glob("*.combined.stl")):
            return mesh_dir.name
    return None


def get_latest_mesh_obj_path(job: UploadJob) -> Optional[pathlib.Path]:
    """
    Return one representative OBJ mesh for the latest valid mesh output.
    """
    latest_mesh_id = get_latest_mesh_id(job)
    if latest_mesh_id is None:
        return None

    obj_files = get_mesh_files(job, latest_mesh_id, "*.combined.obj")
    if len(obj_files) == 0:
        return None

    for obj_file in obj_files:
        if obj_file.name == "255.combined.obj":
            return obj_file
    return obj_files[0]


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

    protocol = "https" # request.url.split(":")[0]
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
            "source": f"zarr://{protocol}://{request.host}/api/job/{job.id}/segmentation/{seg_id}/zarr/",
            "tab": "source",
            "name": f"segmentation {seg_id}",
        }
        jsondata["layers"].append(seg_layer)

    mesh_obj_path = get_latest_mesh_obj_path(job)
    if mesh_obj_path is not None:
        mesh_seg_id = mesh_obj_path.parent.name

        mesh_layer = {
            "type": "mesh",
            "source": {
                "url": f"obj://{protocol}://{request.host}/api/job/{job.id}/segmentation/{mesh_seg_id}/obj/{mesh_obj_path.name}",
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


def normalize_annotation_volume(
    vol: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.5,
) -> tuple[np.ndarray, dict]:
    """
    Normalize a sampled annotation subvolume into a high-contrast uint8 volume.

    The annotation UI and training round-trip go through a browser canvas, so
    they are effectively limited to 8-bit display data. We therefore choose a
    robust intensity window server-side before serializing to PNG.
    """
    vol = np.asarray(vol)
    finite_values = vol[np.isfinite(vol)]
    if finite_values.size == 0:
        return np.zeros(vol.shape, dtype=np.uint8), {
            "source_min": 0.0,
            "source_max": 0.0,
            "window_min": 0.0,
            "window_max": 0.0,
        }

    source_min = float(finite_values.min())
    source_max = float(finite_values.max())

    # Prefer non-zero voxels when possible so padded borders and sparse black
    # backgrounds don't collapse the display range.
    nonzero_values = finite_values[finite_values != 0]
    window_values = nonzero_values if nonzero_values.size >= 128 else finite_values

    window_min = float(np.percentile(window_values, lower_percentile))
    window_max = float(np.percentile(window_values, upper_percentile))

    if window_max <= window_min:
        window_min = source_min
        window_max = source_max

    if window_max <= window_min:
        fill_value = 0 if source_max <= 0 else 255
        return np.full(vol.shape, fill_value, dtype=np.uint8), {
            "source_min": source_min,
            "source_max": source_max,
            "window_min": window_min,
            "window_max": window_max,
        }

    vol_float = vol.astype(np.float32, copy=False)
    normalized = np.clip((vol_float - window_min) / (window_max - window_min), 0.0, 1.0)
    normalized = (normalized * 255.0).round().astype(np.uint8)
    return normalized, {
        "source_min": source_min,
        "source_max": source_max,
        "window_min": window_min,
        "window_max": window_max,
    }


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
