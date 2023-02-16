"""
This file stores utilities and helper methods that are commonly used by other
files in the web application directory. Methods SHOULD go here if they are
used by more than one file in the web application directory. Methods SHOULD
NOT go here if they are only used by one file in the web application directory
or if they would be useful to other parts of the application (and should
therefore live in the ml4paleo package).

"""

from typing import Optional
from job import UploadJob
from config import CONFIG
import pathlib


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
