from typing import Optional
from job import UploadJob
from config import CONFIG
import pathlib


def get_latest_segmentation_id(job: UploadJob):
    """
    Get the latest segmentation for the job.
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
    """
    zarr_path = pathlib.Path(CONFIG.model_directory) / job.id
    if not zarr_path.exists():
        return None
    # Get the latest segmentation (the last one in the list)
    segmentation_path = sorted(zarr_path.glob("*.model"))[-1]
    return segmentation_path
