import logging
import pathlib
from typing import Optional, Tuple

import numpy as np
from config import CONFIG
from intern.utils.parallel import block_compute
from job import JobStatus, JSONFileUploadJobManager, UploadJob
from joblib import Parallel, delayed

from ml4paleo.segmentation import Segmenter3D
from ml4paleo.volume_providers import VolumeProvider

log = logging.getLogger(__name__)


def get_job_manager() -> JSONFileUploadJobManager:
    """
    Return the job manager.
    """
    return JSONFileUploadJobManager("jobs.json")


def get_next_dataset_to_segment() -> Optional[UploadJob]:
    """
    Return the next dataset that has been converted but not yet segmented.
    """
    job_manager = get_job_manager()
    next_job = job_manager.get_jobs_by_status(JobStatus.CONVERTED)
    if len(next_job) == 0:
        return None
    return next_job[0]


def segment_chunk(
    xs: Tuple[int, int],
    ys: Tuple[int, int],
    zs: Tuple[int, int],
    volume_provider: VolumeProvider,
    segmenter: Segmenter3D,
) -> np.ndarray:
    """
    Segment a chunk of a job.
    """
    # Get the volume for the chunk:
    volume = volume_provider[xs[0] : xs[1], ys[0] : ys[1], zs[0] : zs[1]]
    # Segment the volume:
    segmentation = segmenter.segment(volume)
    return segmentation


def segment_job(job: UploadJob) -> None:
    ...
