import pathlib
from typing import Optional
from ml4paleo.volume_providers import ImageStackVolumeProvider
from ml4paleo.volume_providers.io import export_zarr_array
from config import CONFIG
from job import UploadJob, JobStatus, JSONFileUploadJobManager

import logging

log = logging.getLogger(__name__)


def get_job_manager() -> JSONFileUploadJobManager:
    """
    Return the job manager.
    """
    return JSONFileUploadJobManager("jobs.json")


def get_next_uploaded_dataset_to_convert() -> Optional[UploadJob]:
    """
    Return the next dataset that has been uploaded but not yet converted.
    """
    job_manager = get_job_manager()
    next_job = job_manager.get_jobs_by_status(JobStatus.UPLOADED)
    if len(next_job) == 0:
        return None
    return next_job[0]


def convert_next():
    """
    Convert the next dataset that has been uploaded but not yet converted.
    """
    job_manager = get_job_manager()
    log.info("Getting next dataset to convert...")
    next_job = get_next_uploaded_dataset_to_convert()
    if next_job is None:
        log.info("No datasets to convert")
        return
    log.info("Converting dataset %s", next_job.id)
    next_job.start_convert()
    job_manager.update_job(next_job.id, next_job)
    volume_provider = ImageStackVolumeProvider(
        pathlib.Path(CONFIG.upload_directory) / next_job.id, cache_size=0
    )
    export_zarr_array(
        volume_provider,
        pathlib.Path(CONFIG.chunked_directory) / next_job.id,
        chunk_size=CONFIG.chunk_size,
    )
    log.info("Finished converting dataset %s", next_job.id)
    next_job.complete_convert()
    job_manager.update_job(next_job.id, next_job)
    log.info("Updating job %s", next_job.id)


if __name__ == "__main__":
    convert_next()
