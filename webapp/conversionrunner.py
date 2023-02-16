import logging
import pathlib
import time
from typing import Optional

from config import CONFIG
from job import JobStatus, JSONFileUploadJobManager, UploadJob

from ml4paleo.volume_providers import ImageStackVolumeProvider
from ml4paleo.volume_providers.io import export_zarr_array

logging.basicConfig(level=logging.INFO)


def get_job_manager() -> JSONFileUploadJobManager:
    """
    Return the job manager.
    """
    return JSONFileUploadJobManager("volume/jobs.json")


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
    logging.info("Getting next dataset to convert...")
    next_job = get_next_uploaded_dataset_to_convert()
    if next_job is None:
        logging.info("No datasets to convert")
        return
    logging.info("Converting dataset %s", next_job.id)
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
    logging.info("Finished converting dataset %s", next_job.id)
    next_job.complete_convert()
    job_manager.update_job(next_job.id, next_job)
    logging.info("Updating job %s", next_job.id)


if __name__ == "__main__":
    while True:
        convert_next()
        time.sleep(CONFIG.job_poll_sec)
