"""
The conversion runner is responsible for reading image stacks from upload jobs
and converting them to chunked zarr arrays. The images are read from the
upload directory (`CONFIG.upload_directory`) and the zarr arrays are written to
the chunked directory (`CONFIG.chunked_directory`). The conversion runner polls
every few seconds to get new jobs to convert.

If a job is in the `JobStatus.UPLOADED` state, it will be queued for automatic
conversion. The conversion runner uses the `ImageStackVolumeProvider` to read
images from the upload directory, and the `export_zarr_array` function to write
the zarr array to the chunked directory. The conversion runner will update the
job status to `JobStatus.CONVERTING` when it starts converting a job, and to
`JobStatus.CONVERTED` when it finishes converting a job. If the conversion job
fails, the job status will be set to `JobStatus.CONVERT_ERROR`.

"""

import logging
import pathlib
import time
from typing import Any, Optional

from config import CONFIG
from job import JobStatus, JSONFileUploadJobManager, UploadJob

from ml4paleo.volume_providers import ImageStackVolumeProvider
from ml4paleo.volume_providers.io import export_zarr_array

logging.basicConfig(level=logging.INFO)


def get_job_manager() -> JSONFileUploadJobManager:
    """
    Return the job manager that will be used to read and write jobs.

    This is a convenience function so that we can easily change the job manager
    implementation without having to change the rest of the code.

    Arguments:
        None

    Returns:
        JobManager: The job manager that will be used to read and write jobs.

    """
    return JSONFileUploadJobManager("volume/jobs.json")


def get_next_uploaded_dataset_to_convert() -> Optional[UploadJob]:
    """
    Return the next dataset that has been uploaded but not yet converted.

    Arguments:
        None

    Returns:
        UploadJob: The next dataset that needs to be converted.
        None: If there are no datasets that need to be converted.

    """
    job_manager = get_job_manager()
    next_job = job_manager.get_jobs_by_status(JobStatus.UPLOADED)
    if len(next_job) == 0:
        return None
    return next_job[0]


def convert_next():
    """
    Convert the next dataset that has been uploaded but not yet converted.

    This is the function that actually performs the conversion.

    Arguments:
        None

    Returns:
        None

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

    def _progress_callback(completed: int, item: Any, total: int) -> None:
        logging.info(f"Converted {completed} / {total} for job {next_job.id}.")
        job_mgr = get_job_manager()
        job_mgr.update_job(
            next_job.id, update={"current_job_progress": completed / total}
        )

    export_zarr_array(
        volume_provider,
        pathlib.Path(CONFIG.chunked_directory) / next_job.id,
        chunk_size=CONFIG.chunk_size,
        progress_callback=_progress_callback,
    )
    logging.info("Finished converting dataset %s", next_job.id)
    next_job.complete_convert()
    job_manager.update_job(next_job.id, next_job)
    logging.info("Updating job %s", next_job.id)


if __name__ == "__main__":
    while True:
        convert_next()
        time.sleep(CONFIG.job_poll_sec)
