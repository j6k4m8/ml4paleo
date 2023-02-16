import logging
import pathlib
import time
from typing import Optional

from job import JobStatus, JSONFileUploadJobManager, UploadJob

from ml4paleo.meshing import ChunkedMesher
from ml4paleo.volume_providers import ZarrVolumeProvider

from config import CONFIG
from apputils import get_latest_segmentation_id

logging.basicConfig(level=logging.INFO)


def get_job_manager() -> JSONFileUploadJobManager:
    """
    Return the job manager.
    """
    return JSONFileUploadJobManager("volume/jobs.json")


def get_next_dataset_to_mesh() -> Optional[UploadJob]:
    """
    Return the next dataset that has been segmented but not yet meshed.
    """
    job_manager = get_job_manager()
    next_job = job_manager.get_jobs_by_status(JobStatus.MESHING_QUEUED)
    if len(next_job) == 0:
        return None
    return next_job[0]


def mesh_job(job: UploadJob) -> None:
    """
    Mesh the job.
    """
    # Get the latest segmentation ID:
    latest_seg = get_latest_segmentation_id(job)
    if latest_seg is None:
        raise ValueError(f"No segmentation found for job {job.id}")
    logging.info(f"Meshing job {job.id} with segmentation {latest_seg}...")
    # Get the volume provider.
    volume_provider = ZarrVolumeProvider(
        pathlib.Path(CONFIG.segmented_directory) / job.id / latest_seg
    )
    # Get the mesher:
    mesher = ChunkedMesher(
        volume_provider,
        pathlib.Path(CONFIG.meshed_directory) / job.id / latest_seg,
        chunk_size=CONFIG.meshing_chunk_size,
    )
    # Mesh everything:
    mesher.mesh_all()


def main():
    job_mgr = get_job_manager()
    logging.info("Looking for a job to mesh...")
    job = get_next_dataset_to_mesh()
    if job is None:
        logging.info("No job to mesh.")
        return

    logging.info(f"Found job {job.id} to mesh.")
    job_mgr.update_job(job.id, update={"status": JobStatus.MESHING})

    logging.info(f"Meshing job {job.id}...")
    try:
        mesh_job(job)
        logging.info(f"Meshing for job {job.id} is complete!")
    except Exception as e:
        logging.error(f"Meshing for job {job.id} failed with error: {e}")
        # Print the rest of the stack trace:
        logging.error(e, exc_info=True)

        job_mgr.update_job(job.id, update={"status": JobStatus.MESH_ERROR})
        return
    # Update the job status:
    job_mgr.update_job(job.id, update={"status": JobStatus.MESHED})


if __name__ == "__main__":
    while True:
        main()
        time.sleep(CONFIG.job_poll_sec)
