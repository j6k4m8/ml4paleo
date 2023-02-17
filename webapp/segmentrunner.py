"""
This script is responsible for segmenting the datasets once the user has
indicated that they are ready. The user indicates that the dataset should be
segmented by setting the job status to `TRAINING_QUEUED`. This script will then
pick up the job and segment it, setting the status temporarily to `SEGMENTING`,
and then to `SEGMENTED` when the segmentation is complete. If the segmentation
fails, the status will be set to `SEGMENT_ERROR`.

The `model_factory` function is responsible for returning a new instance of the
model that will be used to segment the dataset. This function is called each
time a new dataset is to be segmented, so that the model is re-initialized
before each segmentation. If you want to modify the type or parameters of the
model that is used to perform segmentation, you should modify this function.

Note that this could also be adapted in the future to allow the user to select
the model that they want to use for segmentation, or to adapt the parameters of
the model based on the dataset that is being segmented.
"""

import json
import logging
import pathlib
import time
from typing import Optional, Tuple
from PIL import Image

import numpy as np
from config import CONFIG
from job import JobStatus, JSONFileUploadJobManager, UploadJob

from ml4paleo.segmentation import (
    RandomForest3DSegmenter,
    Segmenter3D,
    segment_volume_to_zarr,
)
from ml4paleo.volume_providers import ZarrVolumeProvider

logging.basicConfig(level=logging.INFO)


def model_factory() -> Tuple[Segmenter3D, dict]:
    """
    Return a new instance of the model.
    """
    rf_kwargs = {
        "n_estimators": 8,
        "max_depth": 8,
        "n_jobs": -1,
    }
    return (
        RandomForest3DSegmenter(rf_kwargs=rf_kwargs),
        {
            "rf_kwargs": rf_kwargs,
            "model_class": "RandomForest3DSegmenter",
        },
    )


def get_job_manager() -> JSONFileUploadJobManager:
    """
    Return the job manager.
    """
    return JSONFileUploadJobManager("volume/jobs.json")


def get_next_dataset_to_segment() -> Optional[UploadJob]:
    """
    Return the next dataset that has been converted but not yet segmented.
    """
    job_manager = get_job_manager()
    next_job = job_manager.get_jobs_by_status(JobStatus.TRAINING_QUEUED)
    if len(next_job) == 0:
        return None
    return next_job[0]


def train_job(job: UploadJob) -> Tuple[Segmenter3D, str]:
    # First train the segmenter on the available training data.
    # The training data live in the CONFIG.training_directory directory, with
    # CONFIG.training_img_prefix and CONFIG.training_seg_prefix.
    # Both image and segmentation are PNG files.
    # TODO: For now, the segmentation is SINGLE channel, and is stored in the
    # red channel of the PNG file.
    imgs_np = []
    segs_np = []
    for img_path, seg_path in zip(
        sorted(
            (pathlib.Path(CONFIG.training_directory) / job.id).glob(
                f"{CONFIG.training_img_prefix}*.png"
            )
        ),
        sorted(
            (pathlib.Path(CONFIG.training_directory) / job.id).glob(
                f"{CONFIG.training_seg_prefix}*.png"
            )
        ),
    ):
        imgs_np.append(np.array(Image.open(img_path))[:, :, 0])
        segs_np.append(np.array(Image.open(seg_path))[:, :, 0])
    training_count = len(imgs_np)
    logging.info(f"Loaded {training_count} training images.")
    imgs_np = np.stack(imgs_np)
    segs_np = np.stack(segs_np)

    # Train the model:
    segmenter, model_params = model_factory()

    # If there's only one training image, we need to add a dimension to the
    # array, so that the model can be trained.
    if training_count == 1:
        imgs_np = np.expand_dims(imgs_np, axis=-1)
        segs_np = np.expand_dims(segs_np, axis=-1)
    logging.info("Training with shapes img=%s and seg=%s", imgs_np.shape, segs_np.shape)
    segmenter.fit(imgs_np, segs_np)

    # Save the model.
    # Models are saved in the CONFIG.model_directory directory, with the job ID
    # as the subdirectory. We save two items: the model itself, and the model
    # parameters, in a JSON file. The two are named with the current timestamp.
    timestamp = str(int(time.time()))
    logging.info(f"Creating model {timestamp} for job {job.id}.")
    model_path = (
        pathlib.Path(CONFIG.model_directory) / str(job.id) / f"{timestamp}.model"
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    segmenter.save(str(model_path))
    # Save the model parameters:
    model_params_path = (
        pathlib.Path(CONFIG.model_directory) / str(job.id) / f"{timestamp}.json"
    )
    with open(model_params_path, "w") as f:
        json.dump(model_params, f)
    return segmenter, timestamp


def train_and_segment_job(job: UploadJob) -> None:
    segmenter, timestamp = train_job(job)

    vol_provider = ZarrVolumeProvider(
        str(pathlib.Path(CONFIG.chunked_directory) / str(job.id))
    )
    seg_path = (
        pathlib.Path(CONFIG.segmented_directory) / str(job.id) / (timestamp + ".zarr")
    )

    segment_volume_to_zarr(
        vol_provider,
        seg_path,
        segmenter=segmenter,
        chunk_size=CONFIG.segmentation_chunk_size,
        parallel=CONFIG.segment_job_parallelism,
        progress=True,
    )


def main():
    # First, we need to find a job to segment.
    job_mgr = get_job_manager()
    logging.info("Looking for a job to segment...")
    job = get_next_dataset_to_segment()
    if job is None:
        logging.info("No job to segment.")
        return

    logging.info(f"Found job {job.id} to segment.")
    job_mgr.update_job(job.id, update={"status": JobStatus.SEGMENTING})

    # Segment the job:
    logging.info(f"Segmenting job {job.id}...")
    try:
        train_and_segment_job(job)
        logging.info(f"Segmentation for job {job.id} is complete!")
    except Exception as e:
        logging.error(f"Segmentation for job {job.id} failed with error: {e}")
        job_mgr.update_job(job.id, update={"status": JobStatus.SEGMENT_ERROR})
        return
    # Update the job status:
    job_mgr.update_job(job.id, update={"status": JobStatus.SEGMENTED})


if __name__ == "__main__":
    while True:
        main()
        time.sleep(CONFIG.job_poll_sec)
