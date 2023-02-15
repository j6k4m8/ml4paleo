import json
import logging
import pathlib
import time
from typing import Optional, Tuple
from PIL import Image

import numpy as np
import tqdm
import zarr
from config import CONFIG
from intern.utils.parallel import block_compute
from job import JobStatus, JSONFileUploadJobManager, UploadJob
from joblib import Parallel, delayed

from ml4paleo.segmentation import RandomForest3DSegmenter, Segmenter3D
from ml4paleo.volume_providers import VolumeProvider, ZarrVolumeProvider

logging.basicConfig(level=logging.INFO)


def model_factory() -> Tuple[Segmenter3D, dict]:
    """
    Return a new instance of the model.
    """
    rf_kwargs = {
        "n_estimators": 10,
        "max_depth": 8,
        "n_jobs": -1,
    }
    return (
        RandomForest3DSegmenter(rf_kwargs=rf_kwargs, features_fn=lambda x: x),
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


def segment_chunk_and_write(
    xs: Tuple[int, int],
    ys: Tuple[int, int],
    zs: Tuple[int, int],
    volume_provider: VolumeProvider,
    segmenter: Segmenter3D,
    seg_path: str,
) -> bool:
    """
    Segment a chunk of a job.
    """
    # Get the volume for the chunk:
    volume = volume_provider[xs[0] : xs[1], ys[0] : ys[1], zs[0] : zs[1]]
    # Segment the volume:
    seg_volume = np.zeros(volume.shape, dtype=np.uint64)
    for z in range(volume.shape[0]):
        seg_volume[z] = segmenter.segment(np.expand_dims(volume[z], axis=-1))
    # Write the seg to the seg path zarr:
    seg_zarr = zarr.open(seg_path, mode="a")
    seg_zarr[xs[0] : xs[1], ys[0] : ys[1], zs[0] : zs[1]] = seg_volume
    return True


def train_and_segment_job(job: UploadJob) -> None:
    # First train the segmenter on the available training data.
    # The training data live in the CONFIG.training_directory directory, with
    # CONFIG.training_img_prefix and CONFIG.training_seg_prefix.
    # Both image and segmentation are PNG files.
    # TODO: For now, the segmentation is SINGLE channel, and is stored in the
    # red channel of the PNG file.
    imgs_np = []
    segs_np = []
    for img_path, seg_path in zip(
        (pathlib.Path(CONFIG.training_directory) / job.id).glob(
            f"{CONFIG.training_img_prefix}*.png"
        ),
        (pathlib.Path(CONFIG.training_directory) / job.id).glob(
            f"{CONFIG.training_seg_prefix}*.png"
        ),
    ):
        imgs_np.append(np.array(Image.open(img_path))[:, :, 0])
        segs_np.append(np.array(Image.open(seg_path))[:, :, 0])
    logging.info(f"Loaded {len(imgs_np)} training images.")
    imgs_np = np.stack(imgs_np)
    segs_np = np.stack(segs_np)

    # Train the model:
    segmenter, model_params = model_factory()
    segs_np = np.expand_dims(segs_np, axis=-1)
    imgs_np = np.expand_dims(imgs_np, axis=-1)
    logging.info(
        "Training with shapes (XYZF) img=%s and seg=%s", imgs_np.shape, segs_np.shape
    )
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

    vol_provider = ZarrVolumeProvider(
        str(pathlib.Path(CONFIG.chunked_directory) / str(job.id))
    )
    seg_path = (
        pathlib.Path(CONFIG.segmented_directory) / str(job.id) / (timestamp + ".zarr")
    )
    seg_path.mkdir(parents=True, exist_ok=True)

    # Create the Zarr file for the segmentation.
    zarr.open(
        str(seg_path),
        mode="w",
        dtype="uint64",
        shape=vol_provider.shape,
        chunks=CONFIG.segmentation_chunk_size,
        write_empty_chunks=False,
    )

    # Now segment the job.
    # We segment the job in chunks, and save the results in the
    # CONFIG.segmentation_directory as another Zarr file.
    chunks_to_segment = block_compute(
        0,
        vol_provider.shape[0],
        0,
        vol_provider.shape[1],
        0,
        vol_provider.shape[2],
        block_size=CONFIG.segmentation_chunk_size,
    )

    # for xs, ys, zs in chunks_to_segment
    _ = Parallel(n_jobs=CONFIG.segment_job_parallelism)(
        delayed(segment_chunk_and_write)(xs, ys, zs, vol_provider, segmenter, seg_path)
        for xs, ys, zs in tqdm.tqdm(chunks_to_segment)
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
        time.sleep(1)
