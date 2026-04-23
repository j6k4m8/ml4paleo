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
import datetime
import pathlib
import time
from typing import Any, Optional, Tuple
from PIL import Image

import numpy as np
from config import CONFIG
from job import JobStatus, JSONFileUploadJobManager, UploadJob
from apputils import get_annotation_pairs, load_annotation_sample_metadata, load_annotation_source_slice

from ml4paleo.segmentation import (
    RandomForest3DSegmenter,
    Segmenter3D,
    segment_volume_to_zarr,
)
from ml4paleo.volume_providers import ZarrVolumeProvider

logging.basicConfig(level=logging.INFO)


def _first_channel(img: Image.Image) -> np.ndarray:
    """
    Return the first image channel, or the grayscale image as-is.
    """
    arr = np.array(img)
    return arr[:, :, 0] if arr.ndim == 3 else arr


def model_factory() -> Tuple[Segmenter3D, dict]:
    """
    Return a new instance of the model.
    """
    rf_kwargs = {
        "n_estimators": 50,
        "max_depth": 12,
        "n_jobs": -1,
    }
    return (
        RandomForest3DSegmenter(rf_kwargs=dict(rf_kwargs)),
        {
            "rf_kwargs": dict(rf_kwargs),
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


def _sample_id_from_path(img_path: pathlib.Path) -> str:
    """
    Return the logical training-sample identifier from an image filename.
    """
    stem = img_path.stem
    prefix = CONFIG.training_img_prefix
    return stem.removeprefix(prefix) if stem.startswith(prefix) else stem


def _foreground_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """
    Return foreground-over-background metrics for one prediction array.
    """
    pred_fg = pred != 0
    truth_fg = truth != 0

    tp = int(np.logical_and(pred_fg, truth_fg).sum())
    fp = int(np.logical_and(pred_fg, ~truth_fg).sum())
    fn = int(np.logical_and(~pred_fg, truth_fg).sum())
    correct = int((pred == truth).sum())
    total = int(truth.size)

    foreground_denom = (2 * tp) + fp + fn
    iou_denom = tp + fp + fn
    dice = (2 * tp / foreground_denom) if foreground_denom else 1.0
    iou = (tp / iou_denom) if iou_denom else 1.0
    accuracy = (correct / total) if total else 1.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "correct": correct,
        "total": total,
        "train_foreground_dice": float(dice),
        "train_foreground_iou": float(iou),
        "train_pixel_accuracy": float(accuracy),
        "train_loss": float(1.0 - dice),
    }


def _evaluate_training_metrics(
    segmenter: Segmenter3D,
    imgs_xy: list[np.ndarray],
    segs_xy: list[np.ndarray],
) -> dict[str, float]:
    """
    Evaluate the trained segmenter on the training slices.
    """
    aggregate = {"tp": 0, "fp": 0, "fn": 0, "correct": 0, "total": 0}
    for img_xy, seg_xy in zip(imgs_xy, segs_xy):
        pred_xy = segmenter._segment_slice(img_xy)  # type: ignore[attr-defined]
        metrics = _foreground_metrics(pred_xy, seg_xy)
        for key in aggregate:
            aggregate[key] += int(metrics[key])

    tp = aggregate["tp"]
    fp = aggregate["fp"]
    fn = aggregate["fn"]
    correct = aggregate["correct"]
    total = aggregate["total"]
    foreground_denom = (2 * tp) + fp + fn
    iou_denom = tp + fp + fn
    dice = (2 * tp / foreground_denom) if foreground_denom else 1.0
    iou = (tp / iou_denom) if iou_denom else 1.0
    accuracy = (correct / total) if total else 1.0

    return {
        "train_foreground_dice": float(dice),
        "train_foreground_iou": float(iou),
        "train_pixel_accuracy": float(accuracy),
        "train_loss": float(1.0 - dice),
        "foreground_tp": int(tp),
        "foreground_fp": int(fp),
        "foreground_fn": int(fn),
        "pixel_correct": int(correct),
        "pixel_total": int(total),
    }


def _write_model_metadata(
    job: UploadJob,
    model_id: str,
    metadata: dict,
) -> pathlib.Path:
    """
    Persist the JSON metadata for one trained model.
    """
    model_params_path = pathlib.Path(CONFIG.model_directory) / str(job.id) / f"{model_id}.json"
    model_params_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_params_path, "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    return model_params_path


def _update_model_metadata(
    job_id: str,
    model_id: str,
    update: dict,
) -> None:
    """
    Merge an update into one model's metadata JSON if it exists.
    """
    model_params_path = pathlib.Path(CONFIG.model_directory) / str(job_id) / f"{model_id}.json"
    if not model_params_path.exists():
        return

    with open(model_params_path, "r") as f:
        existing = json.load(f)
    if not isinstance(existing, dict):
        existing = {}

    existing.update(update)
    with open(model_params_path, "w") as f:
        json.dump(existing, f, indent=2, sort_keys=True)


def train_job(job: UploadJob) -> Tuple[Segmenter3D, str]:
    # First train the segmenter on the available training data.
    # The training data live in the CONFIG.training_directory directory, with
    # CONFIG.training_img_prefix and CONFIG.training_seg_prefix.
    # Both image and segmentation are PNG files.
    # TODO: For now, the segmentation is SINGLE channel, and is stored in the
    # red channel of the PNG file.
    imgs_np = []
    segs_np = []
    training_samples = []
    metadata_backed_samples = 0
    for img_path, seg_path, meta_path in get_annotation_pairs(job):
        sample_id = _sample_id_from_path(img_path)
        sample_metadata = load_annotation_sample_metadata(meta_path)
        if sample_metadata is not None:
            img_xy = load_annotation_source_slice(job.id, sample_metadata)
            metadata_backed_samples += 1
        else:
            img_xy = _first_channel(Image.open(img_path)).T
        seg_xy = _first_channel(Image.open(seg_path)).T
        imgs_np.append(img_xy)
        segs_np.append(seg_xy)
        training_samples.append(
            {
                "sample_id": sample_id,
                "image_filename": img_path.name,
                "mask_filename": seg_path.name,
                "metadata_filename": meta_path.name if meta_path is not None else None,
                "uses_raw_volume_metadata": sample_metadata is not None,
            }
        )
    training_count = len(imgs_np)
    if training_count == 0:
        raise ValueError(f"No training image/mask pairs found for job {job.id}")
    logging.info(f"Loaded {training_count} training images.")
    logging.info(
        "Using raw-volume cutout metadata for %s / %s training samples.",
        metadata_backed_samples,
        training_count,
    )
    imgs_np = np.stack(imgs_np, axis=-1)
    segs_np = np.stack(segs_np, axis=-1)

    # Train the model:
    segmenter, model_params = model_factory()

    logging.info("Training with shapes img=%s and seg=%s", imgs_np.shape, segs_np.shape)
    segmenter.fit(imgs_np, segs_np)
    training_metrics = _evaluate_training_metrics(
        segmenter,
        [imgs_np[:, :, z] for z in range(imgs_np.shape[2])],
        [segs_np[:, :, z] for z in range(segs_np.shape[2])],
    )
    logging.info(
        "Training metrics for job %s: dice=%.4f iou=%.4f loss=%.4f",
        job.id,
        training_metrics["train_foreground_dice"],
        training_metrics["train_foreground_iou"],
        training_metrics["train_loss"],
    )

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
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    model_params.update(
        {
            "model_id": timestamp,
            "job_id": job.id,
            "created_at": created_at,
            "annotation_count": training_count,
            "metadata_backed_sample_count": metadata_backed_samples,
            "training_samples": training_samples,
            "metrics": training_metrics,
            "artifacts": {
                "training_curve": {
                    "status": "ready",
                    "generated_at": created_at,
                    "metric_name": "train_foreground_dice",
                }
            },
        }
    )
    _write_model_metadata(job, timestamp, model_params)
    return segmenter, timestamp


def train_and_segment_job(job: UploadJob) -> None:
    segmenter, timestamp = train_job(job)

    vol_provider = ZarrVolumeProvider(
        str(pathlib.Path(CONFIG.chunked_directory) / str(job.id))
    )
    seg_path = (
        pathlib.Path(CONFIG.segmented_directory) / str(job.id) / (timestamp + ".zarr")
    )

    def progress_callback(completed: int, item: Any, total: int) -> None:
        logging.info(
            f"Segmented {completed} / {total} chunks for job {job.id} (chunk {item})."
        )
        job_mgr = get_job_manager()
        job_mgr.update_job(job.id, update={"current_job_progress": completed / total})

    segment_volume_to_zarr(
        vol_provider,
        seg_path,
        segmenter=segmenter,
        chunk_size=CONFIG.segmentation_chunk_size,
        parallel=CONFIG.segment_job_parallelism,
        progress=True,
        progress_callback=progress_callback,
    )
    _update_model_metadata(job.id, timestamp, {"segmentation_id": seg_path.name})


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
    try:
        job_mgr.update_job(job.id, update={"status": JobStatus.SEGMENTED})
    except Exception as e:
        time.sleep(1)  # potential race condition with 100% update
        job_mgr.update_job(job.id, update={"status": JobStatus.SEGMENTED})


if __name__ == "__main__":
    while True:
        main()
        time.sleep(CONFIG.job_poll_sec)
