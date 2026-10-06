"""
v1 import jobs (see the server's `pipelines/v1import.py`). They run on
workers started with `--v1-volume`, which can read the v1 app's volume
folder (`ml4paleo.v1import`) and run no other jobs.

    v1.probe        check the job, create the empty image, and count what to
                    bring over
    v1.slab         copy a shard-deep slab of the image (x, y, z to z, y, x)
    v1.labels       each placed annotation sample becomes one label edit
                    (foreground and background) and a complete slice ROI
    v1.prediction   the newest finished segmentation becomes the prediction

Grants: the image artifact (read-write) for the probe and slabs, the
prediction artifact (read-write) for the prediction, none for labels (they go
through the server's label writer).
"""

import base64
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import zarr

from ml4paleo.labels import BACKGROUND
from ml4paleo.labels.deltas import ChunkDelta, split_into_deltas
from ml4paleo.ome import OmeImage, write_from_provider
from ml4paleo.segmentation.predict import create_prediction
from ml4paleo.storage import write_manifest
from ml4paleo.v1import import (
    UNCONVERTED,
    annotations,
    image_path,
    read_jobs,
    segmentation,
    status,
)
from ml4paleo.volume_providers.zarrvp import ZarrVolumeProvider

from ..context import JobContext, PermanentError

# v1 segmentations are this deep per read (their chunks are 256³ or 64³).
PREDICTION_READ_DEPTH = 64


def _root(ctx: JobContext) -> Path:
    if ctx.v1_volume is None:
        # The worker is set up wrong, not the job: another try may get a
        # worker with the volume.
        raise RuntimeError("This worker has no v1 volume (start it with --v1-volume).")
    return ctx.v1_volume


def _record(root: Path, job_id: str) -> dict[str, Any]:
    record = read_jobs(root).get(job_id)
    if record is None:
        raise PermanentError(f"There's no v1 job {job_id}.")
    return record


def probe(ctx: JobContext) -> dict[str, Any]:
    root = _root(ctx)
    job_id = ctx.payload["job_id"]
    record = _record(root, job_id)
    if status(record) in UNCONVERTED:
        # Its array, if any, is partial.
        raise PermanentError("This v1 job never finished converting its upload.")
    path = image_path(root, job_id)
    if not (path / ".zarray").is_file():
        raise PermanentError("This v1 job has no converted image to import.")
    provider = ZarrVolumeProvider(path)
    x, y, z = (int(n) for n in provider.shape)
    voxel = provider.voxel_size_xyz_mm
    ctx.check()
    image = OmeImage.create(
        ctx.grants[0],
        shape_czyx=(1, z, y, x),
        dtype=np.dtype(provider.dtype),
        # v1 recorded voxel sizes (in mm) for DICOM scans only.
        voxel_size_zyx=tuple(reversed(voxel)) if voxel else None,
        unit="millimeter" if voxel else None,
        name=str(record.get("name") or job_id),
        overwrite=True,
    )
    shards = image.array(0).shards
    assert shards is not None
    placed, skipped = annotations(root, job_id, (x, y, z))
    found = segmentation(root, job_id, record)
    if found is not None:
        shape = zarr.open_array(
            str(root / "segmented" / job_id / found), mode="r"
        ).shape
        if tuple(shape) != (x, y, z):
            found = None
    return {
        "kind": "v1",
        "shape_zyx": [z, y, x],
        "dtype": np.dtype(provider.dtype).str,
        "levels": image.num_levels,
        "slabs": [
            [start, min(start + shards[1], z)] for start in range(0, z, shards[1])
        ],
        "status": status(record),
        "annotations": len(placed),
        "skipped_annotations": skipped,
        "segmentation": found,
    }


def slab(ctx: JobContext) -> dict[str, Any]:
    root = _root(ctx)
    z0, z1 = ctx.payload["z_range"]

    def progress(done: int, total: int) -> None:
        ctx.progress(done / total)
        ctx.check()

    provider = ZarrVolumeProvider(image_path(root, ctx.payload["job_id"]))
    write_from_provider(
        provider, OmeImage.open(ctx.grants[0]), z_range=(z0, z1), progress=progress
    )
    return {"z_range": [z0, z1]}


def _wire(delta: ChunkDelta) -> dict[str, Any]:
    """A delta as the labels API takes it."""
    return {
        "key": list(delta.key),
        "box": list(delta.box),
        "mask": base64.b64encode(delta.mask).decode(),
        "values": base64.b64encode(delta.values or b"").decode(),
    }


def labels(ctx: JobContext) -> dict[str, Any]:
    root = _root(ctx)
    job_id = ctx.payload["job_id"]
    foreground = int(ctx.payload["foreground"])
    z, y, x = ctx.payload["shape_zyx"]
    placed, _ = annotations(root, job_id, (x, y, z))
    rois = []
    for done, annotation in enumerate(placed, start=1):
        try:
            mask = annotation.foreground()
        except (OSError, ValueError):
            # An unreadable sample is left out, as v1 would have failed on it.
            continue
        values = np.where(mask, foreground, BACKGROUND).astype(np.uint8)[np.newaxis]
        box = annotation.box_zyx
        deltas = split_into_deltas(
            np.ones(values.shape, dtype=bool), (box[0], box[1], box[2]), values=values
        )
        try:
            ctx.apply_label_op(
                {
                    # The same id on a retry, so each sample lands once.
                    "client_op_id": str(uuid.uuid5(ctx.job_id, annotation.stamp)),
                    "deltas": [_wire(delta) for delta in deltas],
                    "tool": {
                        "name": "v1-import",
                        "job": job_id,
                        "sample": annotation.stamp,
                    },
                }
            )
        except ValueError as exc:
            raise PermanentError(
                f"The server refused sample {annotation.stamp}: {exc}"
            ) from exc
        rois.append(box)
        ctx.progress(done / len(placed))
    return {"rois": rois}


def prediction(ctx: JobContext) -> dict[str, Any]:
    """
    v1 segmentations hold 255 where the model found the foreground and 0
    elsewhere; the prediction holds the foreground class and background.
    """
    root = _root(ctx)
    job_id = ctx.payload["job_id"]
    name = ctx.payload["segmentation"]
    foreground = int(ctx.payload["foreground"])
    source = zarr.open_array(str(root / "segmented" / job_id / name), mode="r")
    shape = [int(n) for n in ctx.payload["shape_zyx"]]
    group = create_prediction(ctx.grants[0], shape)
    classes = group["class"]
    assert isinstance(classes, zarr.Array) and classes.shards is not None
    shard = classes.shards
    boxes = [
        (z0, y0, x0)
        for z0 in range(0, shape[0], shard[0])
        for y0 in range(0, shape[1], shard[1])
        for x0 in range(0, shape[2], shard[2])
    ]
    # Shard by shard, so each is written once, read a few slices at a time.
    for done, (z0, y0, x0) in enumerate(boxes, start=1):
        z1, y1, x1 = (
            min(o + s, n) for o, s, n in zip((z0, y0, x0), shard, shape, strict=True)
        )
        out = np.empty((z1 - z0, y1 - y0, x1 - x0), dtype=np.uint8)
        for zz in range(z0, z1, PREDICTION_READ_DEPTH):
            top = min(zz + PREDICTION_READ_DEPTH, z1)
            block = np.asarray(source[x0:x1, y0:y1, zz:top]).transpose(2, 1, 0)
            out[zz - z0 : top - z0] = np.where(block > 0, foreground, BACKGROUND)
            ctx.check()
        classes[z0:z1, y0:y1, x0:x1] = out
        ctx.progress(done / len(boxes))
    write_manifest(
        ctx.grants[0],
        {
            "kind": "prediction",
            "shape_zyx": shape,
            "class_values": [foreground],
            "v1_job_id": job_id,
            "v1_segmentation": name,
        },
    )
    return {"shards": len(boxes)}
