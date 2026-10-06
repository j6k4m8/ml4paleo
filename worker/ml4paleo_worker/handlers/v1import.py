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
import math
import uuid
from collections.abc import Iterator, Sequence
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
    JOB_ID,
    SEGMENTATION_NAME,
    UNCONVERTED,
    annotations,
    image_path,
    read_jobs,
    segmentation,
    status,
)
from ml4paleo.volume_providers.volume_provider import VolumeProvider
from ml4paleo.volume_providers.zarrvp import ZarrVolumeProvider

from ..context import JobContext, PermanentError

# A slab job reads at least this much of the image at once.
MIN_READ_BYTES = 16 * 1024**2
# The namespace of the label edits' ids, one per v1 job and sample.
SAMPLE_OPS = uuid.UUID("2bf25a04-4522-4724-907e-b2b2dd0a9691")

Box = list[tuple[int, int]]


def _pieces(box: Box, chunks: Sequence[int], most: int) -> Iterator[Box]:
    """
    Split a box ((start, stop) per axis) at chunk boundaries until each piece
    touches at most `most` chunks, halving its most-chunked side first.
    """
    counts = [
        (stop - 1) // size - start // size + 1
        for (start, stop), size in zip(box, chunks, strict=True)
    ]
    if math.prod(counts) <= most or max(counts) == 1:
        yield box
        return
    axis = counts.index(max(counts))
    start, stop = box[axis]
    middle = (start // chunks[axis] + counts[axis] // 2) * chunks[axis]
    for side in ((start, middle), (middle, stop)):
        yield from _pieces([*box[:axis], side, *box[axis + 1 :]], chunks, most)


def _chunks_at_once(ctx: JobContext, array: zarr.Array) -> int:
    """
    How many of a v1 array's chunks one read may touch. Reading decodes each
    whole (zarr decodes several at once) and copies it out, which takes about
    four times a chunk each; that gets a third of the job's memory.
    """
    chunk = math.prod(array.chunks) * array.dtype.itemsize
    return max(1, ctx.memory_budget_bytes // 3 // (4 * chunk))


class _FewChunks(VolumeProvider):
    """A v1 image, read at most a few of its chunks at a time."""

    def __init__(self, array: zarr.Array, most: int):
        self.array = array
        self.most = most

    @property
    def shape(self) -> tuple[int, int, int]:
        x, y, z = self.array.shape
        return x, y, z

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.array.dtype)

    def __getitem__(self, key) -> np.ndarray:
        box = [s.indices(n)[:2] for s, n in zip(key, self.shape, strict=True)]
        out = np.empty([stop - start for start, stop in box], dtype=self.dtype)
        for piece in _pieces(box, self.array.chunks, self.most):
            into = tuple(
                slice(a - start, b - start)
                for (a, b), (start, _) in zip(piece, box, strict=True)
            )
            out[into] = self.array[tuple(slice(a, b) for a, b in piece)]
        return out


def _root(ctx: JobContext) -> Path:
    if ctx.v1_volume is None:
        # The worker is set up wrong, not the job: another try may get a
        # worker with the volume.
        raise RuntimeError("This worker has no v1 volume (start it with --v1-volume).")
    return ctx.v1_volume


def _job_id(ctx: JobContext) -> str:
    """The job's v1 job id, which names folders in the volume."""
    job_id = ctx.payload["job_id"]
    if not (isinstance(job_id, str) and JOB_ID.fullmatch(job_id)):
        raise PermanentError(f"{job_id!r} isn't a v1 job id.")
    return job_id


def _record(root: Path, job_id: str) -> dict[str, Any]:
    record = read_jobs(root).get(job_id)
    if record is None:
        raise PermanentError(f"There's no v1 job {job_id}.")
    return record


def probe(ctx: JobContext) -> dict[str, Any]:
    root = _root(ctx)
    job_id = _job_id(ctx)
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
        # v1 recorded voxel sizes (in mm) only for DICOM scans, from #76 on.
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

    source = ZarrVolumeProvider(image_path(root, _job_id(ctx))).zarr
    most = _chunks_at_once(ctx, source)
    chunk = math.prod(source.chunks) * source.dtype.itemsize
    # The rest of the job's memory goes to what's read, which is held about
    # three times over while it's written.
    read = max(MIN_READ_BYTES, (ctx.memory_budget_bytes - 4 * most * chunk) // 3)
    write_from_provider(
        _FewChunks(source, most),
        OmeImage.open(ctx.grants[0]),
        z_range=(z0, z1),
        max_read_bytes=read,
        progress=progress,
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
    job_id = _job_id(ctx)
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
                    # Named by the v1 job and sample, so each sample lands once
                    # however often the labels are brought over.
                    "client_op_id": str(
                        uuid.uuid5(SAMPLE_OPS, f"{job_id}/{annotation.stamp}")
                    ),
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
    v1 segmentations hold 0 for background and anything else (usually 255)
    where the model found the foreground; the prediction holds background and
    the foreground class.
    """
    root = _root(ctx)
    job_id = _job_id(ctx)
    name = ctx.payload["segmentation"]
    if not (isinstance(name, str) and SEGMENTATION_NAME.fullmatch(name)):
        raise PermanentError(f"{name!r} isn't a v1 segmentation's name.")
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
    most = _chunks_at_once(ctx, source)
    chunk_x, chunk_y, chunk_z = source.chunks
    # Shard by shard, so each is written once. Each read is a source chunk
    # deep, so each source chunk is decoded once, and a few chunks wide.
    for done, (z0, y0, x0) in enumerate(boxes, start=1):
        z1, y1, x1 = (
            min(o + s, n) for o, s, n in zip((z0, y0, x0), shard, shape, strict=True)
        )
        out = np.empty((z1 - z0, y1 - y0, x1 - x0), dtype=np.uint8)
        for [(za, zb)] in _pieces([(z0, z1)], [chunk_z], 1):
            for (xa, xb), (ya, yb) in _pieces(
                [(x0, x1), (y0, y1)], [chunk_x, chunk_y], most
            ):
                found = np.asarray(source[xa:xb, ya:yb, za:zb]) > 0
                part = out[za - z0 : zb - z0, ya - y0 : yb - y0, xa - x0 : xb - x0]
                part[...] = BACKGROUND
                part[found.transpose(2, 1, 0)] = foreground
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
