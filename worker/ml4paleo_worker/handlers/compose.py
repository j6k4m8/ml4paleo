"""
Final segmentation jobs (see the server's `pipelines/compose.py`).

Grants, in order: the prediction artifact (read), the project's labels
(read; their blobs), and the segmentation artifact (write), which holds the
pinned labels (`inputs.json`) and, while the pipeline runs, each shard's
piece summary under `scratch/`.
"""

import io
import json
from typing import Any

import numpy as np

from ml4paleo.segmentation.compose import (
    apply_shard,
    find_specks,
    label_shard,
    shard_grid,
    shard_inputs,
)
from ml4paleo.segmentation.dataset import BlobLabels
from ml4paleo.segmentation.predict import create_prediction, open_prediction
from ml4paleo.storage import (
    StorageGrant,
    delete_object,
    get_bytes,
    put_bytes,
    write_manifest,
)

from ..context import JobContext, PermanentError


def _inputs(grant: StorageGrant) -> dict[str, Any]:
    raw = get_bytes(grant, "inputs.json")
    if raw is None:
        raise PermanentError("The pinned labels are missing.")
    return json.loads(raw)


def _merged(ctx: JobContext) -> tuple[np.ndarray, np.ndarray]:
    prediction_grant, labels_grant, segmentation_grant = ctx.grants
    inputs = _inputs(segmentation_grant)
    labels = BlobLabels(
        labels_grant, {(cz, cy, cx): sha for cz, cy, cx, sha in inputs["chunks"]}
    )
    box = tuple(int(n) for n in ctx.payload["box"])
    return shard_inputs(
        open_prediction(prediction_grant)["class"],  # type: ignore[arg-type]
        labels,
        box,  # type: ignore[arg-type]
        [tuple(roi) for roi in inputs["complete_rois"]],  # type: ignore[misc]
    )


def prepare(ctx: JobContext) -> dict[str, Any]:
    create_prediction(
        ctx.grants[2], ctx.payload["shape_zyx"], arrays=("class",), kind="segmentation"
    )
    return {}


def block(ctx: JobContext) -> dict[str, Any]:
    merged, labeled = _merged(ctx)
    ctx.check()
    put_bytes(
        ctx.grants[2],
        f"scratch/{ctx.payload['shard']}.npz",
        label_shard(merged, labeled),
    )
    return {}


def merge(ctx: JobContext) -> dict[str, Any]:
    grant = ctx.grants[2]
    summaries = []
    for index in range(int(ctx.payload["shards"])):
        summary = get_bytes(grant, f"scratch/{index}.npz")
        if summary is None:
            raise PermanentError(f"Shard {index}'s piece summary is missing.")
        summaries.append(summary)
    remove = find_specks(
        summaries, shard_grid(ctx.payload["shape_zyx"]), int(ctx.payload["min_voxels"])
    )
    for index, ids in enumerate(remove):
        buffer = io.BytesIO()
        np.save(buffer, ids)
        put_bytes(grant, f"scratch/{index}.remove.npy", buffer.getvalue())
    return {"specks": int(sum(len(ids) for ids in remove))}


def apply(ctx: JobContext) -> dict[str, Any]:
    grant = ctx.grants[2]
    raw = get_bytes(grant, f"scratch/{ctx.payload['shard']}.remove.npy")
    if raw is None:
        raise PermanentError("The list of specks to remove is missing.")
    merged, _ = _merged(ctx)
    final = apply_shard(merged, np.load(io.BytesIO(raw)))
    box = tuple(int(n) for n in ctx.payload["box"])
    region = tuple(slice(box[a], box[a + 3]) for a in range(3))
    open_prediction(grant)["class"][region] = final  # type: ignore[index]
    return {}


def finalize(ctx: JobContext) -> dict[str, Any]:
    grant = ctx.grants[2]
    for index in range(int(ctx.payload["shards"])):
        delete_object(grant, f"scratch/{index}.npz")
        delete_object(grant, f"scratch/{index}.remove.npy")
    write_manifest(
        grant,
        {
            "kind": "segmentation",
            "shape_zyx": ctx.payload["shape_zyx"],
            "min_voxels": ctx.payload["min_voxels"],
            "model_id": ctx.payload["model_id"],
            "prediction_artifact_id": ctx.payload["prediction_artifact_id"],
            "label_seq": ctx.payload["label_seq"],
        },
    )
    return {}
