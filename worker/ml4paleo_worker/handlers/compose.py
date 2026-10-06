"""
Final segmentation jobs (see the server's `pipelines/compose.py`).

Grants, in order: the prediction artifact (read), the project's labels
(read; their blobs), and the segmentation artifact (write), which holds the
pinned labels (`inputs.json`) and, while the pipeline runs, each shard's
pieces under `scratch/`: `<n>.npz`, its seam pieces for `cc.merge`;
`<n>.local.npz`, which of its pieces `cc.block` found are specks; and
`<n>.joined.npz`, which of its seam pieces `cc.merge` found are.
"""

import io
import json
from typing import Any

import numpy as np

from ml4paleo.segmentation.compose import (
    TooLarge,
    apply_shard,
    find_specks,
    label_shard,
    npz,
    seams,
    shard_grid,
    shard_inputs,
    shard_specks,
    slab_depth,
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

SCRATCH = ("npz", "local.npz", "joined.npz")


def _inputs(grant: StorageGrant) -> dict[str, Any]:
    raw = get_bytes(grant, "inputs.json")
    if raw is None:
        raise PermanentError("The pinned labels are missing.")
    return json.loads(raw)


def _box(ctx: JobContext) -> tuple[int, int, int, int, int, int]:
    return tuple(int(n) for n in ctx.payload["box"])  # type: ignore[return-value]


def _merged(ctx: JobContext) -> tuple[np.ndarray, np.ndarray]:
    prediction_grant, labels_grant, segmentation_grant = ctx.grants
    inputs = _inputs(segmentation_grant)
    labels = BlobLabels(
        labels_grant, {(cz, cy, cx): sha for cz, cy, cx, sha in inputs["chunks"]}
    )
    return shard_inputs(
        open_prediction(prediction_grant)["class"],  # type: ignore[arg-type]
        labels,
        _box(ctx),
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
    try:
        pieces = label_shard(
            merged,
            labeled,
            int(ctx.payload["min_voxels"]),
            seams(_box(ctx), ctx.payload["shape_zyx"]),
            slab=slab_depth(ctx.memory_budget_bytes, merged.shape),
            budget_bytes=ctx.memory_budget_bytes,
        )
    except TooLarge as exc:
        raise PermanentError(str(exc)) from exc
    del merged, labeled
    grant, shard = ctx.grants[2], ctx.payload["shard"]
    put_bytes(grant, f"scratch/{shard}.npz", pieces.summary)
    put_bytes(
        grant,
        f"scratch/{shard}.local.npz",
        npz(specks=pieces.specks, seam_ids=pieces.seam_ids),
    )
    return {
        "specks": int(pieces.specks.sum()),
        "seam_pieces": len(pieces.seam_ids),
    }


def merge(ctx: JobContext) -> dict[str, Any]:
    grant = ctx.grants[2]

    def summaries():
        for index in range(int(ctx.payload["shards"])):
            summary = get_bytes(grant, f"scratch/{index}.npz")
            if summary is None:
                raise PermanentError(f"Shard {index}'s piece summary is missing.")
            yield summary

    try:
        joined = find_specks(
            summaries(),
            shard_grid(ctx.payload["shape_zyx"]),
            int(ctx.payload["min_voxels"]),
            budget_bytes=ctx.memory_budget_bytes,
        )
    except TooLarge as exc:
        raise PermanentError(str(exc)) from exc
    for index, specks in enumerate(joined.specks):
        put_bytes(grant, f"scratch/{index}.joined.npz", npz(specks=specks))
    return {
        "specks": int(sum(specks.sum() for specks in joined.specks)),
        "pairs": joined.pairs,
    }


def apply(ctx: JobContext) -> dict[str, Any]:
    grant, shard = ctx.grants[2], ctx.payload["shard"]
    local = get_bytes(grant, f"scratch/{shard}.local.npz")
    joined = get_bytes(grant, f"scratch/{shard}.joined.npz")
    if local is None or joined is None:
        raise PermanentError("The list of specks to remove is missing.")
    with np.load(io.BytesIO(local)) as found, np.load(io.BytesIO(joined)) as seam:
        specks = shard_specks(found["specks"], found["seam_ids"], seam["specks"])
    del local, joined
    # Only the classes: which voxels people labeled isn't needed here.
    merged = _merged(ctx)[0]
    try:
        final = apply_shard(
            merged,
            specks,
            slab=slab_depth(ctx.memory_budget_bytes, merged.shape),
            budget_bytes=ctx.memory_budget_bytes,
        )
    except TooLarge as exc:
        raise PermanentError(str(exc)) from exc
    box = _box(ctx)
    region = tuple(slice(box[a], box[a + 3]) for a in range(3))
    open_prediction(grant)["class"][region] = final  # type: ignore[index]
    return {}


def finalize(ctx: JobContext) -> dict[str, Any]:
    grant = ctx.grants[2]
    for index in range(int(ctx.payload["shards"])):
        for name in SCRATCH:
            delete_object(grant, f"scratch/{index}.{name}")
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
