"""
Prediction jobs (see the server's `pipelines/predict.py`).

Grants, in order: the image artifact (read), the model artifact (read), and
the prediction artifact (write). A shard job predicts its shard on the
job's share of the worker's CPUs, in blocks sized to its memory budget (see
`predict_box`).
"""

import json
import pathlib
import tempfile
import threading
from typing import Any

from ml4paleo.ome import OmeImage
from ml4paleo.segmentation.plugin import Predictor, get_plugin
from ml4paleo.segmentation.predict import (
    create_prediction,
    open_prediction,
    predict_box,
)
from ml4paleo.storage import StorageGrant, get_bytes, write_manifest

from ..context import JobContext, PermanentError

# The last model loaded, since a worker usually runs many shards of one
# prediction in a row.
_loaded: dict[str, Predictor] = {}
_lock = threading.Lock()


def _predictor(ctx: JobContext, grant: StorageGrant) -> Predictor:
    model_id = ctx.payload["model_id"]
    with _lock:
        if model_id in _loaded:
            return _loaded[model_id]
    try:
        plugin = get_plugin(ctx.payload["plugin"])()
    except ValueError as exc:
        raise PermanentError(str(exc)) from exc
    manifest = get_bytes(grant, "_MANIFEST.json")
    if manifest is None:
        raise PermanentError("The model's files are missing.")
    files = json.loads(manifest).get("files", [])
    with tempfile.TemporaryDirectory(prefix="m4p-model-") as scratch:
        directory = pathlib.Path(scratch)
        for name in files:
            data = get_bytes(grant, name)
            if data is None:
                raise PermanentError(f"The model file {name} is missing.")
            (directory / name).write_bytes(data)
        predictor = plugin.load(directory)
    with _lock:
        _loaded.clear()
        _loaded[model_id] = predictor
    return predictor


def prepare(ctx: JobContext) -> dict[str, Any]:
    create_prediction(ctx.grants[2], ctx.payload["shape_zyx"])
    return {}


def shard(ctx: JobContext) -> dict[str, Any]:
    image_grant, model_grant, prediction_grant = ctx.grants
    box = tuple(int(n) for n in ctx.payload["box"])
    predictor = _predictor(ctx, model_grant)
    # The job's share of the worker's CPUs, as for training; the blocks are
    # sized for it.
    predictor.threads = ctx.threads

    def progress(fraction: float) -> None:
        ctx.progress(fraction)
        ctx.check()

    predict_box(
        predictor,
        OmeImage.open(image_grant).array(0),
        box,  # type: ignore[arg-type]
        tuple(ctx.payload["window"]),  # type: ignore[arg-type]
        list(ctx.payload["class_values"]),
        open_prediction(prediction_grant),
        ctx.memory_budget_bytes,
        progress=progress,
    )
    return {"box": list(box)}


def region(ctx: JobContext) -> dict[str, Any]:
    """
    A proposal: one box (an ROI) predicted into arrays of the image's size,
    then the manifest, all in one job.
    """
    create_prediction(ctx.grants[2], ctx.payload["shape_zyx"])
    shard(ctx)
    write_manifest(
        ctx.grants[2],
        {
            "kind": "prediction",
            "model_id": ctx.payload["model_id"],
            "class_values": ctx.payload["class_values"],
            "shape_zyx": ctx.payload["shape_zyx"],
            "window": ctx.payload["window"],
            "box": ctx.payload["box"],
        },
    )
    return {"box": ctx.payload["box"]}


def finalize(ctx: JobContext) -> dict[str, Any]:
    write_manifest(
        ctx.grants[2],
        {
            "kind": "prediction",
            "model_id": ctx.payload["model_id"],
            "class_values": ctx.payload["class_values"],
            "shape_zyx": ctx.payload["shape_zyx"],
            # The window the image was normalized with.
            "window": ctx.payload["window"],
        },
    )
    return {}
