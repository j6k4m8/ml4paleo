"""
`model.train`: train a segmentation model on a training set.

Grants, in order: the image artifact (read), the project's labels (read;
their blobs), the training set (read; its manifest), and the model artifact
(write). The plugin writes its files into a scratch directory; they are
copied into the model artifact, and the artifact's manifest goes last.

Training crops are sized so one, with its halo, fits in half the job's
memory budget at the plugin's cost per voxel (the plugin keeps its samples
and model in the other half), and the plugin trains on the job's share of
the worker's CPUs (`ctx.threads`).
"""

import json
import pathlib
import tempfile
from typing import Any

from ml4paleo.ome import OmeImage
from ml4paleo.segmentation.dataset import (
    BlobLabels,
    MissingLabels,
    RoiSpec,
    TrainingSet,
    tile_for,
)
from ml4paleo.segmentation.plugin import get_plugin
from ml4paleo.storage import get_bytes, put_bytes, write_manifest

from ..context import JobContext, PermanentError


def run(ctx: JobContext) -> dict[str, Any]:
    image_grant, labels_grant, training_grant, model_grant = ctx.grants
    raw = get_bytes(training_grant, "manifest.json")
    if raw is None:
        raise PermanentError("The training set's manifest is missing.")
    manifest = json.loads(raw)
    try:
        plugin_class = get_plugin(ctx.payload["plugin"])
        params = plugin_class.Params(**ctx.payload.get("params", {}))
    except ValueError as exc:
        raise PermanentError(str(exc)) from exc
    class_shas = {(row[0], row[1], row[2]): row[3] for row in manifest["chunks"]}
    source_shas = (
        {
            (row[0], row[1], row[2]): row[4]
            for row in manifest["chunks"]
            if len(row) >= 5 and row[4] is not None
        }
        if int(manifest.get("version", 1)) >= 2
        else None
    )
    labels = BlobLabels(labels_grant, class_shas, source_shas)
    plugin = plugin_class()
    image = OmeImage.open(image_grant).array(0)
    cost = plugin.crop_cost(params, int(image.shape[0]), ctx.threads)
    data = TrainingSet(
        image=image,
        labels=labels,
        labeled_chunks=labels.shas.keys(),
        rois=[
            RoiSpec(tuple(roi["bbox"]), roi["status"], roi["split"])  # type: ignore[arg-type]
            for roi in manifest["rois"]
        ],
        class_values=list(manifest["class_values"]),
        window=tuple(manifest["image"]["window"]),  # type: ignore[arg-type]
        tile=tile_for(ctx.memory_budget_bytes // 2, cost),
    )
    with tempfile.TemporaryDirectory(prefix="m4p-train-") as scratch:
        out = pathlib.Path(scratch)
        try:
            result = plugin.train(data, params, out, ctx)
        except MissingLabels as exc:
            # The training set pins labels that are gone; retrying won't
            # bring them back.
            raise PermanentError(str(exc)) from exc
        except ValueError as exc:
            # Bad training data (for example labels of only one class).
            raise PermanentError(str(exc)) from exc
        for name in result.files:
            put_bytes(model_grant, name, (out / name).read_bytes())
    samples = {str(k): v for k, v in result.samples.items()}
    write_manifest(
        model_grant,
        {
            "kind": "model",
            "plugin": plugin_class.name,
            "plugin_version": plugin_class.version,
            "params": params.model_dump(),
            "class_values": data.class_values,
            # The window crops were normalized with, for prediction to use.
            "window": [float(v) for v in data.window],
            "training_set": ctx.payload["training_set"],
            "metrics": result.metrics,
            "samples": samples,
            "files": result.files,
        },
    )
    return {
        "metrics": result.metrics,
        "samples": samples,
        "plugin_version": plugin_class.version,
    }
