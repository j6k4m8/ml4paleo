"""
`model.train`: train a segmentation model on a training set.

Grants, in order: the image artifact (read), the project's labels (read;
their blobs), the training set (read; its manifest), and the model artifact
(write). The plugin writes its files into a scratch directory; they are
copied into the model artifact, and the artifact's manifest goes last.
"""

import json
import pathlib
import tempfile
from typing import Any

from ml4paleo.ome import OmeImage
from ml4paleo.segmentation.dataset import BlobLabels, RoiSpec, TrainingSet
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
    labels = BlobLabels(
        labels_grant, {(cz, cy, cx): sha for cz, cy, cx, sha in manifest["chunks"]}
    )
    data = TrainingSet(
        image=OmeImage.open(image_grant).array(0),
        labels=labels,
        labeled_chunks=labels.shas.keys(),
        rois=[
            RoiSpec(tuple(roi["bbox"]), roi["status"], roi["split"])  # type: ignore[arg-type]
            for roi in manifest["rois"]
        ],
        class_values=list(manifest["class_values"]),
        window=tuple(manifest["image"]["window"]),  # type: ignore[arg-type]
    )
    plugin = plugin_class()
    with tempfile.TemporaryDirectory(prefix="m4p-train-") as scratch:
        out = pathlib.Path(scratch)
        try:
            result = plugin.train(data, params, out, ctx)
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
