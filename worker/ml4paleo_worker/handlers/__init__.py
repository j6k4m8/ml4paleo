"""
Job handlers, by job kind. A handler takes a `JobContext` and returns a small
JSON-able dict (the job's result); it raises to fail the job.

Raise `ml4paleo_worker.context.PermanentError` for failures that a retry
can't fix, such as a file that isn't an image.
"""

from collections.abc import Callable
from typing import Any

from ..context import JobContext
from . import (
    compose,
    export,
    ingest,
    labelimport,
    mesh,
    noop,
    predict,
    train,
    v1import,
)

Handler = Callable[[JobContext], dict[str, Any]]

HANDLERS: dict[str, Handler] = {
    "noop": noop.run,
    "ingest.probe": ingest.probe,
    "ingest.slab": ingest.slab,
    "pyramid.level": ingest.pyramid,
    "artifact.finalize": ingest.finalize,
    "model.train": train.run,
    "predict.prepare": predict.prepare,
    "predict.shard": predict.shard,
    "prediction.finalize": predict.finalize,
    "predict.region": predict.region,
    "predict.live": predict.region,
    "compose.prepare": compose.prepare,
    "cc.block": compose.block,
    "cc.merge": compose.merge,
    "cc.apply": compose.apply,
    "compose.finalize": compose.finalize,
    "mesh.block": mesh.block,
    "mesh.join": mesh.join_class,
    "mesh.finalize": mesh.finalize,
    "export.files": export.files,
    "export.images": export.images,
    "labels.probe": labelimport.probe,
    "labels.import": labelimport.run,
    "v1.probe": v1import.probe,
    "v1.slab": v1import.slab,
    "v1.labels": v1import.labels,
    "v1.prediction": v1import.prediction,
}

# A worker with the v1 volume can read every v1 job's files, so it runs the
# import's own jobs and nothing that parses people's uploads.
V1_HANDLERS = {kind: run for kind, run in HANDLERS.items() if kind.startswith("v1.")}
