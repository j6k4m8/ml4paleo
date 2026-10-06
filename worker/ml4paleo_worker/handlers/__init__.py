"""
Job handlers, by job kind. A handler takes a `JobContext` and returns a small
JSON-able dict (the job's result); it raises to fail the job.

Raise `ml4paleo_worker.context.PermanentError` for failures that a retry
can't fix, such as a file that isn't an image.
"""

from collections.abc import Callable
from typing import Any

from ..context import JobContext
from . import compose, export, ingest, mesh, noop, predict, train

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
}
