"""
Pipelines: chains of jobs that make something for a project.

A pipeline starts as one job; when a job succeeds, `after_success` may add
the next jobs from its result, or start other pipelines (in the same
transaction that records the success, so they are never lost). Each pipeline
kind lives in its own module.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from ..db import Job
from ..settings import Settings
from . import compose, export, ingest, labelimport, mesh, predict, train, v1import

# What people call a pipeline, by the kind of its first job.
NAMES = {
    "ingest.probe": "ingest",
    "model.train": "training",
    "predict.prepare": "prediction",
    "predict.region": "proposal",
    "compose.prepare": "segmentation",
    "mesh.block": "meshes",
    "export.files": "export",
    "export.images": "export",
    "v1.probe": "import",
    "v1.labels": "import labels",
    "v1.prediction": "import prediction",
    "labels.probe": "label check",
    "labels.import": "label import",
    "noop": "check",
}


async def _after_finalize(db: AsyncSession, settings: Settings, job: Job) -> None:
    # A v1 import's image is in: start the rest of the import.
    if job.payload.get("source", {}).get("kind") == "v1":
        await v1import.after_image(db, settings, job)


_CONTINUATIONS = {
    "ingest.probe": ingest.after_probe,
    "model.train": train.after_train,
    "artifact.finalize": _after_finalize,
    "v1.probe": v1import.after_probe,
    "v1.labels": v1import.after_labels,
}
_RESULT_CHECKS = {
    "ingest.probe": ingest.check_probe_result,
    "model.train": train.check_result,
    "v1.probe": v1import.check_probe_result,
    "v1.labels": v1import.check_labels_result,
    "labels.probe": labelimport.check_probe_result,
    "labels.import": labelimport.check_import_result,
}


def check_result(job: Job, result: dict) -> None:
    """
    Raise ValueError if a job's result can't continue its pipeline. Runs
    before the success is recorded, so a bad result fails the job instead.
    """
    if check := _RESULT_CHECKS.get(job.kind):
        check(result)


async def after_success(db: AsyncSession, settings: Settings, job: Job) -> None:
    """
    Continue a job's pipeline. Raise `jobs.Rejected` to fail the job instead.
    """
    if continuation := _CONTINUATIONS.get(job.kind):
        await continuation(db, settings, job)


__all__ = [
    "NAMES",
    "after_success",
    "check_result",
    "compose",
    "export",
    "ingest",
    "labelimport",
    "mesh",
    "predict",
    "train",
    "v1import",
]
