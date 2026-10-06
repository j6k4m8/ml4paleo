"""
Pipelines: chains of jobs that make something for a project.

A pipeline starts as one job; when a job succeeds, `after_success` may add
the next jobs from its result (in the same transaction that records the
success, so they are never lost). Each pipeline kind lives in its own module.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from ..db import Job
from . import ingest, predict, train

# What people call a pipeline, by the kind of its first job.
NAMES = {
    "ingest.probe": "ingest",
    "model.train": "training",
    "predict.prepare": "prediction",
    "noop": "check",
}

_CONTINUATIONS = {"ingest.probe": ingest.after_probe, "model.train": train.after_train}
_RESULT_CHECKS = {
    "ingest.probe": ingest.check_probe_result,
    "model.train": train.check_result,
}


def check_result(job: Job, result: dict) -> None:
    """
    Raise ValueError if a job's result can't continue its pipeline. Runs
    before the success is recorded, so a bad result fails the job instead.
    """
    if check := _RESULT_CHECKS.get(job.kind):
        check(result)


async def after_success(db: AsyncSession, job: Job) -> None:
    if continuation := _CONTINUATIONS.get(job.kind):
        await continuation(db, job)


__all__ = ["NAMES", "after_success", "check_result", "ingest", "predict", "train"]
