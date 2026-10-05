"""
Pipelines: chains of jobs that make something for a project.

A pipeline starts as one job; when a job succeeds, `after_success` may add
the next jobs from its result (in the same transaction that records the
success, so they are never lost). Each pipeline kind lives in its own module.
"""

from sqlalchemy.ext.asyncio import AsyncSession

from ..db import Job
from . import ingest

# What people call a pipeline, by the kind of its first job.
NAMES = {"ingest.probe": "ingest", "noop": "check"}

_CONTINUATIONS = {"ingest.probe": ingest.after_probe}


async def after_success(db: AsyncSession, job: Job) -> None:
    if continuation := _CONTINUATIONS.get(job.kind):
        await continuation(db, job)


__all__ = ["NAMES", "after_success", "ingest"]
