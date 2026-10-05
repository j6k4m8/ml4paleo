"""
The job queue, kept in Postgres.

A job moves through these states:

    blocked -> queued -> leased -> succeeded
                 ^         |
                 +---------+-> failed, cancelled

- `blocked`: waiting for the jobs it depends on to succeed.
- `queued`: ready. Workers claim queued jobs in priority order (tier, then
  pipeline submission time) with `FOR UPDATE SKIP LOCKED`, so two workers
  never get the same job.
- `leased`: a worker holds it. The lease lasts `LEASE` and the worker renews
  it with heartbeats. A worker that goes silent loses the job when the lease
  runs out (`reap`); the job is queued again after a backoff, up to
  `max_attempts` attempts.

Each claim gets a fresh random lease token, and every report about the job
must carry it. A worker that lost its lease (it was too slow, or the job was
given to someone else) gets `LeaseLost` and must throw its output away.

When a job fails for good, the rest of its pipeline is cancelled. Cancelling
a pipeline cancels its waiting jobs at once and asks the workers running the
others to stop (they see `cancel` in their next heartbeat).

These functions work inside the caller's transaction; the caller commits.
`enqueue` and the functions that make jobs claimable send a Postgres
notification so that waiting claims wake up (see `JobSignal`).
"""

import datetime
import hashlib
import secrets
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, not_, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from ml4paleo.protocol import Tier, WorkerCaps

from ..db import Job, JobAttempt, JobDep, Worker, uuid7

LEASE = datetime.timedelta(seconds=120)
HEARTBEAT = datetime.timedelta(seconds=30)
# A failed attempt waits 30 s, then 60 s, then 120 s, ... before its retry.
FIRST_RETRY = datetime.timedelta(seconds=30)
NOTIFY_CHANNEL = "m4p_jobs"

WAITING = ("blocked", "queued")
FINISHED = ("succeeded", "failed", "cancelled")


class LeaseLost(Exception):
    """
    The worker no longer holds the lease it reported on.
    """


class JobCancelled(Exception):
    """
    The job was cancelled while the worker ran it; its output is not wanted.
    """


@dataclass(frozen=True)
class Claimed:
    job: Job
    lease_token: str


def now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _hash(lease_token: str) -> str:
    return hashlib.sha256(lease_token.encode()).hexdigest()


async def notify(db: AsyncSession) -> None:
    """
    Wake waiting claims once the current transaction commits.
    """
    await db.execute(text(f"SELECT pg_notify('{NOTIFY_CHANNEL}', '')"))


async def enqueue(
    db: AsyncSession,
    kind: str,
    payload: dict[str, Any],
    *,
    project_id: uuid.UUID | None = None,
    created_by: uuid.UUID | None = None,
    tier: Tier = Tier.NORMAL,
    pipeline: Job | None = None,
    parent: Job | None = None,
    depends_on: Sequence[Job] = (),
    required_labels: Iterable[str] = (),
    min_vram_gb: float = 0,
    weight: float = 1,
    max_attempts: int = 3,
    idempotency_key: str | None = None,
) -> Job:
    """
    Add a job and return it.

    Without `pipeline`, the job starts a new pipeline (its own root). With it,
    the job joins that pipeline and keeps the pipeline's place in the queue.
    The job waits ("blocked") until every job in `depends_on` has succeeded;
    those must be in the same pipeline, so that a failure there cancels this
    job too.

    If a job with the same `idempotency_key` exists, return it instead.
    """
    if idempotency_key is not None:
        existing = await db.scalar(
            select(Job).where(Job.idempotency_key == idempotency_key)
        )
        if existing is not None:
            return existing
    root_id = pipeline.root_id if pipeline is not None else None
    if any(d.root_id != root_id for d in depends_on):
        raise ValueError("Jobs can only depend on jobs in the same pipeline")
    if any(d.status in ("failed", "cancelled") for d in depends_on):
        raise ValueError("A dependency has already failed or been cancelled")
    job = Job(
        id=uuid7(),
        kind=kind,
        payload=payload,
        project_id=project_id
        if project_id is not None or pipeline is None
        else pipeline.project_id,
        created_by=created_by,
        tier=int(tier),
        status="queued"
        if all(d.status == "succeeded" for d in depends_on)
        else "blocked",
        required_labels=sorted(set(required_labels)),
        min_vram_gb=min_vram_gb,
        weight=weight,
        max_attempts=max_attempts,
        idempotency_key=idempotency_key,
        scale_trigger=tier != Tier.BACKGROUND,
        cancel_requested=False,
        parent_id=parent.id if parent is not None else None,
    )
    if pipeline is None:
        job.root_id = job.id
        job.submitted_at = now()
    else:
        job.root_id = pipeline.root_id
        job.submitted_at = pipeline.submitted_at
    db.add(job)
    await db.flush()
    for dependency in depends_on:
        db.add(JobDep(job_id=job.id, depends_on=dependency.id))
    await db.flush()
    if job.status == "queued":
        await notify(db)
    return job


async def claim(db: AsyncSession, worker: Worker, caps: WorkerCaps) -> Claimed | None:
    """
    Lease the first queued job that `caps` can run, or return None.
    """
    current = now()
    job = await db.scalar(
        select(Job)
        .where(
            Job.status == "queued",
            Job.not_before <= current,
            Job.kind.in_(caps.kinds),
            Job.required_labels.contained_by(sorted(set(caps.labels))),
            Job.min_vram_gb <= caps.vram_gb,
        )
        .order_by(Job.tier, Job.submitted_at, Job.id)
        .limit(1)
        .with_for_update(skip_locked=True)
        .execution_options(populate_existing=True)
    )
    if job is None:
        return None
    lease_token = secrets.token_urlsafe(32)
    job.status = "leased"
    job.lease_worker_id = worker.id
    job.lease_token_hash = _hash(lease_token)
    job.lease_expires_at = current + LEASE
    job.attempts += 1
    job.started_at = job.started_at or current
    job.message = None
    db.add(JobAttempt(job_id=job.id, attempt=job.attempts, worker_id=worker.id))
    await db.flush()
    return Claimed(job=job, lease_token=lease_token)


async def _leased_job(
    db: AsyncSession, job_id: uuid.UUID, worker: Worker, lease_token: str
) -> Job:
    """
    Lock a job and check that `worker` holds its lease with `lease_token`.
    """
    job = await db.scalar(
        select(Job)
        .where(Job.id == job_id)
        .with_for_update()
        # Bulk updates (like cancel_pipeline) don't refresh loaded jobs.
        .execution_options(populate_existing=True)
    )
    if (
        job is None
        or job.status != "leased"
        or job.lease_worker_id != worker.id
        or job.lease_token_hash is None
        or not secrets.compare_digest(job.lease_token_hash, _hash(lease_token))
    ):
        raise LeaseLost
    return job


def _holds_finished(job: Job | None, worker: Worker, lease_token: str) -> bool:
    """
    Whether this lease already finished the job (for repeated reports).
    """
    return (
        job is not None
        and job.lease_worker_id == worker.id
        and job.lease_token_hash is not None
        and secrets.compare_digest(job.lease_token_hash, _hash(lease_token))
    )


async def heartbeat(
    db: AsyncSession,
    job_id: uuid.UUID,
    worker: Worker,
    lease_token: str,
    progress: float | None = None,
    message: str | None = None,
) -> Job:
    """
    Renew a lease and record progress. Returns the job; check
    `cancel_requested` to see whether the worker should stop.
    """
    job = await _leased_job(db, job_id, worker, lease_token)
    job.lease_expires_at = now() + LEASE
    # A worker busy with a long job doesn't claim, so it is seen here instead.
    worker.last_seen_at = now()
    if progress is not None:
        job.progress = progress
    if message is not None:
        job.message = message
    await db.flush()
    return job


async def complete(
    db: AsyncSession,
    job_id: uuid.UUID,
    worker: Worker,
    lease_token: str,
    result: dict[str, Any],
) -> Job:
    """
    Mark a job succeeded and queue the jobs that were waiting only for it.

    Reporting the same success twice is harmless. Raises `JobCancelled` if the
    job was cancelled while it ran.
    """
    try:
        job = await _leased_job(db, job_id, worker, lease_token)
    except LeaseLost:
        job = await db.get(Job, job_id, populate_existing=True)
        if job is not None and job.status == "succeeded":
            if _holds_finished(job, worker, lease_token):
                return job
        raise
    if job.cancel_requested:
        await _finish(db, job, "cancelled", outcome="cancelled")
        raise JobCancelled
    job.result = result
    job.progress = 1
    await _finish(db, job, "succeeded", outcome="succeeded")
    await _unblock_children(db, job.id)
    return job


async def fail(
    db: AsyncSession,
    job_id: uuid.UUID,
    worker: Worker,
    lease_token: str,
    error: str,
    retryable: bool = True,
) -> Job:
    """
    Record a failed attempt. A retryable failure is queued again after a
    backoff while attempts remain; otherwise the job fails for good and the
    rest of its pipeline is cancelled.
    """
    try:
        job = await _leased_job(db, job_id, worker, lease_token)
    except LeaseLost:
        job = await db.get(Job, job_id, populate_existing=True)
        if job is not None and job.status != "leased":
            if _holds_finished(job, worker, lease_token):
                return job
        raise
    await _end_attempt(db, job, error=error, retryable=retryable)
    return job


async def release(
    db: AsyncSession, job_id: uuid.UUID, worker: Worker, lease_token: str
) -> Job:
    """
    Give a job back without counting the attempt, so another worker can take
    it right away.
    """
    job = await _leased_job(db, job_id, worker, lease_token)
    await _record_attempt(db, job, "released")
    job.attempts -= 1
    if job.cancel_requested:
        await _finish(db, job, "cancelled")
        return job
    _requeue(job, delay=datetime.timedelta(0))
    await notify(db)
    return job


async def _end_attempt(
    db: AsyncSession,
    job: Job,
    *,
    error: str,
    retryable: bool,
    outcome: str = "failed",
) -> None:
    if job.cancel_requested:
        await _finish(db, job, "cancelled", outcome="cancelled", error=error)
        return
    if retryable and job.attempts < job.max_attempts:
        await _record_attempt(db, job, outcome, error)
        job.error = error
        _requeue(job, delay=FIRST_RETRY * 2 ** (job.attempts - 1))
        await notify(db)
        return
    await _finish(db, job, "failed", outcome=outcome, error=error)
    await cancel_pipeline(db, job.root_id)


def _requeue(job: Job, delay: datetime.timedelta) -> None:
    job.status = "queued"
    job.not_before = now() + delay
    job.lease_expires_at = None
    job.progress = 0
    # Keep lease_worker_id and lease_token_hash, so a repeated report from
    # the old lease is recognized until the job is claimed again.


async def _record_attempt(
    db: AsyncSession, job: Job, outcome: str, error: str | None = None
) -> None:
    await db.execute(
        update(JobAttempt)
        .where(
            JobAttempt.job_id == job.id,
            JobAttempt.attempt == job.attempts,
            JobAttempt.outcome.is_(None),
        )
        .values(outcome=outcome, error=error, ended_at=now())
    )


async def _finish(
    db: AsyncSession,
    job: Job,
    status: str,
    *,
    outcome: str | None = None,
    error: str | None = None,
) -> None:
    if outcome is not None:
        await _record_attempt(db, job, outcome, error)
    job.status = status
    job.finished_at = now()
    job.lease_expires_at = None
    if error is not None:
        job.error = error
    await db.flush()


async def _unblock_children(db: AsyncSession, job_id: uuid.UUID) -> None:
    """
    Queue the blocked jobs that depend on `job_id` and have no other
    unfinished dependencies.

    The children are locked first, and checked in a separate statement after
    the locks are held. When two parents of one child finish at the same time,
    the second to lock therefore sees the first one's committed success, so
    the child can't be left blocked.
    """
    children = select(JobDep.job_id).where(JobDep.depends_on == job_id)
    locked = (
        await db.scalars(
            select(Job.id)
            .where(Job.id.in_(children), Job.status == "blocked")
            .order_by(Job.id)
            .with_for_update()
        )
    ).all()
    if not locked:
        return
    await _queue_ready(db, Job.id.in_(locked))


async def _queue_ready(db: AsyncSession, *conditions) -> int:
    unfinished = (
        select(JobDep.job_id)
        .join(Job, Job.id == JobDep.depends_on)
        .where(Job.status != "succeeded")
    )
    queued = await db.execute(
        update(Job)
        .where(Job.status == "blocked", not_(Job.id.in_(unfinished)), *conditions)
        .values(status="queued", not_before=func.now())
        .returning(Job.id)
        .execution_options(synchronize_session=False)
    )
    count = len(queued.all())
    if count:
        await notify(db)
    return count


async def cancel_pipeline(db: AsyncSession, root_id: uuid.UUID) -> None:
    """
    Cancel a pipeline: waiting jobs stop at once, and workers running the
    others are asked to stop.
    """
    await db.execute(
        update(Job)
        .where(Job.root_id == root_id, Job.status.in_(WAITING))
        .values(status="cancelled", finished_at=func.now(), cancel_requested=True)
        .execution_options(synchronize_session=False)
    )
    await db.execute(
        update(Job)
        .where(Job.root_id == root_id, Job.status == "leased")
        .values(cancel_requested=True)
        .execution_options(synchronize_session=False)
    )


async def reap(db: AsyncSession) -> int:
    """
    Take jobs back from workers that went silent, and queue blocked jobs whose
    dependencies have all succeeded (a safety net). Returns how many leases
    expired.
    """
    expired = (
        await db.scalars(
            select(Job)
            .where(Job.status == "leased", Job.lease_expires_at < now())
            .with_for_update(skip_locked=True)
            .execution_options(populate_existing=True)
        )
    ).all()
    for job in expired:
        await _end_attempt(
            db,
            job,
            error=f"The worker stopped responding (attempt {job.attempts}).",
            retryable=True,
            outcome="expired",
        )
    await _queue_ready(db)
    return len(expired)


async def requeue_worker_jobs(db: AsyncSession, worker_id: uuid.UUID) -> None:
    """
    Take back every job a worker holds, for example when its token is revoked.
    """
    held = (
        await db.scalars(
            select(Job)
            .where(Job.status == "leased", Job.lease_worker_id == worker_id)
            .with_for_update()
            .execution_options(populate_existing=True)
        )
    ).all()
    for job in held:
        await _end_attempt(
            db, job, error="The worker was revoked.", retryable=True, outcome="expired"
        )


@dataclass(frozen=True)
class PipelineStatus:
    root_id: uuid.UUID
    # "waiting", "running", "succeeded", "failed", or "cancelled".
    status: str
    progress: float
    jobs: int


async def pipeline_status(db: AsyncSession, root_id: uuid.UUID) -> PipelineStatus:
    """
    Summarize a pipeline. Its status is derived from its jobs' statuses, and
    its progress is their weighted mean progress.
    """
    rows = (
        await db.execute(
            select(Job.status, Job.weight, Job.progress).where(Job.root_id == root_id)
        )
    ).all()
    statuses = {row.status for row in rows}
    total_weight = sum(row.weight for row in rows) or 1
    done = sum(
        row.weight * (1 if row.status == "succeeded" else row.progress) for row in rows
    )
    if "failed" in statuses:
        status = "failed"
    elif statuses and statuses <= {"succeeded"}:
        status = "succeeded"
    elif "cancelled" in statuses and not statuses & {"leased", "queued", "blocked"}:
        status = "cancelled"
    elif "leased" in statuses or "succeeded" in statuses:
        status = "running"
    else:
        status = "waiting"
    return PipelineStatus(
        root_id=root_id,
        status=status,
        progress=min(1.0, done / total_weight),
        jobs=len(rows),
    )


async def touch_worker(db: AsyncSession, worker: Worker, caps: WorkerCaps) -> None:
    worker.caps = caps.model_dump()
    worker.last_seen_at = now()
    await db.flush()


__all__ = [
    "FINISHED",
    "HEARTBEAT",
    "LEASE",
    "NOTIFY_CHANNEL",
    "Claimed",
    "JobCancelled",
    "LeaseLost",
    "PipelineStatus",
    "cancel_pipeline",
    "claim",
    "complete",
    "enqueue",
    "fail",
    "heartbeat",
    "notify",
    "pipeline_status",
    "reap",
    "release",
    "requeue_worker_jobs",
    "touch_worker",
]
