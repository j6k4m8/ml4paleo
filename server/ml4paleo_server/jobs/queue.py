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

When a job fails for good, the rest of its pipeline is cancelled. A pipeline
is cancelled when its root job's `cancel_requested` flag is set; that flag is
the authority. `cancel_pipeline` locks only the root, then cancels waiting
jobs and flags running ones without waiting for rows that other transactions
hold. Every other path checks the root's flag too (claims skip cancelled
pipelines, reports and heartbeats pick the flag up, and the reaper cancels
what is left), so a job that was busy at that moment can't slip through.

Locks are taken in a fixed order to avoid deadlocks: a transaction locks the
jobs it reports on or depends on, then that pipeline's root, then (for
completions) the waiting children. `cancel_pipeline` waits only for the root;
it skips other rows that are busy. Worker rows are only touched after job
rows. Row locks are `FOR NO KEY UPDATE` (no key column ever changes), so they
don't collide with the `KEY SHARE` locks that foreign-key checks take on the
root and worker rows.

These functions work inside the caller's transaction; the caller commits.
`enqueue` and the functions that make jobs claimable send a Postgres
notification so that waiting claims wake up (see `JobSignal`).
"""

import datetime
import hashlib
import secrets
import uuid
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import exists, func, not_, or_, select, text, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from ml4paleo.protocol import Tier, WorkerCaps
from ml4paleo.storage import StorageGrant

from ..db import Job, JobAttempt, JobDep, Worker, uuid7

LEASE = datetime.timedelta(seconds=120)
HEARTBEAT = datetime.timedelta(seconds=30)
# A failed attempt waits 30 s, then 60 s, then 120 s, ... before its retry.
FIRST_RETRY = datetime.timedelta(seconds=30)
NOTIFY_CHANNEL = "m4p_jobs"

WAITING = ("blocked", "queued")
FINISHED = ("succeeded", "failed", "cancelled")
# Heartbeats refresh a worker's last_seen_at at most this often.
SEEN_EVERY = datetime.timedelta(seconds=10)

_root = aliased(Job)


def _root_cancelled():
    """
    SQL: the job's pipeline has been cancelled.
    """
    return exists().where(_root.id == Job.root_id, _root.cancel_requested)


class LeaseLost(Exception):
    """
    The worker no longer holds the lease it reported on.
    """


class JobCancelled(Exception):
    """
    The job was cancelled while the worker ran it; its output is not wanted.
    """


class Rejected(Exception):
    """
    A job's output can't be accepted (for example it is missing its manifest,
    or it doesn't fit the owner's storage quota). The attempt fails, and is
    retried if `retryable`.
    """

    def __init__(self, message: str, *, retryable: bool):
        super().__init__(message)
        self.retryable = retryable


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
    grants: Sequence[dict[str, str]] = (),
) -> Job:
    """
    Add a job and return it.

    Without `pipeline`, the job starts a new pipeline (its own root). With it,
    the job joins that pipeline and keeps the pipeline's place in the queue.
    The job waits ("blocked") until every job in `depends_on` has succeeded;
    those must be in the same pipeline, so that a failure there cancels this
    job too.

    If a job with the same `idempotency_key` exists, return it instead.

    The pipeline's root and the dependencies are re-read under a share lock,
    so their state can't change underneath: a dependency that is finishing
    either sees this job (and releases it) or has finished before this check.
    """
    if idempotency_key is not None:
        existing = await db.scalar(
            select(Job).where(Job.idempotency_key == idempotency_key)
        )
        if existing is not None:
            return existing
    root_id = pipeline.root_id if pipeline is not None else None
    # Dependencies first, then the root: the same order as the paths that
    # report on a job and then cancel its pipeline.
    dependency_ids = sorted({d.id for d in depends_on})
    dependencies = (
        await db.execute(
            select(Job.id, Job.root_id, Job.status)
            .where(Job.id.in_(dependency_ids))
            .order_by(Job.id)
            .with_for_update(read=True)
        )
    ).all()
    if len(dependencies) != len(dependency_ids):
        raise ValueError("A dependency does not exist")
    if any(d.root_id != root_id for d in dependencies):
        raise ValueError("Jobs can only depend on jobs in the same pipeline")
    if any(d.status in ("failed", "cancelled") for d in dependencies):
        raise ValueError("A dependency has already failed or been cancelled")
    if root_id is not None:
        cancelled = await db.scalar(
            select(Job.cancel_requested)
            .where(Job.id == root_id)
            .with_for_update(read=True)
        )
        if cancelled:
            raise ValueError("The pipeline was cancelled")
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
        if all(d.status == "succeeded" for d in dependencies)
        else "blocked",
        required_labels=sorted(set(required_labels)),
        min_vram_gb=min_vram_gb,
        weight=weight,
        max_attempts=max_attempts,
        idempotency_key=idempotency_key,
        scale_trigger=tier != Tier.BACKGROUND,
        cancel_requested=False,
        parent_id=parent.id if parent is not None else None,
        grants=[_check_grant(g) for g in grants],
    )
    if pipeline is None:
        job.root_id = job.id
        job.submitted_at = now()
    else:
        job.root_id = pipeline.root_id
        job.submitted_at = pipeline.submitted_at
    try:
        async with db.begin_nested():
            db.add(job)
            await db.flush()
    except IntegrityError:
        # Another transaction added a job with this key first.
        if idempotency_key is None:
            raise
        existing = await db.scalar(
            select(Job).where(Job.idempotency_key == idempotency_key)
        )
        if existing is None:
            raise
        return existing
    for dependency_id in dependency_ids:
        db.add(JobDep(job_id=job.id, depends_on=dependency_id))
    await db.flush()
    if job.status == "queued":
        await notify(db)
    return job


def _check_grant(grant: dict[str, str]) -> dict[str, str]:
    path, access = grant.get("path", ""), grant.get("access")
    if access not in ("r", "rw") or set(grant) != {"path", "access"}:
        raise ValueError(f"Bad job grant {grant!r}")
    # Raises for empty, ".", "..", and other unsafe path segments.
    StorageGrant(url="s3://check").child(path)
    if not path.startswith("projects/"):
        raise ValueError(f"Job grants must be under projects/: {path!r}")
    return {"path": path, "access": access}


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
            not_(Job.cancel_requested),
            not_(_root_cancelled()),
            Job.kind.in_(caps.kinds),
            Job.required_labels.contained_by(sorted(set(caps.labels))),
            Job.min_vram_gb <= caps.vram_gb,
        )
        .order_by(Job.tier, Job.submitted_at, Job.id)
        .limit(1)
        .with_for_update(skip_locked=True, key_share=True)
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
        .with_for_update(key_share=True)
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
    await _pick_up_cancellation(db, job)
    return job


async def _pick_up_cancellation(db: AsyncSession, job: Job) -> None:
    """
    Flag a job whose pipeline was cancelled while the job's row was busy.
    """
    if job.cancel_requested or job.root_id == job.id:
        return
    if await db.scalar(select(Job.cancel_requested).where(Job.id == job.root_id)):
        job.cancel_requested = True


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
    await db.execute(
        update(Worker)
        .where(
            Worker.id == worker.id,
            or_(
                Worker.last_seen_at.is_(None),
                Worker.last_seen_at < now() - SEEN_EVERY,
            ),
        )
        .values(last_seen_at=func.now())
        .execution_options(synchronize_session=False)
    )
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
    check: Callable[[Job], Awaitable[None]] | None = None,
) -> Job:
    """
    Mark a job succeeded and queue the jobs that were waiting only for it.

    `check` runs first, in the same transaction (the server commits the job's
    artifacts there); if it raises `Rejected`, the attempt fails instead and
    the exception propagates.

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
    if check is not None:
        try:
            # A savepoint, so a rejection undoes everything the check did
            # (for example quota reserved for an earlier artifact).
            async with db.begin_nested():
                await check(job)
        except Rejected as exc:
            await db.refresh(job)
            await _end_attempt(db, job, error=str(exc), retryable=exc.retryable)
            raise
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
            .with_for_update(key_share=True)
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
        .where(
            Job.status == "blocked",
            not_(Job.id.in_(unfinished)),
            not_(_root_cancelled()),
            *conditions,
        )
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

    Only the root is locked and waited for. Jobs whose rows other transactions
    hold right now are skipped here and caught by the root's flag instead.
    """
    root = await db.scalar(
        select(Job)
        .where(Job.id == root_id)
        .with_for_update(key_share=True)
        .execution_options(populate_existing=True)
    )
    if root is None:
        return
    root.cancel_requested = True
    if root.status in WAITING:
        root.status = "cancelled"
        root.finished_at = now()
    await db.flush()
    await _cancel_waiting(db, Job.root_id == root_id)
    running = (
        select(Job.id)
        .where(Job.root_id == root_id, Job.status == "leased")
        .with_for_update(skip_locked=True, key_share=True)
    )
    await db.execute(
        update(Job)
        .where(Job.id.in_(running))
        .values(cancel_requested=True)
        .execution_options(synchronize_session=False)
    )


async def _cancel_waiting(db: AsyncSession, *conditions) -> None:
    waiting = (
        select(Job.id)
        .where(Job.status.in_(WAITING), *conditions)
        .with_for_update(skip_locked=True, key_share=True)
    )
    await db.execute(
        update(Job)
        .where(Job.id.in_(waiting))
        .values(status="cancelled", finished_at=func.now(), cancel_requested=True)
        .execution_options(synchronize_session=False)
    )


async def reap_one(db: AsyncSession) -> bool:
    """
    Take one job back from a worker that went silent. Returns False when
    there are none. The housekeeper commits after each, so it never holds
    locks on several jobs at once.
    """
    job = await db.scalar(
        select(Job)
        .where(Job.status == "leased", Job.lease_expires_at < now())
        .limit(1)
        .with_for_update(skip_locked=True, key_share=True)
        .execution_options(populate_existing=True)
    )
    if job is None:
        return False
    await _pick_up_cancellation(db, job)
    await _end_attempt(
        db,
        job,
        error=f"The worker stopped responding (attempt {job.attempts}).",
        retryable=True,
        outcome="expired",
    )
    return True


async def sweep(db: AsyncSession) -> None:
    """
    Safety nets: cancel waiting jobs left in cancelled pipelines, and queue
    blocked jobs whose dependencies have all succeeded.
    """
    await _cancel_waiting(db, _root_cancelled())
    await _queue_ready(db)


async def reap(db: AsyncSession) -> int:
    """
    Take back every expired lease, then `sweep`. Returns how many leases
    expired.
    """
    expired = 0
    while await reap_one(db):
        expired += 1
    await sweep(db)
    return expired


async def requeue_worker_jobs(db: AsyncSession, worker_id: uuid.UUID) -> None:
    """
    Take back every job a worker holds, for example when its token is revoked.
    """
    held = (
        await db.scalars(
            select(Job)
            .where(Job.status == "leased", Job.lease_worker_id == worker_id)
            .with_for_update(key_share=True)
            .execution_options(populate_existing=True)
        )
    ).all()
    for job in held:
        await _pick_up_cancellation(db, job)
        await _end_attempt(
            db, job, error="The worker was revoked.", retryable=True, outcome="expired"
        )


async def worker_is_active(db: AsyncSession, worker_id: uuid.UUID) -> bool:
    row = (
        await db.execute(
            select(Worker.status, Worker.expires_at).where(Worker.id == worker_id)
        )
    ).first()
    return (
        row is not None
        and row.status == "active"
        and (row.expires_at is None or row.expires_at > now())
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
    cancelled = await db.scalar(select(Job.cancel_requested).where(Job.id == root_id))
    statuses = {row.status for row in rows}
    total_weight = sum(row.weight for row in rows) or 1
    done = sum(
        row.weight * (1 if row.status == "succeeded" else row.progress) for row in rows
    )
    if "failed" in statuses:
        status = "failed"
    elif statuses and statuses <= {"succeeded"}:
        status = "succeeded"
    elif cancelled and "leased" not in statuses:
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
    "Rejected",
    "cancel_pipeline",
    "claim",
    "complete",
    "enqueue",
    "fail",
    "heartbeat",
    "notify",
    "pipeline_status",
    "reap",
    "reap_one",
    "release",
    "requeue_worker_jobs",
    "sweep",
    "touch_worker",
    "worker_is_active",
]
