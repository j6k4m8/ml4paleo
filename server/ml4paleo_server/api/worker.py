"""
The worker protocol (`/api/worker/v1`). See `ml4paleo.protocol` for the
messages and `ml4paleo_server.jobs.queue` for the job states.
"""

import asyncio
import uuid

from fastapi import APIRouter, HTTPException, Request

from ml4paleo.protocol import (
    MAX_CLAIM_WAIT_SECONDS,
    ClaimIn,
    ClaimOut,
    CompleteIn,
    FailIn,
    HeartbeatIn,
    HeartbeatOut,
    HelloIn,
    HelloOut,
    JobLease,
    ReleaseIn,
)

from .. import jobs
from ..auth.deps import DbSession
from ..jobs.workers import CurrentWorker

router = APIRouter(prefix="/api/worker/v1", tags=["worker"])

# A waiting claim looks for work at least this often, even without a
# notification.
CLAIM_POLL_SECONDS = 5.0


def _lease_lost() -> HTTPException:
    return HTTPException(
        status_code=409, detail="lease_lost: discard this job's output."
    )


@router.post("/hello")
async def hello(body: HelloIn, worker: CurrentWorker, db: DbSession) -> HelloOut:
    await jobs.touch_worker(db, worker, body.caps)
    await db.commit()
    return HelloOut(
        worker_id=worker.id,
        name=worker.name,
        heartbeat_seconds=jobs.HEARTBEAT.total_seconds(),
        lease_seconds=jobs.LEASE.total_seconds(),
    )


@router.post("/claim")
async def claim(
    body: ClaimIn, worker: CurrentWorker, db: DbSession, request: Request
) -> ClaimOut:
    """
    Lease the next job this worker can run, waiting up to `wait_seconds`
    (at most 25) for one to turn up.
    """
    signal: jobs.JobSignal = request.app.state.job_signal
    loop = asyncio.get_running_loop()
    deadline = loop.time() + min(body.wait_seconds, MAX_CLAIM_WAIT_SECONDS)
    await jobs.touch_worker(db, worker, body.caps)
    await db.commit()
    while True:
        # Don't hand a job to a worker that went away or was revoked while
        # it waited; the job would sit unclaimed until its lease ran out.
        if await request.is_disconnected():
            return ClaimOut(job=None)
        if not await jobs.worker_is_active(db, worker.id):
            raise HTTPException(
                status_code=401,
                detail="A valid worker token is required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        claimed = await jobs.claim(db, worker, body.caps)
        # Commit either way, so no connection is held while waiting.
        await db.commit()
        if claimed is not None:
            job = claimed.job
            assert job.lease_expires_at is not None
            return ClaimOut(
                job=JobLease(
                    job_id=job.id,
                    kind=job.kind,
                    payload=job.payload,
                    lease_token=claimed.lease_token,
                    lease_expires_at=job.lease_expires_at,
                    attempt=job.attempts,
                )
            )
        remaining = deadline - loop.time()
        if remaining <= 0:
            return ClaimOut(job=None)
        await signal.wait(min(remaining, CLAIM_POLL_SECONDS))


@router.post("/jobs/{job_id}/heartbeat")
async def heartbeat(
    job_id: uuid.UUID, body: HeartbeatIn, worker: CurrentWorker, db: DbSession
) -> HeartbeatOut:
    try:
        job = await jobs.heartbeat(
            db, job_id, worker, body.lease_token, body.progress, body.message
        )
    except jobs.LeaseLost:
        raise _lease_lost() from None
    await db.commit()
    assert job.lease_expires_at is not None
    return HeartbeatOut(
        lease_expires_at=job.lease_expires_at, cancel=job.cancel_requested
    )


@router.post("/jobs/{job_id}/complete", status_code=204)
async def complete(
    job_id: uuid.UUID, body: CompleteIn, worker: CurrentWorker, db: DbSession
) -> None:
    try:
        await jobs.complete(db, job_id, worker, body.lease_token, body.result)
    except jobs.LeaseLost:
        raise _lease_lost() from None
    except jobs.JobCancelled:
        await db.commit()
        raise HTTPException(
            status_code=409, detail="job_cancelled: discard this job's output."
        ) from None
    await db.commit()


@router.post("/jobs/{job_id}/fail", status_code=204)
async def fail(
    job_id: uuid.UUID, body: FailIn, worker: CurrentWorker, db: DbSession
) -> None:
    try:
        await jobs.fail(
            db, job_id, worker, body.lease_token, body.error, body.retryable
        )
    except jobs.LeaseLost:
        raise _lease_lost() from None
    await db.commit()


@router.post("/jobs/{job_id}/release", status_code=204)
async def release(
    job_id: uuid.UUID, body: ReleaseIn, worker: CurrentWorker, db: DbSession
) -> None:
    try:
        await jobs.release(db, job_id, worker, body.lease_token)
    except jobs.LeaseLost:
        raise _lease_lost() from None
    await db.commit()
