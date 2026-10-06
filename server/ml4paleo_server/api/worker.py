"""
The worker protocol (`/api/worker/v1`). See `ml4paleo.protocol` for the
messages and `ml4paleo_server.jobs.queue` for the job states.
"""

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from starlette.concurrency import run_in_threadpool

from ml4paleo.labels import Source
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
    LabelOpIn,
    ReleaseIn,
)

from .. import artifacts, broker, jobs, labels, pipelines
from ..auth.deps import DbSession, SettingsDep
from ..jobs.workers import CurrentWorker
from .labels import MAX_TOOL_BYTES, DeltaIn, allowed_values, check_values

router = APIRouter(prefix="/api/worker/v1", tags=["worker"])

# A waiting claim looks for work at least this often, even without a
# notification.
CLAIM_POLL_SECONDS = 5.0
# The kinds of job that may write labels, and the source their edits get.
LABEL_WRITERS = {"v1.labels": Source.HUMAN}


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
    body: ClaimIn,
    worker: CurrentWorker,
    db: DbSession,
    settings: SettingsDep,
    request: Request,
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
                    # The server's address as this worker reached it.
                    grants=broker.grants_for(
                        settings,
                        worker,
                        job,
                        claimed.lease_token,
                        str(request.base_url),
                    ),
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
    job_id: uuid.UUID,
    body: CompleteIn,
    worker: CurrentWorker,
    db: DbSession,
    settings: SettingsDep,
) -> None:
    """
    Mark the job succeeded, committing the artifacts it produced and adding
    its pipeline's next jobs in the same transaction.
    """

    async def commit_artifacts(job):
        try:
            pipelines.check_result(job, body.result)
        except ValueError as exc:
            raise jobs.Rejected(
                f"The job's result is malformed: {exc}", retryable=False
            ) from None
        await artifacts.commit_outputs(db, settings, job)

    async def continue_pipeline(job):
        await pipelines.after_success(db, job)

    try:
        await jobs.complete(
            db,
            job_id,
            worker,
            body.lease_token,
            body.result,
            check=commit_artifacts,
            after=continue_pipeline,
        )
    except jobs.LeaseLost:
        raise _lease_lost() from None
    except jobs.JobCancelled:
        await db.commit()
        raise HTTPException(
            status_code=409, detail="job_cancelled: discard this job's output."
        ) from None
    except jobs.Rejected as exc:
        await db.commit()
        raise HTTPException(status_code=409, detail=f"job_rejected: {exc}") from None
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


@router.post("/jobs/{job_id}/label-ops", status_code=201)
async def label_op(
    job_id: uuid.UUID,
    body: LabelOpIn,
    worker: CurrentWorker,
    db: DbSession,
    settings: SettingsDep,
) -> dict[str, Any]:
    """
    Apply a label edit to the job's project, as the annotator's edits are
    applied, for the kinds of job that bring in labels. The job stays locked
    until the edit commits, so an edit can't land after its job has finished.
    """
    try:
        job = await jobs.leased_job(db, job_id, worker, body.lease_token)
    except jobs.LeaseLost:
        raise _lease_lost() from None
    source = LABEL_WRITERS.get(job.kind)
    if source is None or job.project_id is None:
        raise HTTPException(status_code=403, detail="This job can't write labels.")
    if len(json.dumps(body.tool)) > MAX_TOOL_BYTES:
        raise HTTPException(status_code=422, detail="tool is too large")
    try:
        deltas = [DeltaIn.model_validate(delta).to_delta() for delta in body.deltas]
        allowed = await allowed_values(db, job.project_id)
        await run_in_threadpool(check_values, deltas, allowed)
        result = await labels.apply_edit(
            db,
            settings,
            job.project_id,
            client_op_id=body.client_op_id,
            deltas=deltas,
            source=source,
            tool=body.tool,
            job_id=job.id,
        )
    except labels.NoImage:
        raise HTTPException(
            status_code=422, detail="This project has no image yet."
        ) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    await db.commit()
    return {"seq": result.seq}
