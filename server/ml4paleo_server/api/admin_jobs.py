"""
Site administration for workers and jobs: worker tokens, the job list, and a
diagnostic job that checks the workers end to end.
"""

import datetime
import uuid
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from ml4paleo.protocol import Tier

from .. import audit, jobs
from ..auth.deps import AdminAuth, DbSession
from ..auth.tokens import token_hash
from ..db import Job, Worker
from ..jobs.workers import LOCAL_WORKER_NAME, new_worker_token

router = APIRouter(prefix="/api/admin", tags=["admin"])

JobStatus = Literal["blocked", "queued", "leased", "succeeded", "failed", "cancelled"]

# A worker counts as online if it was seen within this long (a waiting claim
# returns at least every 25 seconds, and a running job heartbeats every 30).
ONLINE_WINDOW = datetime.timedelta(minutes=2)


class WorkerOut(BaseModel):
    id: uuid.UUID
    name: str
    pool: str
    status: str
    online: bool
    running_jobs: int
    caps: dict[str, Any]
    last_seen_at: datetime.datetime | None
    created_at: datetime.datetime


class WorkerIn(BaseModel):
    name: str = Field(min_length=1, max_length=100, pattern=r"^[A-Za-z0-9_.-]+$")
    pool: Literal["remote", "burst"] = "remote"

    @field_validator("name")
    @classmethod
    def _not_local(cls, name: str) -> str:
        if name.lower() == LOCAL_WORKER_NAME:
            raise ValueError(f"{LOCAL_WORKER_NAME!r} is the local workers' name")
        return name


class NewWorkerOut(BaseModel):
    worker: WorkerOut
    # Shown once. Give it to the worker with --token-file.
    token: str


def _worker_out(worker: Worker, running: int) -> WorkerOut:
    seen = worker.last_seen_at
    return WorkerOut(
        id=worker.id,
        name=worker.name,
        pool=worker.pool,
        status=worker.status,
        online=seen is not None
        and seen > datetime.datetime.now(datetime.UTC) - ONLINE_WINDOW,
        running_jobs=running,
        caps=worker.caps,
        last_seen_at=seen,
        created_at=worker.created_at,
    )


@router.get("/workers")
async def list_workers(auth: AdminAuth, db: DbSession) -> list[WorkerOut]:
    running = dict(
        (
            await db.execute(
                select(Job.lease_worker_id, func.count())
                .where(Job.status == "leased")
                .group_by(Job.lease_worker_id)
            )
        ).all()
    )
    workers = (await db.scalars(select(Worker).order_by(Worker.name))).all()
    return [_worker_out(w, running.get(w.id, 0)) for w in workers]


@router.post("/workers", status_code=201)
async def create_worker(
    body: WorkerIn, request: Request, auth: AdminAuth, db: DbSession
) -> NewWorkerOut:
    token = new_worker_token()
    worker = Worker(
        name=body.name,
        pool=body.pool,
        token_hash=token_hash(token),
        created_by=auth.user.id,
        caps={},
    )
    db.add(worker)
    try:
        await db.flush()
    except IntegrityError:
        raise HTTPException(
            status_code=409, detail="A worker with that name exists."
        ) from None
    audit.record(
        db,
        actor_id=auth.user.id,
        action="worker.create",
        target_type="worker",
        target_id=worker.id,
        request=request,
        details={"name": worker.name, "pool": worker.pool},
    )
    await db.commit()
    await db.refresh(worker)
    return NewWorkerOut(worker=_worker_out(worker, 0), token=token)


@router.delete("/workers/{worker_id}", status_code=204)
async def revoke_worker(
    worker_id: uuid.UUID, request: Request, auth: AdminAuth, db: DbSession
) -> None:
    """
    Revoke a worker's token. Jobs it was running go back in the queue.
    """
    worker = await db.get(Worker, worker_id)
    if worker is None:
        raise HTTPException(status_code=404, detail="No such worker.")
    # Jobs before the worker row: the queue's lock order.
    await jobs.requeue_worker_jobs(db, worker.id)
    worker.status = "revoked"
    audit.record(
        db,
        actor_id=auth.user.id,
        action="worker.revoke",
        target_type="worker",
        target_id=worker.id,
        request=request,
        details={"name": worker.name},
    )
    await db.commit()


class JobOut(BaseModel):
    id: uuid.UUID
    root_id: uuid.UUID
    project_id: uuid.UUID | None
    kind: str
    tier: int
    status: str
    progress: float
    message: str | None
    attempts: int
    max_attempts: int
    error: str | None
    result: dict[str, Any] | None
    worker: str | None
    created_at: datetime.datetime
    started_at: datetime.datetime | None
    finished_at: datetime.datetime | None


def _job_out(job: Job, worker_name: str | None) -> JobOut:
    return JobOut(
        id=job.id,
        root_id=job.root_id,
        project_id=job.project_id,
        kind=job.kind,
        tier=job.tier,
        status=job.status,
        progress=job.progress,
        message=job.message,
        attempts=job.attempts,
        max_attempts=job.max_attempts,
        error=job.error,
        result=job.result,
        worker=worker_name,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )


@router.get("/jobs")
async def list_jobs(
    auth: AdminAuth,
    db: DbSession,
    status: JobStatus | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[JobOut]:
    query = (
        select(Job, Worker.name)
        .outerjoin(Worker, Worker.id == Job.lease_worker_id)
        .order_by(Job.created_at.desc())
        .limit(limit)
    )
    if status is not None:
        query = query.where(Job.status == status)
    return [_job_out(job, name) for job, name in (await db.execute(query)).all()]


@router.get("/jobs/{job_id}")
async def get_job(job_id: uuid.UUID, auth: AdminAuth, db: DbSession) -> JobOut:
    row = (
        await db.execute(
            select(Job, Worker.name)
            .outerjoin(Worker, Worker.id == Job.lease_worker_id)
            .where(Job.id == job_id)
        )
    ).first()
    if row is None:
        raise HTTPException(status_code=404, detail="No such job.")
    return _job_out(*row)


@router.post("/jobs/{job_id}/cancel", status_code=204)
async def cancel_job(
    job_id: uuid.UUID, request: Request, auth: AdminAuth, db: DbSession
) -> None:
    """
    Cancel the pipeline the job belongs to.
    """
    job = await db.get(Job, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="No such job.")
    await jobs.cancel_pipeline(db, job.root_id)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="job.cancel",
        target_type="job",
        target_id=job.root_id,
        request=request,
    )
    await db.commit()


class NoopIn(BaseModel):
    # How long the job runs, and whether it fails at the end (to try retries).
    seconds: float = Field(default=1, ge=0, le=600)
    fail: bool = False


@router.post("/jobs/noop", status_code=201)
async def enqueue_noop(body: NoopIn, auth: AdminAuth, db: DbSession) -> JobOut:
    """
    Queue a diagnostic job that any worker can run, to check that workers are
    picking up work.
    """
    job = await jobs.enqueue(
        db,
        "noop",
        body.model_dump(),
        created_by=auth.user.id,
        tier=Tier.INTERACTIVE,
        max_attempts=1 if body.fail else 3,
    )
    await db.commit()
    await db.refresh(job)
    return _job_out(job, None)
