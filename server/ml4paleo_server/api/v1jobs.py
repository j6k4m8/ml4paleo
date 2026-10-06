"""
Claiming jobs from the ml4paleo v1 app, which this one replaced (see
`pipelines/v1import.py`).

    POST /api/v1-jobs/{job}/claim     import a v1 job into a new project of yours
    POST /api/v1-jobs/{job}/release   (admins) let a wrongly claimed job go

v1 had no accounts: anyone with a job's link could open it. So the first
person to claim a job gets it, claims are rate-limited (ids are only six hex
digits) per account, per address, and for everyone once too many miss, and
an admin can release a job someone else claimed. Releasing stops what runs
in the claimer's project and deletes it (garbage collection gives its
storage back), so the job's owner can claim it again; the account it was
released from can't.
"""

import datetime
import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from sqlalchemy import func, select
from starlette.concurrency import run_in_threadpool

from ml4paleo.v1import import UNCONVERTED, normalize_job_id, read_jobs, status

from .. import artifacts, audit, jobs
from ..auth import ratelimit
from ..auth.deps import AdminAuth, CurrentAuth, DbSession, EngineDep, SettingsDep
from ..auth.ratelimit import client_key
from ..db import AuditEvent, Job, Project, ProjectMember
from ..jobs.queue import FINISHED
from ..pipelines import v1import
from ..settings import Settings

router = APIRouter(prefix="/api/v1-jobs", tags=["v1"])

HOUR = datetime.timedelta(hours=1)
# Claims of ids that aren't v1 jobs, from anyone.
MISSES = "v1-claim:misses"
NOT_HERE = "This server has no v1 jobs to import."
RELEASED = "An admin released this job from your account. If it's yours, ask them."


class ClaimOut(BaseModel):
    project_id: uuid.UUID
    # The import's pipeline; None if you had already claimed the job (and its
    # import hadn't failed).
    pipeline_id: uuid.UUID | None


def _volume(settings: Settings):
    if settings.v1.volume_path is None:
        raise HTTPException(status_code=404, detail=NOT_HERE)
    return settings.v1.volume_path


def _job_id(text: str) -> str:
    job_id = normalize_job_id(text)
    if job_id is None:
        raise HTTPException(status_code=404, detail="There's no v1 job with that id.")
    return job_id


async def _lock(db: DbSession, job_id: str) -> None:
    await db.execute(
        select(func.pg_advisory_xact_lock(func.hashtextextended(f"m4p_v1:{job_id}", 0)))
    )


async def _released(db: DbSession, job_id: str, user_id: uuid.UUID) -> bool:
    """Whether an admin released the job from a project of this user's."""
    release = await db.scalar(
        select(AuditEvent.id)
        .where(
            AuditEvent.action == "v1.release",
            AuditEvent.details.contains(
                {"v1_job_id": job_id, "owner_id": str(user_id)}
            ),
        )
        .limit(1)
    )
    return release is not None


async def _import_failed(db: DbSession, project: Project) -> bool:
    """Whether the project's import ended without an image, and nothing runs."""
    if await artifacts.head(db, project.id, "image") is not None:
        return False
    running = await db.scalar(
        select(Job.id)
        .where(Job.project_id == project.id, Job.status.not_in(FINISHED))
        .limit(1)
    )
    return running is None


@router.post("/{job_id}/claim", status_code=201)
async def claim(
    job_id: str,
    request: Request,
    response: Response,
    auth: CurrentAuth,
    db: DbSession,
    engine: EngineDep,
    settings: SettingsDep,
) -> ClaimOut:
    """
    Import a v1 job into a new project of yours. Claiming a job you already
    claimed gives that project, and starts its import again if it failed; a
    job someone else claimed gets 409.
    """
    root = _volume(settings)
    # Per account and per address, since one address can sign up a few
    # accounts an hour.
    for key in (f"v1-claim:user:{auth.user.id}", f"v1-claim:ip:{client_key(request)}"):
        await ratelimit.hit(engine, key, limit=settings.v1.claims_per_hour, window=HOUR)
    # Guessing ids misses far more often than claiming your own jobs does, so
    # once too many claims miss, nobody can claim until the hour is up.
    misses = settings.v1.failed_claims_per_hour
    await ratelimit.peek(engine, MISSES, limit=misses, window=HOUR)
    job_id = _job_id(job_id)
    record = (await run_in_threadpool(read_jobs, root)).get(job_id)
    if record is None:
        audit.record(
            db,
            actor_id=auth.user.id,
            action="v1.claim.miss",
            target_type="v1_job",
            target_id=job_id,
            request=request,
        )
        await db.commit()
        await ratelimit.hit(engine, MISSES, limit=misses, window=HOUR)
        raise HTTPException(status_code=404, detail="There's no v1 job with that id.")
    if status(record) in UNCONVERTED:
        raise HTTPException(
            status_code=409,
            detail="This v1 job never finished converting its upload, so there's "
            "no scan to import.",
        )
    # The first claim wins.
    await _lock(db, job_id)
    if await _released(db, job_id, auth.user.id):
        raise HTTPException(status_code=409, detail=RELEASED)
    existing = await db.scalar(select(Project).where(Project.v1_job_id == job_id))
    if existing is not None and existing.deleted_at is None:
        if existing.owner_id != auth.user.id:
            raise HTTPException(
                status_code=409,
                detail="Someone has already imported this job. If it's yours, ask "
                "an admin to release it.",
            )
        response.status_code = 200
        if not await _import_failed(db, existing):
            return ClaimOut(project_id=existing.id, pipeline_id=None)
        # For example, it didn't fit in your storage then: try again.
        project = existing
    else:
        if existing is not None:
            # A deleted project lets go of its job.
            existing.v1_job_id = None
            await db.flush()
        name = str(record.get("name") or "").strip() or f"v1 job {job_id}"
        project = Project(name=name[:100], owner_id=auth.user.id, v1_job_id=job_id)
        db.add(project)
        await db.flush()
        db.add(ProjectMember(project_id=project.id, user_id=auth.user.id))
    probe, _ = await v1import.start(db, project, job_id, auth.user.id)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="v1.claim",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"v1_job_id": job_id, "pipeline_id": str(probe.id)},
    )
    await db.commit()
    return ClaimOut(project_id=project.id, pipeline_id=probe.id)


@router.post("/{job_id}/release", status_code=204)
async def release(
    job_id: str,
    request: Request,
    auth: AdminAuth,
    db: DbSession,
) -> None:
    """
    Stop and delete the project that claimed a v1 job, so the job can be
    claimed again (by anyone but that project's owner).
    """
    job_id = _job_id(job_id)
    await _lock(db, job_id)
    project = await db.scalar(select(Project).where(Project.v1_job_id == job_id))
    if project is None:
        raise HTTPException(status_code=404, detail="Nobody has claimed that job.")
    # Pipelines before the project: completing a job locks the job, then its
    # project.
    running = await db.scalars(
        select(Job.root_id)
        .where(Job.project_id == project.id, Job.status.not_in(FINISHED))
        .distinct()
        .order_by(Job.root_id)
    )
    for root_id in running.all():
        await jobs.cancel_pipeline(db, root_id)
    project.v1_job_id = None
    if project.deleted_at is None:
        project.deleted_at = datetime.datetime.now(datetime.UTC)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="v1.release",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"v1_job_id": job_id, "owner_id": str(project.owner_id)},
    )
    await db.commit()
