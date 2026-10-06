"""
Claiming jobs from the ml4paleo v1 app, which this one replaced (see
`pipelines/v1import.py`).

    POST /api/v1-jobs/{job}/claim     import a v1 job into a new project of yours
    POST /api/v1-jobs/{job}/release   (admins) give a job to an account {to}

v1 had no accounts: anyone with a job's link could open it. So the first
person to claim a job gets it; claims are rate-limited, since ids are only
six hex digits (each account and address may try a few an hour, and fewer
that miss, and once too many miss from anyone, nobody can claim until the
hour is up); and an admin can give a job to the account it belongs to. That
stops what runs in the project someone else made from it and deletes it
(garbage collection gives its storage back), and then only that account can
claim the job; giving it again undoes a mistake.

Claims and releases are recorded in the audit log against the job (target
"v1_job", the job id), where a claim finds the last of them.
"""

import datetime
import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.concurrency import run_in_threadpool

from ml4paleo.v1import import UNCONVERTED, normalize_job_id, read_jobs, status

from .. import audit, jobs
from ..auth import ratelimit
from ..auth.deps import AdminAuth, CurrentAuth, DbSession, EngineDep, SettingsDep
from ..auth.ratelimit import client_key
from ..db import AuditEvent, Project, ProjectMember, User
from ..pipelines import train, v1import
from ..settings import Settings

router = APIRouter(prefix="/api/v1-jobs", tags=["v1"])

HOUR = datetime.timedelta(hours=1)
# Claims of ids that aren't v1 jobs, from anyone.
SITE_MISSES = "v1-miss:site"
NOT_HERE = "This server has no v1 jobs to import."
GIVEN = "An admin gave this job to another account. If it's yours, ask an admin."


class ReleaseIn(BaseModel):
    # The username of the account the job belongs to.
    to: str = Field(min_length=1, max_length=64)


class ClaimOut(BaseModel):
    project_id: uuid.UUID
    # The pipelines the claim started: the import, or the parts of it that an
    # earlier try didn't finish. Empty when nothing needed starting.
    pipeline_ids: list[uuid.UUID]


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


async def _count_miss(
    engine: AsyncEngine, limits: list[tuple[str, int]]
) -> list[tuple[str, datetime.datetime]]:
    """
    Count a miss against each key, in order, refusing (with nothing counted)
    once one is at its limit. Returns what to give back if it's no miss.
    """
    counted: list[tuple[str, datetime.datetime]] = []
    try:
        for key, limit in limits:
            counted.append(
                (key, await ratelimit.take(engine, key, limit=limit, window=HOUR))
            )
    except BaseException:
        await _give_back(engine, counted)
        raise
    return counted


async def _give_back(
    engine: AsyncEngine, counted: list[tuple[str, datetime.datetime]]
) -> None:
    for key, started in counted:
        await ratelimit.give_back(engine, key, started)


async def _given_to(db: DbSession, job_id: str) -> uuid.UUID | None:
    """The account an admin last gave the job to, unless it's claimed since."""
    last = await db.scalar(
        select(AuditEvent)
        .where(
            AuditEvent.target_type == "v1_job",
            AuditEvent.target_id == job_id,
            AuditEvent.action.in_(("v1.claim", "v1.release")),
        )
        .order_by(AuditEvent.id.desc())
        .limit(1)
    )
    if last is None or last.action != "v1.release":
        return None
    return uuid.UUID(last.details["to_user_id"])


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
    claimed gives that project, and starts again whatever parts of its import
    failed; a job someone else claimed gets 409.
    """
    root = _volume(settings)
    job_id = _job_id(job_id)
    limits = settings.v1
    # Per account and per address, since one address can sign up a few
    # accounts an hour.
    keys = (f"user:{auth.user.id}", f"ip:{client_key(request)}")
    # Count a miss before looking, and give it back if the job is there, so
    # guesses made at once can't get past the limits. Guessing misses nearly
    # every time, so once too many miss, from anyone, nobody can claim until
    # the hour is up; that comes first, and then each one's own misses, so a
    # claim refused for either counts against nothing else.
    misses = await _count_miss(
        engine,
        [(SITE_MISSES, limits.site_misses_per_hour)]
        + [(f"v1-miss:{key}", limits.misses_per_hour) for key in keys],
    )
    try:
        for key in keys:
            await ratelimit.hit(
                engine, f"v1-claim:{key}", limit=limits.claims_per_hour, window=HOUR
            )
        record = (await run_in_threadpool(read_jobs, root)).get(job_id)
    except BaseException:
        await _give_back(engine, misses)
        raise
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
        raise HTTPException(status_code=404, detail="There's no v1 job with that id.")
    await _give_back(engine, misses)
    if status(record) in UNCONVERTED:
        raise HTTPException(
            status_code=409,
            detail="This v1 job never finished converting its upload, so there's "
            "no scan to import.",
        )
    # The first claim wins.
    await _lock(db, job_id)
    existing = await db.scalar(select(Project).where(Project.v1_job_id == job_id))
    if existing is not None and existing.deleted_at is None:
        if existing.owner_id != auth.user.id:
            raise HTTPException(
                status_code=409,
                detail="Someone has already imported this job. If it's yours, ask "
                "an admin to give it to you.",
            )
        response.status_code = 200
        # For example, a prediction that didn't fit in your storage then.
        try:
            started = await v1import.resume(db, settings, existing, auth.user.id)
        except jobs.Rejected as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None
        if started:
            audit.record(
                db,
                actor_id=auth.user.id,
                action="v1.resume",
                target_type="v1_job",
                target_id=job_id,
                request=request,
                details={
                    "project_id": str(existing.id),
                    "pipeline_ids": [str(job.id) for job in started],
                },
            )
        await db.commit()
        return ClaimOut(
            project_id=existing.id, pipeline_ids=[job.id for job in started]
        )
    given = await _given_to(db, job_id)
    if given is not None and given != auth.user.id:
        raise HTTPException(status_code=409, detail=GIVEN)
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
        target_type="v1_job",
        target_id=job_id,
        request=request,
        details={"project_id": str(project.id), "pipeline_ids": [str(probe.id)]},
    )
    await db.commit()
    return ClaimOut(project_id=project.id, pipeline_ids=[probe.id])


@router.post("/{job_id}/release", status_code=204)
async def release(
    job_id: str,
    body: ReleaseIn,
    request: Request,
    auth: AdminAuth,
    db: DbSession,
    settings: SettingsDep,
) -> None:
    """
    Give a v1 job to the account it belongs to: only that account can claim
    it next. A project someone else made from it is stopped and deleted.
    """
    root = _volume(settings)
    job_id = _job_id(job_id)
    if job_id not in await run_in_threadpool(read_jobs, root):
        raise HTTPException(status_code=404, detail="There's no v1 job with that id.")
    to = await db.scalar(
        select(User).where(
            User.username == body.to.strip().lower(), User.status != "disabled"
        )
    )
    if to is None:
        raise HTTPException(status_code=404, detail="No one with that username.")
    await _lock(db, job_id)
    project = await db.scalar(
        select(Project).where(Project.v1_job_id == job_id, Project.deleted_at.is_(None))
    )
    if project is not None:
        if project.owner_id == to.id:
            raise HTTPException(status_code=409, detail="That account has the job.")
        # As deleting a project does, with its pipelines before the project
        # (completing a job locks the job, then its project).
        await train.stop_project(db, project)
        # Leave v1_job_id (a key) for the next claim to clear: changing a key
        # here would wait for jobs that are adding rows to the project, which
        # can be waiting for the jobs this just locked.
        project.deleted_at = datetime.datetime.now(datetime.UTC)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="v1.release",
        target_type="v1_job",
        target_id=job_id,
        request=request,
        details={
            "to_user_id": str(to.id),
            "project_id": str(project.id) if project else None,
            "owner_id": str(project.owner_id) if project else None,
        },
    )
    await db.commit()
