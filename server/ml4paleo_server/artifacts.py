"""
Artifacts: what jobs make, and which one is current.

A pipeline creates its output artifacts in `staging` and gives its jobs
read-write grants to them. Jobs write their files, and the job that produces
an artifact writes `_MANIFEST.json` last. When that job reports success,
`commit_outputs` runs inside the same transaction: it checks for the
manifest, measures the files (the server's own measurement, which is what
counts against the quota), reserves that much of the project owner's storage,
marks the artifact committed, and moves its head slot to it. If any of that
fails, the job is not marked succeeded. Partial output therefore never
becomes current.

Garbage collection (run by the housekeeper) is the only thing that deletes
artifact files: failed artifacts after `keep_failed_hours`, replaced ones
after `keep_superseded_days`, expired ones (caches such as exports), and
everything in deleted projects. It skips artifacts that a waiting or running
job may still read.
"""

import datetime
import json
import logging
import uuid
from typing import Any

import obstore
from fastapi import HTTPException
from sqlalchemy import and_, delete, exists, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ml4paleo.storage import MANIFEST_KEY, object_store

from . import quotas
from .db import Artifact, ArtifactHead, Job, Project, User
from .jobs.queue import Rejected
from .settings import Settings
from .storage import project_storage

log = logging.getLogger(__name__)

MAX_MANIFEST_BYTES = 1024 * 1024
# Garbage collection deletes at most this many artifacts per pass.
COLLECT_BATCH = 50
# Staging artifacts that no job will commit are abandoned after this long.
ABANDONED_AFTER = datetime.timedelta(hours=48)


def now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def project_path(project_id: uuid.UUID) -> str:
    return f"projects/{project_id}"


def artifact_path(artifact: Artifact) -> str:
    return f"{project_path(artifact.project_id)}/artifacts/{artifact.id}"


def grant_for(artifact: Artifact, access: str = "rw") -> dict[str, str]:
    """
    A job grant (see `Job.grants`) for an artifact's files.
    """
    return {"path": artifact_path(artifact), "access": access}


async def create_staging(
    db: AsyncSession,
    *,
    project_id: uuid.UUID,
    kind: str,
    inputs: dict[str, Any] | None = None,
    head_slot: str | None = None,
    cache_key: str | None = None,
    expires_at: datetime.datetime | None = None,
) -> Artifact:
    """
    Create an artifact for jobs to fill. Set `produced_by_job` to the job
    whose success should commit it.

    Head artifacts are kept until replaced, so they can't also expire.
    """
    if head_slot is not None and expires_at is not None:
        raise ValueError("A head artifact can't have an expiry")
    artifact = Artifact(
        project_id=project_id,
        kind=kind,
        state="staging",
        inputs=inputs or {},
        head_slot=head_slot,
        cache_key=cache_key,
        expires_at=expires_at,
    )
    db.add(artifact)
    await db.flush()
    return artifact


async def commit_outputs(db: AsyncSession, settings: Settings, job: Job) -> None:
    """
    Commit the artifacts `job` produces. Called by the queue just before the
    job is marked succeeded; raises `Rejected` to stop that.
    """
    staged = (
        await db.scalars(
            select(Artifact)
            .where(Artifact.produced_by_job == job.id, Artifact.state == "staging")
            .order_by(Artifact.id)
            .with_for_update(key_share=True)
            .execution_options(populate_existing=True)
        )
    ).all()
    for artifact in staged:
        await _commit(db, settings, artifact)


async def _commit(db: AsyncSession, settings: Settings, artifact: Artifact) -> None:
    project = await db.get(Project, artifact.project_id)
    if project is None or project.deleted_at is not None:
        raise Rejected("The project was deleted.", retryable=False)
    store = object_store(project_storage(settings).child(artifact_path(artifact)))
    manifest = await _read_manifest(store)
    if manifest is None:
        raise Rejected(
            f"The job finished without writing {MANIFEST_KEY} for artifact "
            f"{artifact.id}.",
            retryable=True,
        )
    size = await _total_size(store)
    owner = await db.get(User, project.owner_id)
    assert owner is not None
    try:
        await quotas.reserve_storage(db, settings, owner, size)
    except HTTPException as exc:
        raise Rejected(
            f"{exc.detail}: the results take {size} bytes.", retryable=False
        ) from None
    artifact.state = "committed"
    artifact.bytes = size
    artifact.manifest = manifest
    artifact.state_changed_at = now()
    if artifact.head_slot is not None:
        await set_head(db, artifact)


async def _read_manifest(store) -> dict[str, Any] | None:
    try:
        result = await obstore.get_async(store, MANIFEST_KEY)
    except FileNotFoundError:
        return None
    if result.meta["size"] > MAX_MANIFEST_BYTES:
        raise Rejected(f"{MANIFEST_KEY} is too large.", retryable=False)
    try:
        manifest = json.loads(bytes(await result.bytes_async()))
    except ValueError:
        manifest = None
    if not isinstance(manifest, dict):
        raise Rejected(f"{MANIFEST_KEY} is not a JSON object.", retryable=False)
    return manifest


async def _total_size(store) -> int:
    total = 0
    async for batch in obstore.list(store):
        total += sum(meta["size"] for meta in batch)
    return total


async def set_head(db: AsyncSession, artifact: Artifact) -> None:
    """
    Make `artifact` the current one in its slot; the one it replaces becomes
    superseded.
    """
    assert artifact.head_slot is not None and artifact.state == "committed"
    await db.execute(
        insert(ArtifactHead)
        .values(
            project_id=artifact.project_id,
            slot=artifact.head_slot,
            artifact_id=artifact.id,
        )
        .on_conflict_do_nothing()
    )
    head = await db.scalar(
        select(ArtifactHead)
        .where(
            ArtifactHead.project_id == artifact.project_id,
            ArtifactHead.slot == artifact.head_slot,
        )
        .with_for_update(key_share=True)
        .execution_options(populate_existing=True)
    )
    assert head is not None
    if head.artifact_id == artifact.id:
        return
    previous = head.artifact_id
    head.artifact_id = artifact.id
    head.updated_at = now()
    await db.flush()
    await db.execute(
        update(Artifact)
        .where(Artifact.id == previous, Artifact.state == "committed")
        .values(state="superseded", state_changed_at=func.now())
        .execution_options(synchronize_session=False)
    )


async def head(db: AsyncSession, project_id: uuid.UUID, slot: str) -> Artifact | None:
    return await db.scalar(
        select(Artifact)
        .join(ArtifactHead, ArtifactHead.artifact_id == Artifact.id)
        .where(ArtifactHead.project_id == project_id, ArtifactHead.slot == slot)
    )


async def abandon_staging(db: AsyncSession) -> int:
    """
    Mark failed the staging artifacts that nothing will commit: their job
    finished without committing them, or they have no job, are old, and no
    waiting or running job has a grant for them (a pipeline's first job may
    wait a long time before the job that commits it exists).
    """
    result = await db.execute(
        update(Artifact)
        .where(
            Artifact.state == "staging",
            or_(
                exists().where(
                    Job.id == Artifact.produced_by_job,
                    Job.status.in_(("succeeded", "failed", "cancelled")),
                ),
                and_(
                    Artifact.produced_by_job.is_(None),
                    Artifact.created_at < now() - ABANDONED_AFTER,
                    ~_in_use(),
                ),
            ),
        )
        .values(state="failed", state_changed_at=func.now())
        .returning(Artifact.id)
        .execution_options(synchronize_session=False)
    )
    return len(result.all())


def _in_use():
    """
    SQL: a waiting or running job has a grant for the artifact's files.
    """
    path = func.concat("projects/", Artifact.project_id, "/artifacts/", Artifact.id)
    return exists().where(
        Job.status.in_(("blocked", "queued", "leased")),
        Job.grants.contains(
            func.jsonb_build_array(func.jsonb_build_object("path", path))
        ),
    )


def _collectable(settings: Settings):
    """
    SQL: artifacts whose files garbage collection may delete now. A current
    head is never collected unless its project was deleted.
    """
    current = now()
    storage = settings.storage
    in_use = _in_use()
    deleted_project = exists().where(
        Project.id == Artifact.project_id, Project.deleted_at.is_not(None)
    )
    is_head = exists().where(ArtifactHead.artifact_id == Artifact.id)
    return and_(
        Artifact.state.not_in(("deleting", "deleted")),
        ~in_use,
        or_(
            and_(Artifact.state != "staging", deleted_project),
            and_(
                ~is_head,
                or_(
                    and_(
                        Artifact.state == "failed",
                        Artifact.state_changed_at
                        < current - datetime.timedelta(hours=storage.keep_failed_hours),
                    ),
                    and_(
                        Artifact.state == "superseded",
                        Artifact.state_changed_at
                        < current
                        - datetime.timedelta(days=storage.keep_superseded_days),
                    ),
                    and_(Artifact.state == "committed", Artifact.expires_at < current),
                ),
            ),
        ),
    )


async def collect_garbage(
    sessionmaker: async_sessionmaker[AsyncSession], settings: Settings
) -> int:
    """
    Delete the files of artifacts that are no longer needed, and release the
    storage they counted against. Returns how many artifacts were deleted.

    Each artifact is handled on its own. First, under a lock, it is checked
    again and marked `deleting` (releasing its quota and any head), so no new
    job can be given it (`jobs.enqueue` refuses); then its files are deleted;
    then it is marked `deleted`. If deleting the files fails, the next pass
    tries again. One artifact's failure doesn't stop the others.
    """
    async with sessionmaker() as db:
        await abandon_staging(db)
        await db.commit()
        candidates = (
            await db.scalars(
                select(Artifact.id)
                .where(or_(Artifact.state == "deleting", _collectable(settings)))
                .order_by(Artifact.state_changed_at)
                .limit(COLLECT_BATCH)
            )
        ).all()
    deleted = 0
    for artifact_id in candidates:
        try:
            if await _collect(sessionmaker, settings, artifact_id):
                deleted += 1
        except Exception:
            log.exception("Could not delete artifact %s; will try again", artifact_id)
    return deleted


async def _collect(
    sessionmaker: async_sessionmaker[AsyncSession],
    settings: Settings,
    artifact_id: uuid.UUID,
) -> bool:
    async with sessionmaker() as db:
        artifact = await db.scalar(
            select(Artifact)
            .where(Artifact.id == artifact_id)
            .with_for_update(key_share=True)
            .execution_options(populate_existing=True)
        )
        if artifact is None or artifact.state == "deleted":
            return False
        if artifact.state != "deleting":
            # Check again in a new statement, which sees jobs committed while
            # this one waited for the lock.
            still = await db.scalar(
                select(Artifact.id).where(
                    Artifact.id == artifact_id, _collectable(settings)
                )
            )
            if still is None:
                return False
            await _start_deleting(db, artifact)
        await db.commit()
    await _delete_files(settings, artifact)
    async with sessionmaker() as db:
        await db.execute(
            update(Artifact)
            .where(Artifact.id == artifact_id, Artifact.state == "deleting")
            .values(state="deleted", state_changed_at=func.now())
        )
        await db.commit()
    log.info("Deleted artifact %s (%s)", artifact.id, artifact.kind)
    return True


async def _delete_files(settings: Settings, artifact: Artifact) -> None:
    store = object_store(project_storage(settings).child(artifact_path(artifact)))
    async for batch in obstore.list(store, chunk_size=1000):
        await obstore.delete_async(store, [meta["path"] for meta in batch])


async def _start_deleting(db: AsyncSession, artifact: Artifact) -> None:
    if artifact.state in ("committed", "superseded") and artifact.bytes:
        owner_id = await db.scalar(
            select(Project.owner_id).where(Project.id == artifact.project_id)
        )
        if owner_id is not None:
            await quotas.release_storage(db, owner_id, artifact.bytes)
    # Only artifacts of deleted projects can still be heads here.
    await db.execute(
        delete(ArtifactHead).where(ArtifactHead.artifact_id == artifact.id)
    )
    artifact.state = "deleting"
    artifact.state_changed_at = now()
