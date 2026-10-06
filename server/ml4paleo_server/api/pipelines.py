"""
A project's pipelines and image.

    POST /api/projects/{id}/ingest                       start ingesting an upload
    GET  /api/projects/{id}/pipelines                    recent pipelines
    GET  /api/projects/{id}/pipelines/{pipeline}         one pipeline's status
    GET  /api/projects/{id}/pipelines/{pipeline}/events  status updates (SSE)
    POST /api/projects/{id}/pipelines/{pipeline}/cancel  cancel it
    GET  /api/projects/{id}/image                        the current image

A pipeline's id is the id of its first job.
"""

import asyncio
import datetime
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select

from .. import artifacts, audit, jobs, pipelines, streams
from ..auth.deps import CurrentAuth, DbSession, SettingsDep
from ..db import Job, Project, ProjectMember, Upload, UserSession
from ..jobs.queue import FINISHED
from ..viewer import neuroglancer_available, neuroglancer_link
from .gateway import zarr_path
from .projects import MemberProject

router = APIRouter(prefix="/api/projects/{project_id}", tags=["pipelines"])

# Status updates are sent at most this often, and a stream ends after
# STREAM_FOR (browsers reconnect by themselves).
EVENT_INTERVAL_SECONDS = 1.0
STREAM_FOR = datetime.timedelta(minutes=30)


class IngestIn(BaseModel):
    upload_id: uuid.UUID


class PipelineOut(BaseModel):
    id: uuid.UUID
    kind: str
    # "waiting", "running", "succeeded", "failed", or "cancelled".
    status: str
    progress: float
    jobs: int
    created_at: datetime.datetime
    # Why it failed, in a sentence.
    error: str | None = None
    # The model a training or prediction is for.
    model_id: uuid.UUID | None = None


async def _pipeline_out(db, root: Job) -> PipelineOut:
    status = await jobs.pipeline_status(db, root.id)
    error = None
    if status.status == "failed":
        failed = await db.scalar(
            select(Job.error)
            .where(Job.root_id == root.id, Job.status == "failed")
            .order_by(Job.finished_at)
            .limit(1)
        )
        error = (failed or "").strip().splitlines()[0][:500] if failed else None
    return PipelineOut(
        id=root.id,
        kind=pipelines.NAMES.get(root.kind, root.kind),
        status=status.status,
        progress=round(status.progress, 4),
        jobs=status.jobs,
        created_at=root.created_at,
        error=error,
        model_id=root.payload.get("model_id"),
    )


async def _root(db, project, pipeline_id: uuid.UUID) -> Job:
    root = await db.scalar(
        select(Job).where(
            Job.id == pipeline_id,
            Job.root_id == pipeline_id,
            Job.project_id == project.id,
        )
    )
    if root is None:
        raise HTTPException(status_code=404, detail="No such pipeline.")
    return root


@router.post("/ingest", status_code=202)
async def start_ingest(
    body: IngestIn,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> PipelineOut:
    """
    Turn a finished upload into the project's image.
    """
    upload = await db.scalar(
        select(Upload).where(
            Upload.id == body.upload_id, Upload.project_id == project.id
        )
    )
    if upload is None or upload.state != "complete":
        raise HTTPException(status_code=404, detail="No finished upload with that id.")
    root, artifact = await pipelines.ingest.start(db, upload, auth.user.id)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="ingest.start",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"upload_id": str(upload.id), "pipeline_id": str(root.id)},
    )
    await db.commit()
    await db.refresh(root)
    return await _pipeline_out(db, root)


@router.get("/pipelines")
async def list_pipelines(project: MemberProject, db: DbSession) -> list[PipelineOut]:
    roots = (
        await db.scalars(
            select(Job)
            .where(Job.project_id == project.id, Job.id == Job.root_id)
            .order_by(Job.created_at.desc())
            .limit(50)
        )
    ).all()
    return [await _pipeline_out(db, root) for root in roots]


@router.get("/pipelines/{pipeline_id}")
async def get_pipeline(
    pipeline_id: uuid.UUID, project: MemberProject, db: DbSession
) -> PipelineOut:
    return await _pipeline_out(db, await _root(db, project, pipeline_id))


@router.get("/pipelines/{pipeline_id}/events", response_model=None)
async def pipeline_events(
    pipeline_id: uuid.UUID,
    project: MemberProject,
    auth: CurrentAuth,
    db: DbSession,
    request: Request,
) -> StreamingResponse | Response:
    """
    Server-sent events: a `status` event (a pipeline, as JSON) whenever the
    pipeline changes, ending once it has finished. For a pipeline that has
    already finished the answer is 204, which tells a browser's EventSource
    not to reconnect (fetch the pipeline instead).
    """
    root = await _root(db, project, pipeline_id)
    if (await _pipeline_out(db, root)).status in FINISHED:
        return Response(status_code=204)
    # Read what the stream needs before ending the transaction, which
    # expires loaded objects (the job is detached, so it keeps its values).
    user_id, session_hash, project_id = (
        auth.user.id,
        auth.session.token_hash,
        project.id,
    )
    db.expunge(root)
    # Give the request's connection back; each update uses a short session.
    await db.rollback()
    sessionmaker = request.app.state.sessionmaker

    async def still_allowed(session) -> bool:
        """
        The person is still signed in and still a member.
        """
        allowed = await session.scalar(
            select(ProjectMember.user_id)
            .join(Project, Project.id == ProjectMember.project_id)
            .where(
                ProjectMember.project_id == project_id,
                ProjectMember.user_id == user_id,
                Project.deleted_at.is_(None),
            )
        )
        signed_in = await session.scalar(
            select(UserSession.token_hash).where(
                UserSession.token_hash == session_hash,
                UserSession.expires_at > datetime.datetime.now(datetime.UTC),
            )
        )
        return allowed is not None and signed_in is not None

    async def events():
        deadline = datetime.datetime.now(datetime.UTC) + STREAM_FOR
        last = None
        while datetime.datetime.now(datetime.UTC) < deadline:
            async with sessionmaker() as session:
                if not await still_allowed(session):
                    return
                current = await _pipeline_out(session, root)
            payload = current.model_dump_json()
            if payload != last:
                last = payload
                yield f"event: status\ndata: {payload}\n\n"
            if current.status in FINISHED or await request.is_disconnected():
                return
            await asyncio.sleep(EVENT_INTERVAL_SECONDS)

    return streams.response(streams.reserve(user_id), events())


@router.post("/pipelines/{pipeline_id}/cancel", status_code=204)
async def cancel_pipeline(
    pipeline_id: uuid.UUID,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> None:
    root = await _root(db, project, pipeline_id)
    await jobs.cancel_pipeline(db, root.id)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="pipeline.cancel",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"pipeline_id": str(root.id)},
    )
    await db.commit()


class ImageOut(BaseModel):
    artifact_id: uuid.UUID
    committed_at: datetime.datetime
    # The manifest: shape_czyx, dtype, levels, voxel_size_zyx, unit, window,
    # histogram, and source.
    manifest: dict[str, Any]
    # The OME-Zarr image, through the data gateway.
    zarr_url: str
    # The image in Neuroglancer, if this server has it.
    neuroglancer_url: str | None


@router.get("/image")
async def current_image(
    project: MemberProject, db: DbSession, settings: SettingsDep
) -> ImageOut:
    image = await artifacts.head(db, project.id, "image")
    if image is None:
        raise HTTPException(status_code=404, detail="This project has no image yet.")
    zarr_url = zarr_path(project.id, image.id)
    manifest = image.manifest or {}
    return ImageOut(
        artifact_id=image.id,
        committed_at=image.state_changed_at,
        manifest=manifest,
        zarr_url=zarr_url,
        neuroglancer_url=neuroglancer_link(settings.public_url, zarr_url, manifest)
        if neuroglancer_available(settings)
        else None,
    )
