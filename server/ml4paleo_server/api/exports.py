"""
Exports: the project's image, prediction, final segmentation, or meshes as
one zip archive to download (see `pipelines/export.py`).

    POST   /api/projects/{id}/exports                     {source, format}
    GET    /api/projects/{id}/exports                     exports being made or kept
    GET    /api/projects/{id}/exports/{export}/download   the archive (byte ranges work)
    DELETE /api/projects/{id}/exports/{export}            stop it, or let it go now

Asking for an export that is kept, or still being made, gives that one.
"""

import datetime
import uuid
from typing import Literal

import obstore
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import distinct_on
from sqlalchemy.ext.asyncio import AsyncSession

from ml4paleo.export import part_key
from ml4paleo.storage import object_store

from .. import artifacts, audit, jobs, objects, quotas
from ..auth.deps import CurrentAuth, DbSession, SettingsDep
from ..db import Artifact, Job, User
from ..pipelines import export as export_pipeline
from ..storage import project_storage
from .projects import MemberProject

router = APIRouter(prefix="/api/projects/{project_id}/exports", tags=["exports"])

ACTIVE = ("blocked", "queued", "leased")
LISTED = ("staging", "committed", "failed")
# A download keeps its export at least this long, so it can't outlive it.
DOWNLOAD_GRACE = datetime.timedelta(hours=6)
# PNG holds 8- or 16-bit unsigned values (numpy's notation, as manifests use).
PNG_DTYPES = ("|u1", "<u2", ">u2")


class ExportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["image", "prediction", "segmentation", "meshes"]
    # Volumes: "zarr" (the zarr group), "tiff", or "png" (one image per
    # slice). Meshes: "zip".
    format: Literal["zarr", "tiff", "png", "zip"]


class ExportOut(BaseModel):
    id: uuid.UUID
    source: str
    # The artifact it was made from; not the current one if that changed.
    source_artifact_id: uuid.UUID | None
    format: str
    filename: str
    # "making", "ready", or "failed".
    status: str
    pipeline_id: uuid.UUID | None
    progress: float
    # Why it failed, in a sentence.
    error: str | None = None
    # The archive's size, once it's ready.
    bytes: int
    created_at: datetime.datetime
    # When it will be deleted.
    expires_at: datetime.datetime
    download_url: str | None = None


def _making(export: Artifact, job: Job | None) -> bool:
    """Still being made: not finished, and nobody asked to stop it."""
    return (
        export.state == "staging"
        and job is not None
        and job.status in ACTIVE
        and not job.cancel_requested
    )


async def _out(db: AsyncSession, export: Artifact) -> ExportOut:
    job = await db.get(Job, export.produced_by_job) if export.produced_by_job else None
    if export.state == "committed":
        status = "ready"
    elif _making(export, job):
        status = "making"
    else:
        status = "failed"
    error = None
    if status == "failed":
        if job is not None and job.cancel_requested:
            error = "Stopped."
        elif job is not None and job.error:
            error = job.error.strip().splitlines()[0][:500]
        error = error or "The export didn't finish."
    assert export.expires_at is not None
    return ExportOut(
        id=export.id,
        source=export.inputs.get("source", ""),
        source_artifact_id=export.inputs.get("source_artifact_id"),
        format=export.inputs.get("format", ""),
        filename=export.inputs.get("filename", "export.zip"),
        status=status,
        pipeline_id=job.root_id if job else None,
        progress=1.0 if status == "ready" else (job.progress if job else 0.0),
        error=error,
        bytes=int((export.manifest or {}).get("size", 0)),
        created_at=export.created_at,
        expires_at=export.expires_at,
        download_url=(
            f"/api/projects/{export.project_id}/exports/{export.id}/download"
            if status == "ready"
            else None
        ),
    )


@router.post("", status_code=202)
async def make_export(
    body: ExportIn,
    project: MemberProject,
    request: Request,
    response: Response,
    auth: CurrentAuth,
    db: DbSession,
    settings: SettingsDep,
) -> ExportOut:
    if body.format not in export_pipeline.FORMATS[body.source]:
        raise HTTPException(
            status_code=422,
            detail="Meshes export as zip; volumes as zarr, tiff, or png.",
        )
    source = await artifacts.head(db, project.id, body.source)
    if source is None or source.state != "committed":
        raise HTTPException(
            status_code=409, detail=f"This project has no {body.source} to export."
        )
    dtype = (source.manifest or {}).get("dtype", "|u1")
    if body.format == "png" and dtype not in PNG_DTYPES:
        raise HTTPException(
            status_code=422,
            detail=f"PNG holds 8- or 16-bit unsigned values, and this {body.source} "
            f"is {dtype}: export TIFF instead.",
        )
    # One request at a time per project, so two can't start the same export.
    await db.execute(
        select(
            func.pg_advisory_xact_lock(
                func.hashtextextended(f"m4p_exports:{project.id}", 0)
            )
        )
    )
    keep_until = artifacts.now() + export_pipeline.KEEP
    existing = await db.scalar(
        select(Artifact)
        .where(
            Artifact.project_id == project.id,
            Artifact.kind == "export",
            Artifact.cache_key == export_pipeline.cache_key(source, body.format),
            Artifact.state.in_(("staging", "committed")),
        )
        .order_by(Artifact.created_at.desc())
        .limit(1)
        # Garbage collection takes the same lock before deleting an export.
        .with_for_update()
    )
    if existing is not None:
        job = (
            await db.get(Job, existing.produced_by_job)
            if existing.produced_by_job
            else None
        )
        if existing.state == "committed" or _making(existing, job):
            existing.expires_at = max(existing.expires_at or keep_until, keep_until)
            await db.commit()
            if existing.state == "committed":
                response.status_code = 200
            return await _out(db, existing)
    await _check_room(db, settings, project.owner_id, source)
    _, export = await export_pipeline.start(
        db,
        source=source,
        slot=body.source,
        fmt=body.format,
        project_name=project.name,
        created_by=auth.user.id,
    )
    audit.record(
        db,
        actor_id=auth.user.id,
        action="export.make",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"export_id": str(export.id), **body.model_dump()},
    )
    await db.commit()
    await db.refresh(export)
    return await _out(db, export)


async def _check_room(
    db: AsyncSession, settings, owner_id: uuid.UUID, source: Artifact
) -> None:
    """
    Refuse an export that clearly won't fit in the owner's storage, before
    making it: it's about as big as its source.
    """
    owner = await db.get(User, owner_id)
    assert owner is not None
    limit = quotas.limits_for(owner, settings).storage_bytes
    if limit is None:
        return
    used = (await quotas.usage_for(db, owner_id)).storage_bytes
    if used + source.bytes > limit:
        left = max(0, limit - used)
        raise HTTPException(
            status_code=403,
            detail=f"This export would take about {_size(source.bytes)}, and the "
            f"project's owner has {_size(left)} of storage left. Delete other "
            "exports, or ask for more storage on the account page.",
        )


def _size(nbytes: int) -> str:
    if nbytes < 1024**3:
        return f"{max(1, round(nbytes / 1024**2))} MB"
    return f"{nbytes / 1024**3:.1f} GB"


@router.get("")
async def list_exports(project: MemberProject, db: DbSession) -> list[ExportOut]:
    # The newest of each, so a failed try doesn't hide behind a later one.
    newest = (
        await db.scalars(
            select(Artifact)
            .where(
                Artifact.project_id == project.id,
                Artifact.kind == "export",
                Artifact.state.in_(LISTED),
                Artifact.expires_at > artifacts.now(),
            )
            .ext(distinct_on(Artifact.cache_key))
            .order_by(Artifact.cache_key, Artifact.created_at.desc())
        )
    ).all()
    exports = sorted(newest, key=lambda e: e.created_at, reverse=True)[:50]
    return [await _out(db, export) for export in exports]


async def _export(
    db: AsyncSession, project_id: uuid.UUID, export_id: uuid.UUID, lock: bool = False
) -> Artifact:
    query = select(Artifact).where(
        Artifact.id == export_id,
        Artifact.project_id == project_id,
        Artifact.kind == "export",
        Artifact.state.in_(LISTED),
        Artifact.expires_at > artifacts.now(),
    )
    if lock:
        query = query.with_for_update().execution_options(populate_existing=True)
    export = await db.scalar(query)
    if export is None:
        raise HTTPException(status_code=404, detail="No such export.")
    return export


@router.delete("/{export_id}", status_code=204)
async def delete_export(
    export_id: uuid.UUID,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> None:
    """
    Stop an export being made, or let garbage collection delete a finished
    one (which gives its storage back; downloads already running get a
    little longer).
    """
    export = await _export(db, project.id, export_id)
    # The job first, then the export: the order finishing a job takes them.
    if export.state == "staging" and export.produced_by_job is not None:
        job = await db.get(Job, export.produced_by_job)
        if job is not None and job.status in ACTIVE:
            await jobs.cancel_pipeline(db, job.root_id)
    export = await _export(db, project.id, export_id, lock=True)
    export.expires_at = artifacts.now()
    audit.record(
        db,
        actor_id=auth.user.id,
        action="export.delete",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"export_id": str(export.id)},
    )
    await db.commit()


@router.head("/{export_id}/download", summary="Head an export's archive")
@router.get("/{export_id}/download", summary="Download an export's archive")
async def download(
    export_id: uuid.UUID,
    project: MemberProject,
    db: DbSession,
    settings: SettingsDep,
    request: Request,
) -> Response:
    """
    The archive, served from its parts as one file.
    """
    export = await _export(db, project.id, export_id)
    if export.state != "committed" or not export.manifest:
        raise HTTPException(status_code=404, detail="That export isn't ready.")
    # Keep it while this download runs, even past its week.
    await db.execute(
        update(Artifact)
        .where(Artifact.id == export.id)
        .values(
            expires_at=func.greatest(Artifact.expires_at, func.now() + DOWNLOAD_GRACE)
        )
        .execution_options(synchronize_session=False)
    )
    await db.commit()
    sizes = [int(n) for n in export.manifest["parts"]]
    total = sum(sizes)
    filename = export.inputs.get("filename", "export.zip")
    store = object_store(
        project_storage(settings).child(artifacts.artifact_path(export))
    )
    # Give the connection back before streaming (which expires `export`).
    await db.rollback()
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        "Accept-Ranges": "bytes",
        "ETag": f'"{export_id}"',
        "Cache-Control": "private, max-age=31536000, immutable",
    }
    start, end, status = 0, total, 200
    if range_header := request.headers.get("range"):
        wanted = objects.parse_range(range_header)
        if isinstance(wanted, tuple):
            start, end = wanted[0], min(wanted[1], total)
        elif "offset" in wanted:
            start = wanted["offset"]
        else:
            start = max(0, total - wanted["suffix"])
        if start >= end:
            raise HTTPException(
                status_code=416,
                detail="Range not satisfiable.",
                headers={"Content-Range": f"bytes */{total}"},
            )
        status = 206
        headers["Content-Range"] = f"bytes {start}-{end - 1}/{total}"
    headers["Content-Length"] = str(end - start)
    if request.method == "HEAD":
        return Response(
            status_code=status, headers=headers, media_type="application/zip"
        )

    async def body():
        offset = 0
        for index, size in enumerate(sizes):
            lo, hi = max(start, offset), min(end, offset + size)
            if lo < hi:
                result = await obstore.get_async(
                    store,
                    part_key(index),
                    options={"range": (lo - offset, hi - offset)},
                )
                async for chunk in result.stream():
                    yield memoryview(chunk)
            offset += size
            if offset >= end:
                break

    return StreamingResponse(
        body(), status_code=status, headers=headers, media_type="application/zip"
    )
