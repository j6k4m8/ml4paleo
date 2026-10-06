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
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ml4paleo.export import part_key
from ml4paleo.storage import object_store

from .. import artifacts, audit, jobs, objects
from ..auth.deps import CurrentAuth, DbSession, SettingsDep
from ..db import Artifact, Job
from ..pipelines import export as export_pipeline
from ..storage import project_storage
from .projects import MemberProject

router = APIRouter(prefix="/api/projects/{project_id}/exports", tags=["exports"])

ACTIVE = ("blocked", "queued", "leased")
LISTED = ("staging", "committed", "failed")


class ExportIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["image", "prediction", "segmentation", "meshes"]
    # Volumes: "zarr" (the zarr group), "tiff", or "png" (one image per
    # slice). Meshes: "zip".
    format: Literal["zarr", "tiff", "png", "zip"]


class ExportOut(BaseModel):
    id: uuid.UUID
    source: str
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


async def _out(db: AsyncSession, export: Artifact) -> ExportOut:
    job = await db.get(Job, export.produced_by_job) if export.produced_by_job else None
    if export.state == "committed":
        status = "ready"
    elif export.state == "staging" and job is not None and job.status in ACTIVE:
        status = "making"
    else:
        status = "failed"
    error = None
    if status == "failed":
        error = (
            (job.error or "").strip().splitlines()[0][:500]
            if job and job.error
            else None
        )
        error = error or "The export didn't finish."
    assert export.expires_at is not None
    return ExportOut(
        id=export.id,
        source=export.inputs.get("source", ""),
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
        making = (
            existing.state == "staging" and job is not None and job.status in ACTIVE
        )
        if existing.state == "committed" or making:
            existing.expires_at = max(existing.expires_at or keep_until, keep_until)
            await db.commit()
            if existing.state == "committed":
                response.status_code = 200
            return await _out(db, existing)
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


@router.get("")
async def list_exports(project: MemberProject, db: DbSession) -> list[ExportOut]:
    exports = (
        await db.scalars(
            select(Artifact)
            .where(
                Artifact.project_id == project.id,
                Artifact.kind == "export",
                Artifact.state.in_(LISTED),
                Artifact.expires_at > artifacts.now(),
            )
            .order_by(Artifact.created_at.desc())
            .limit(50)
        )
    ).all()
    # The newest of each, so a failed try doesn't hide behind a later one.
    newest: dict[str | None, Artifact] = {}
    for export in exports:
        newest.setdefault(export.cache_key, export)
    return [await _out(db, export) for export in newest.values()]


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
    export = await db.scalar(query.with_for_update() if lock else query)
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
    one on its next pass (which gives its storage back).
    """
    export = await _export(db, project.id, export_id, lock=True)
    if export.state == "staging" and export.produced_by_job is not None:
        job = await db.get(Job, export.produced_by_job)
        if job is not None and job.status in ACTIVE:
            await jobs.cancel_pipeline(db, job.root_id)
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
