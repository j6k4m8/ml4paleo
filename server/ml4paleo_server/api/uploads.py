"""
Uploads into a project (see `ml4paleo_server.uploads` for how they work).

    POST   /api/projects/{id}/uploads                    start: {filename, size}
    GET    /api/projects/{id}/uploads/{upload}            status, with stored parts
    POST   /api/projects/{id}/uploads/{upload}/part-urls  presigned part URLs
    POST   /api/projects/{id}/uploads/{upload}/complete   finish
    DELETE /api/projects/{id}/uploads/{upload}            abort or delete
"""

import datetime
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select
from starlette.concurrency import run_in_threadpool

from .. import audit, quotas
from ..auth.deps import CurrentAuth, DbSession, SettingsDep
from ..db import Upload, User
from ..uploads import (
    MAX_UPLOAD_BYTES,
    PART_URL_LIFETIME,
    MultipartStorage,
    PartsMissing,
    UploadLost,
    UploadsUnsupported,
    abort_or_delete,
    finish,
    good_parts,
    in_use,
    multipart_storage,
    new_upload,
    now,
    part_count,
)
from .projects import MemberProject

router = APIRouter(prefix="/api/projects/{project_id}/uploads", tags=["uploads"])

# Unfinished uploads a project may have at once.
MAX_OPEN_UPLOADS = 20
MAX_URLS_PER_REQUEST = 100


def get_storage(settings: SettingsDep) -> MultipartStorage:
    try:
        return multipart_storage(settings)
    except UploadsUnsupported:
        raise HTTPException(
            status_code=501,
            detail="Uploads need S3-compatible project storage.",
        ) from None


Storage = Annotated[MultipartStorage, Depends(get_storage)]


class UploadIn(BaseModel):
    filename: str = Field(min_length=1, max_length=255)
    size: int = Field(gt=0, le=MAX_UPLOAD_BYTES)

    @field_validator("filename")
    @classmethod
    def _plain_name(cls, filename: str) -> str:
        if any(c in filename for c in "/\\") or any(
            ord(c) < 0x20 or ord(c) == 0x7F for c in filename
        ):
            raise ValueError("Give the file's name, without folders")
        return filename.strip()


class UploadOut(BaseModel):
    id: uuid.UUID
    filename: str
    size: int
    part_size: int
    part_count: int
    state: str
    created_at: datetime.datetime
    completed_at: datetime.datetime | None
    expires_at: datetime.datetime
    # Parts already stored (only while uploading, and only when asked for).
    stored_parts: list[int] | None = None


def _out(upload: Upload, stored_parts: list[int] | None = None) -> UploadOut:
    return UploadOut(
        id=upload.id,
        filename=upload.filename,
        size=upload.size,
        part_size=upload.part_size,
        part_count=part_count(upload),
        state=upload.state,
        created_at=upload.created_at,
        completed_at=upload.completed_at,
        expires_at=upload.expires_at,
        stored_parts=stored_parts,
    )


async def _upload(
    db: DbSession, project: MemberProject, upload_id: uuid.UUID, lock: bool = False
) -> Upload:
    query = select(Upload).where(
        Upload.id == upload_id,
        Upload.project_id == project.id,
        Upload.state.in_(("uploading", "completing", "complete")),
    )
    if lock:
        query = query.with_for_update(key_share=True).execution_options(
            populate_existing=True
        )
    upload = await db.scalar(query)
    if upload is None:
        raise HTTPException(status_code=404, detail="No such upload.")
    return upload


@router.post("", status_code=201)
async def start_upload(
    body: UploadIn,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
    settings: SettingsDep,
    storage: Storage,
) -> UploadOut:
    """
    Start an upload. Its size counts against the project owner's storage
    quota until it is deleted.
    """
    open_uploads = await db.scalar(
        select(func.count())
        .select_from(Upload)
        .where(Upload.project_id == project.id, Upload.state == "uploading")
    )
    if (open_uploads or 0) >= MAX_OPEN_UPLOADS:
        raise HTTPException(
            status_code=409, detail="Finish or cancel some uploads first."
        )
    owner = await db.get(User, project.owner_id)
    assert owner is not None
    await quotas.reserve_storage(db, settings, owner, body.size)
    upload = new_upload(project.id, auth.user.id, body.filename, body.size)
    db.add(upload)
    await db.flush()
    upload.multipart_id = await run_in_threadpool(storage.start, upload)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="upload.start",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"upload_id": str(upload.id), "filename": upload.filename},
    )
    await db.commit()
    await db.refresh(upload)
    return _out(upload)


@router.get("")
async def list_uploads(project: MemberProject, db: DbSession) -> list[UploadOut]:
    uploads = (
        await db.scalars(
            select(Upload)
            .where(
                Upload.project_id == project.id,
                Upload.state.in_(("uploading", "complete")),
            )
            .order_by(Upload.created_at.desc())
        )
    ).all()
    return [_out(upload) for upload in uploads]


@router.get("/{upload_id}")
async def get_upload(
    upload_id: uuid.UUID, project: MemberProject, db: DbSession, storage: Storage
) -> UploadOut:
    """
    An upload's state. While it is uploading, `stored_parts` lists the parts
    storage already has, so an interrupted upload can send only the rest.
    """
    upload = await _upload(db, project, upload_id)
    if upload.state != "uploading":
        return _out(upload)
    stored = await run_in_threadpool(storage.stored_parts, upload)
    return _out(upload, good_parts(upload, stored or {}))


class PartUrlsIn(BaseModel):
    parts: list[int] = Field(min_length=1, max_length=MAX_URLS_PER_REQUEST)


class PartUrlsOut(BaseModel):
    # Part number -> URL to PUT that part's bytes to (exactly that many).
    urls: dict[int, str]
    expires_at: datetime.datetime


@router.post("/{upload_id}/part-urls")
async def part_urls(
    upload_id: uuid.UUID,
    body: PartUrlsIn,
    project: MemberProject,
    db: DbSession,
    storage: Storage,
) -> PartUrlsOut:
    upload = await _upload(db, project, upload_id)
    if upload.state != "uploading":
        raise HTTPException(status_code=409, detail="This upload is finished.")
    count = part_count(upload)
    if any(not 1 <= number <= count for number in body.parts):
        raise HTTPException(status_code=422, detail=f"Parts run from 1 to {count}.")
    urls = {number: storage.part_url(upload, number) for number in set(body.parts)}
    return PartUrlsOut(urls=urls, expires_at=now() + PART_URL_LIFETIME)


@router.post("/{upload_id}/complete")
async def complete_upload(
    upload_id: uuid.UUID,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
    settings: SettingsDep,
    storage: Storage,
) -> UploadOut:
    """
    Finish an upload once every part is stored. Finishing twice is harmless,
    and finishing again after an interrupted attempt completes it.
    """
    upload = await _upload(db, project, upload_id, lock=True)
    if upload.state == "complete":
        return _out(upload)
    try:
        upload = await finish(db, storage, upload)
    except PartsMissing as exc:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "Some parts are missing or the wrong size; send them again.",
                "parts": exc.parts,
            },
        ) from None
    except UploadLost:
        await abort_or_delete(db, settings, upload, "deleted")
        raise HTTPException(
            status_code=409, detail="The upload was lost in storage; start it again."
        ) from None
    audit.record(
        db,
        actor_id=auth.user.id,
        action="upload.complete",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"upload_id": str(upload.id), "size": upload.size},
    )
    await db.commit()
    return _out(upload)


@router.delete("/{upload_id}", status_code=204)
async def delete_upload(
    upload_id: uuid.UUID,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
    settings: SettingsDep,
    storage: Storage,
) -> None:
    """
    Cancel an unfinished upload, or delete a finished one (unless a job is
    using it). Either way its storage is given back.
    """
    upload = await _upload(db, project, upload_id, lock=True)
    busy = await db.scalar(select(Upload.id).where(Upload.id == upload.id, in_use()))
    if busy is not None:
        raise HTTPException(status_code=409, detail="A job is using this upload.")
    audit.record(
        db,
        actor_id=auth.user.id,
        action="upload.delete",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"upload_id": str(upload.id)},
    )
    await abort_or_delete(db, settings, upload, "deleted")
