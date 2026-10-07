"""
Labels from a file: a TIFF stack, or a zip of TIFF or PNG slices, the size
of the project's image (see `pipelines.labelimport`).

    GET    /api/projects/{id}/labels/imports               recent imports
    POST   /api/projects/{id}/labels/imports               {upload_id}: check it
    GET    /api/projects/{id}/labels/imports/{import}
    POST   /api/projects/{id}/labels/imports/{import}/start
                                                   {mapping, overwrite}: import it
    DELETE /api/projects/{id}/labels/imports/{import}      stop, let the file go

An import is named by its check's pipeline. Once checked, it lists each value
in the file with its voxels; starting it says what each value becomes
(background, a class, or a new class; values left out stay unlabeled), and
`pipeline` then follows the import.
"""

import datetime
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from ml4paleo.labelimport import MAX_VALUES
from ml4paleo.labels import BACKGROUND, MAX_CLASS

from .. import audit, jobs, labels
from ..auth.deps import CurrentAuth, DbSession
from ..db import Job, LabelClass, Project, Upload
from ..pipelines import labelimport
from ..uploads import now
from .labels import ClassIn, allowed_values, new_class_values
from .pipelines import PipelineOut, pipeline_out
from .projects import MemberProject

router = APIRouter(prefix="/api/projects/{project_id}/labels/imports", tags=["labels"])

RUNNING = ("waiting", "running")
# An import's state, from its pipeline's.
IMPORT_STATES = {"waiting": "importing", "running": "importing", "succeeded": "done"}


class CheckIn(BaseModel):
    upload_id: uuid.UUID


class FoundValue(BaseModel):
    value: int
    voxels: int


class LabelImportOut(BaseModel):
    id: uuid.UUID
    filename: str
    upload_id: uuid.UUID
    created_at: datetime.datetime
    created_by: uuid.UUID | None
    # "checking", "ready" (to say what the values become, and import),
    # "importing", "done", "failed", "cancelled", or "expired" (checked, but
    # the file is gone).
    state: str
    error: str | None
    # The pipeline to follow: the check's, then the import's.
    pipeline: PipelineOut
    # Once checked: the labels' size (z, y, x), and each value with its voxels.
    shape_zyx: list[int] | None
    values: list[FoundValue] | None
    # Once started: what each value became, and whether it replaced labels.
    lookup: list[tuple[int, int]] | None
    overwrite: bool | None


def _kept(upload: Upload | None) -> bool:
    """Whether a checked file is still here to import."""
    return (
        upload is not None and upload.state == "complete" and upload.expires_at > now()
    )


async def _out(db, probe: Job) -> LabelImportOut:
    check = await pipeline_out(db, probe)
    started = await labelimport.import_of(db, probe)
    upload = await db.get(Upload, uuid.UUID(probe.payload["upload_id"]))
    result = probe.result if check.status == "succeeded" else None
    if started is not None:
        current = await pipeline_out(db, started)
        status = IMPORT_STATES.get(current.status, current.status)
    else:
        current = check
        if check.status in RUNNING:
            status = "checking"
        elif check.status == "succeeded":
            status = "ready" if _kept(upload) else "expired"
        else:
            status = check.status
    return LabelImportOut(
        id=probe.id,
        filename=probe.payload["filename"],
        upload_id=probe.payload["upload_id"],
        created_at=probe.created_at,
        created_by=probe.created_by,
        state=status,
        error=current.error,
        pipeline=current,
        shape_zyx=result["shape_zyx"] if result else None,
        values=[FoundValue(value=v, voxels=n) for v, n in result["values"]]
        if result
        else None,
        lookup=[(v, label) for v, label in started.payload["lookup"]]
        if started
        else None,
        overwrite=started.payload["overwrite"] if started else None,
    )


async def _probe(db, project, import_id: uuid.UUID, lock: bool = False) -> Job:
    query = select(Job).where(
        Job.id == import_id,
        Job.project_id == project.id,
        Job.kind == labelimport.PROBE,
    )
    if lock:
        query = query.with_for_update(key_share=True).execution_options(
            populate_existing=True
        )
    probe = await db.scalar(query)
    if probe is None:
        raise HTTPException(status_code=404, detail="No such import.")
    return probe


async def _upload(db, upload_id: Any) -> Upload | None:
    return await db.scalar(
        select(Upload)
        .where(Upload.id == uuid.UUID(str(upload_id)))
        .with_for_update(key_share=True)
        .execution_options(populate_existing=True)
    )


async def _image_shape(db, project) -> list[int]:
    try:
        return list(await labels.volume_shape(db, project.id))
    except labels.NoImage:
        raise HTTPException(
            status_code=409, detail="This project has no image yet."
        ) from None


@router.get("")
async def list_imports(project: MemberProject, db: DbSession) -> list[LabelImportOut]:
    return [await _out(db, probe) for probe in await labelimport.checks(db, project.id)]


@router.post("", status_code=202)
async def check_file(
    body: CheckIn,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> LabelImportOut:
    """
    Check a finished upload: that it's the image's size, and what values it
    holds. Checking an upload again gives its first check.
    """
    shape = await _image_shape(db, project)
    upload = await _upload(db, body.upload_id)
    if upload is None or upload.project_id != project.id or not _kept(upload):
        raise HTTPException(status_code=404, detail="No finished upload with that id.")
    probe = await labelimport.start_check(db, upload, shape, auth.user.id)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="labels.import.check",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"upload_id": str(upload.id), "import_id": str(probe.id)},
    )
    await db.commit()
    await db.refresh(probe)
    return await _out(db, probe)


@router.get("/{import_id}")
async def get_import(
    import_id: uuid.UUID, project: MemberProject, db: DbSession
) -> LabelImportOut:
    return await _out(db, await _probe(db, project, import_id))


class MapIn(BaseModel):
    """
    What one value in the file becomes: background (1) or a class (`label`),
    or a new class.
    """

    model_config = ConfigDict(extra="forbid")

    value: int
    label: int | None = Field(default=None, ge=BACKGROUND, le=MAX_CLASS)
    new_class: ClassIn | None = None

    @model_validator(mode="after")
    def _one(self) -> "MapIn":
        if (self.label is None) == (self.new_class is None):
            raise ValueError("give each value a label or a new class")
        return self


class StartIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Values left out stay unlabeled.
    mapping: list[MapIn] = Field(min_length=1, max_length=MAX_VALUES + 1)
    # Replace labels already there, rather than fill only unlabeled voxels.
    overwrite: bool = False


@router.post("/{import_id}/start", status_code=202)
async def start_import(
    import_id: uuid.UUID,
    body: StartIn,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> LabelImportOut:
    """
    Import a checked file's labels, making the new classes it asks for.
    """
    # Locked, so the same file can't start importing twice.
    probe = await _probe(db, project, import_id, lock=True)
    check = await pipeline_out(db, probe)
    if check.status != "succeeded" or probe.result is None:
        raise HTTPException(
            status_code=409, detail="The file can't import until its check passes."
        )
    if await labelimport.import_of(db, probe) is not None:
        raise HTTPException(status_code=409, detail="This file was already imported.")
    found = {value for value, _ in probe.result["values"]}
    mapped = [item.value for item in body.mapping]
    if len(set(mapped)) != len(mapped):
        raise HTTPException(status_code=422, detail="Say what each value becomes once.")
    if missing := sorted(set(mapped) - found):
        raise HTTPException(
            status_code=422,
            detail=f"The file doesn't hold {', '.join(map(str, missing))}.",
        )
    if await _image_shape(db, project) != probe.result["shape_zyx"]:
        raise HTTPException(
            status_code=409,
            detail="The image changed size since the file was checked; upload labels "
            "the size of the new image.",
        )
    allowed = await allowed_values(db, project.id)
    if unknown := sorted(
        {item.label for item in body.mapping if item.label is not None} - allowed
    ):
        raise HTTPException(
            status_code=422,
            detail=f"This project has no class {', '.join(map(str, unknown))}.",
        )
    # The project before the file, the order starting a prediction takes
    # them in (new classes need the project anyway).
    await db.scalar(
        select(Project.id)
        .where(Project.id == project.id)
        .with_for_update(key_share=True)
    )
    upload = await _upload(db, probe.payload["upload_id"])
    if upload is None or not _kept(upload):
        raise HTTPException(
            status_code=409, detail="The file is gone; upload it again."
        )
    new = [item for item in body.mapping if item.new_class is not None]
    values = iter(await new_class_values(db, project.id, len(new)) if new else [])
    lookup = []
    added = []
    for item in body.mapping:
        label = item.label
        if item.new_class is not None:
            label = next(values)
            db.add(
                LabelClass(
                    project_id=project.id,
                    value=label,
                    name=item.new_class.name,
                    color=item.new_class.color,
                )
            )
            added.append({"value": label, "name": item.new_class.name})
        assert label is not None
        lookup.append((item.value, label))
    try:
        started = await labelimport.start_import(
            db,
            probe,
            upload,
            lookup,
            overwrite=body.overwrite,
            created_by=auth.user.id,
        )
    except ValueError:
        # Garbage collection took the file meanwhile.
        raise HTTPException(
            status_code=409, detail="The file is gone; upload it again."
        ) from None
    audit.record(
        db,
        actor_id=auth.user.id,
        action="labels.import.start",
        target_type="project",
        target_id=project.id,
        request=request,
        details={
            "import_id": str(probe.id),
            "pipeline_id": str(started.id),
            "new_classes": added,
        },
    )
    await db.commit()
    return await _out(db, probe)


@router.delete("/{import_id}", status_code=204)
async def discard_import(
    import_id: uuid.UUID,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> None:
    """
    Stop checking or importing a file, and let it go (its storage is given
    back once no job uses it). Labels an import already brought in stay.
    """
    probe = await _probe(db, project, import_id)
    for root in (probe, await labelimport.import_of(db, probe)):
        if root is not None and (await pipeline_out(db, root)).status in RUNNING:
            await jobs.cancel_pipeline(db, root.id)
    upload = await _upload(db, probe.payload["upload_id"])
    if upload is not None and upload.state == "complete":
        upload.expires_at = min(upload.expires_at, now())
    audit.record(
        db,
        actor_id=auth.user.id,
        action="labels.import.discard",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"import_id": str(probe.id)},
    )
    await db.commit()
