"""
Meshes of the final segmentation, one per class.

    POST /api/projects/{id}/meshes   {downsample?, method?, simplify?}
    GET  /api/projects/{id}/meshes   the current meshes and their files

Meshes are in (x, y, z) order, in the scan's physical units when it gave a
voxel size (else in voxels), as STL, OBJ, and GLB per class. GLB files keep
those units too, with the scene scaled to meters, as glTF expects.

One project makes one set of meshes at a time (409 while it does), and not
while its owner is out of storage (403).
"""

import datetime
import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import aliased

from ml4paleo.meshing.blocks import TooDetailed

from .. import artifacts, audit, quotas
from ..auth.deps import CurrentAuth, DbSession, SettingsDep
from ..compression import MESH_MEDIA_TYPE, preview_body
from ..db import Job, LabelClass, Project, User
from ..jobs.queue import WAITING
from ..mesh_preview import MAX_INPUT, PreviewBusy
from ..pipelines import mesh
from .gateway import files_path
from .projects import MemberProject

router = APIRouter(prefix="/api/projects/{project_id}/meshes", tags=["meshes"])


@router.post("/preview")
async def preview_mesh(
    project: MemberProject,
    request: Request,
    chunk: str | None = Query(default=None, pattern=r"^\d{1,8},\d{1,8},\d{1,8}$"),
    downsample: int = Query(default=1, ge=1, le=16),
) -> Response:
    """Mesh a bounded viewer snapshot; store nothing and never treat it as labels."""
    if request.headers.get("content-type") != "application/octet-stream":
        raise HTTPException(415, "Expected binary preview labels.")
    body = await preview_body(request, MAX_INPUT)
    try:
        target = tuple(int(n) * 64 for n in chunk.split(",")) if chunk else None
        result = await request.app.state.mesh_preview.build(
            str(project.id), body, target, downsample
        )
    except PreviewBusy:
        raise HTTPException(
            503, "3D preview is busy. Try again shortly.", headers={"Retry-After": "1"}
        ) from None
    except TooDetailed:
        raise HTTPException(
            422, {"code": "mesh_detail", "message": "Use a coarser 3D preview."}
        ) from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    return Response(
        result,
        media_type=MESH_MEDIA_TYPE,
        headers={"Cache-Control": "no-store"},
    )


class MeshIn(BaseModel):
    # Mesh at 1/downsample resolution (#22).
    downsample: Literal[1, 2, 4, 8] = 1
    # When downsampling, keep a coarse voxel if any of its voxels is the class
    # (keeps thin parts) or if most are (smoother).
    method: Literal["any", "majority"] = "any"
    # How far simplifying may move a surface, in voxels at that resolution
    # (0 keeps every triangle).
    simplify: float = Field(default=1.0, ge=0, le=4)


class MeshStarted(BaseModel):
    pipeline_id: uuid.UUID
    artifact_id: uuid.UUID


@router.post("", status_code=202)
async def make_meshes(
    body: MeshIn,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
    settings: SettingsDep,
) -> MeshStarted:
    segmentation = await artifacts.head(db, project.id, "segmentation")
    if segmentation is None or not segmentation.manifest:
        raise HTTPException(status_code=409, detail="Make a final segmentation first.")
    # Lock the project so two requests can't both start meshing it.
    await db.scalar(
        select(Project.id)
        .where(Project.id == project.id)
        .with_for_update(key_share=True)
    )
    root = aliased(Job)
    busy = await db.scalar(
        select(Job.id)
        .join(root, root.id == Job.root_id)
        .where(
            root.project_id == project.id,
            root.kind == "mesh.block",
            Job.status.in_((*WAITING, "leased")),
        )
        .limit(1)
    )
    if busy is not None:
        raise HTTPException(
            status_code=409, detail="Meshes are already being made for this project."
        )
    owner = await db.get(User, project.owner_id)
    assert owner is not None
    await quotas.check_storage(db, settings, owner)
    classes = [
        {"value": c.value, "name": c.name, "color": c.color}
        for c in await db.scalars(
            select(LabelClass)
            .where(LabelClass.project_id == project.id, LabelClass.deleted_at.is_(None))
            .order_by(LabelClass.value)
        )
    ]
    if not classes:
        raise HTTPException(
            status_code=409, detail="This project has no label classes."
        )
    try:
        started, artifact = await mesh.start(
            db,
            segmentation=segmentation,
            classes=classes,
            downsample=body.downsample,
            method=body.method,
            simplify=body.simplify,
            created_by=auth.user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    audit.record(
        db,
        actor_id=auth.user.id,
        action="meshes.make",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"pipeline_id": str(started.id), **body.model_dump()},
    )
    await db.commit()
    return MeshStarted(pipeline_id=started.id, artifact_id=artifact.id)


class MeshesOut(BaseModel):
    artifact_id: uuid.UUID
    # The final segmentation they were made from.
    segmentation_artifact_id: uuid.UUID | None
    # `mesh_info.json`: axis order, units, voxel size, and per class its
    # name, color, and files.
    info: dict[str, Any]
    # Prefix for the files named in `info`.
    files_url: str
    committed_at: datetime.datetime


@router.get("")
async def current_meshes(project: MemberProject, db: DbSession) -> MeshesOut:
    head = await artifacts.head(db, project.id, "meshes")
    if head is None or not head.manifest:
        raise HTTPException(status_code=404, detail="This project has no meshes yet.")
    info = {k: v for k, v in head.manifest.items() if k != "kind"}
    return MeshesOut(
        artifact_id=head.id,
        segmentation_artifact_id=(head.inputs or {}).get("segmentation_artifact_id"),
        info=info,
        files_url=files_path(project.id, head.id),
        committed_at=head.state_changed_at,
    )
