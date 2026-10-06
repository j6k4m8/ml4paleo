"""
Meshes of the final segmentation, one per class.

    POST /api/projects/{id}/meshes   {downsample?, method?, simplify?}
    GET  /api/projects/{id}/meshes   the current meshes and their files

Meshes are in (x, y, z) order, in the scan's physical units when it gave a
voxel size (else in voxels), as STL, OBJ, and GLB per class.
"""

import datetime
import uuid
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from .. import artifacts, audit
from ..auth.deps import CurrentAuth, DbSession
from ..db import LabelClass
from ..pipelines import mesh
from .gateway import files_path
from .projects import MemberProject

router = APIRouter(prefix="/api/projects/{project_id}/meshes", tags=["meshes"])


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
) -> MeshStarted:
    segmentation = await artifacts.head(db, project.id, "segmentation")
    image = await artifacts.head(db, project.id, "image")
    if segmentation is None or not segmentation.manifest or image is None:
        raise HTTPException(status_code=409, detail="Make a final segmentation first.")
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
    root, artifact = await mesh.start(
        db,
        segmentation=segmentation,
        image=image,
        classes=classes,
        downsample=body.downsample,
        method=body.method,
        simplify=body.simplify,
        created_by=auth.user.id,
    )
    audit.record(
        db,
        actor_id=auth.user.id,
        action="meshes.make",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"pipeline_id": str(root.id), **body.model_dump()},
    )
    await db.commit()
    return MeshStarted(pipeline_id=root.id, artifact_id=artifact.id)


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
