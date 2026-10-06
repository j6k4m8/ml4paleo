"""
The final segmentation.

    POST /api/projects/{id}/segmentation   {min_voxels?}   from the current prediction
    GET  /api/projects/{id}/segmentation                   the current one

The final segmentation is the project's prediction, overruled by its labels
(complete ROIs count as background where unlabeled), with pieces of a class
smaller than `min_voxels` removed unless someone labeled part of them.
"""

import datetime
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from .. import artifacts, audit
from ..auth.deps import CurrentAuth, DbSession
from ..db import TrainedModel
from ..pipelines import compose
from .gateway import zarr_path
from .projects import MemberProject

router = APIRouter(
    prefix="/api/projects/{project_id}/segmentation", tags=["segmentation"]
)


class ComposeIn(BaseModel):
    # Smaller pieces of a class are specks, and become background.
    min_voxels: int = Field(default=50, ge=0, le=10**9)


class ComposeStarted(BaseModel):
    pipeline_id: uuid.UUID
    artifact_id: uuid.UUID


@router.post("", status_code=202)
async def make_segmentation(
    body: ComposeIn,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> ComposeStarted:
    prediction = await artifacts.head(db, project.id, "prediction")
    if prediction is None or not prediction.manifest:
        raise HTTPException(status_code=409, detail="Predict with a model first.")
    try:
        root, artifact = await compose.start(
            db,
            request.app.state.sessionmaker,
            prediction=prediction,
            min_voxels=body.min_voxels,
            created_by=auth.user.id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    audit.record(
        db,
        actor_id=auth.user.id,
        action="segmentation.compose",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"pipeline_id": str(root.id), "min_voxels": body.min_voxels},
    )
    await db.commit()
    return ComposeStarted(pipeline_id=root.id, artifact_id=artifact.id)


class SegmentationOut(BaseModel):
    artifact_id: uuid.UUID
    model_name: str | None
    min_voxels: int
    # The last label edit it includes.
    label_seq: int
    # Its zarr group (an array `class`), through the data gateway.
    zarr_url: str
    committed_at: datetime.datetime


@router.get("")
async def current_segmentation(
    project: MemberProject, db: DbSession
) -> SegmentationOut:
    head = await artifacts.head(db, project.id, "segmentation")
    if head is None or not head.manifest:
        raise HTTPException(
            status_code=404, detail="This project has no final segmentation yet."
        )
    model_id = head.manifest.get("model_id")
    model = await db.get(TrainedModel, uuid.UUID(model_id)) if model_id else None
    return SegmentationOut(
        artifact_id=head.id,
        model_name=model.name if model else None,
        min_voxels=int(head.manifest.get("min_voxels", 0)),
        label_seq=int(head.manifest.get("label_seq", 0)),
        zarr_url=zarr_path(project.id, head.id),
        committed_at=head.state_changed_at,
    )
