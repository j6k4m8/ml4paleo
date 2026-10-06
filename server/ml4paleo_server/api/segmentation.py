"""
The final segmentation.

    POST /api/projects/{id}/segmentation   {min_voxels?}   from the current prediction
    GET  /api/projects/{id}/segmentation                   the current one

The final segmentation is the project's prediction, overruled by its labels
(complete ROIs count as background where unlabeled), with pieces of a class
smaller than `min_voxels` removed unless someone labeled part of them. A
project makes one at a time: while one is waiting or running, POST answers
409 with its `pipeline_id`.
"""

import datetime
import uuid

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from .. import artifacts, audit
from ..auth.deps import CurrentAuth, DbSession
from ..db import LabelOp, Project, TrainedModel
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
    # Lock the project so two requests can't both start one.
    await db.scalar(
        select(Project.id)
        .where(Project.id == project.id)
        .with_for_update(key_share=True)
    )
    if (running := await compose.running(db, project.id)) is not None:
        raise HTTPException(
            status_code=409,
            detail={
                "message": "A final segmentation is already being made.",
                "pipeline_id": str(running),
            },
        )
    prediction = await artifacts.head(db, project.id, "prediction")
    if prediction is None or not prediction.manifest:
        raise HTTPException(status_code=409, detail="Predict with a model first.")
    image = await artifacts.head(db, project.id, "image")
    if image is None or prediction.inputs.get("image_artifact_id") != str(image.id):
        raise HTTPException(
            status_code=409,
            detail="The prediction is from an older image; predict again.",
        )
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
    # When the newest label edit it includes was made; None if it has none.
    labels_as_of: datetime.datetime | None
    # Its zarr group (an array `class`), through the data gateway.
    zarr_url: str
    committed_at: datetime.datetime


async def _model_name(db, project_id: uuid.UUID, model_id: object) -> str | None:
    """The name of this project's model with that id, if there is one."""
    try:
        wanted = uuid.UUID(str(model_id))
    except ValueError:
        return None
    return await db.scalar(
        select(TrainedModel.name).where(
            TrainedModel.id == wanted, TrainedModel.project_id == project_id
        )
    )


@router.get("")
async def current_segmentation(
    project: MemberProject, db: DbSession
) -> SegmentationOut:
    head = await artifacts.head(db, project.id, "segmentation")
    if head is None or not head.manifest:
        raise HTTPException(
            status_code=404, detail="This project has no final segmentation yet."
        )
    # What the server recorded when it started, not what the worker wrote.
    inputs = head.inputs or {}
    labels_as_of = await db.scalar(
        select(LabelOp.created_at)
        .where(
            LabelOp.project_id == project.id,
            LabelOp.seq <= int(inputs.get("label_seq", 0)),
        )
        .order_by(LabelOp.seq.desc())
        .limit(1)
    )
    return SegmentationOut(
        artifact_id=head.id,
        model_name=await _model_name(db, project.id, inputs.get("model_id")),
        min_voxels=int(inputs.get("min_voxels", 0)),
        labels_as_of=labels_as_of,
        zarr_url=zarr_path(project.id, head.id),
        committed_at=head.state_changed_at,
    )
