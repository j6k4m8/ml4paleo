"""
Regions of interest: the cubes and slabs where people label for training.

    GET    /api/projects/{id}/rois
    POST   /api/projects/{id}/rois         {bbox, kind, split?}
    PATCH  /api/projects/{id}/rois/{roi}   {status?, split?}
    DELETE /api/projects/{id}/rois/{roi}

`bbox` is global (z0, y0, x0, z1, y1, x1), half-open, inside the image. A
"slice" ROI is one voxel thick along some axis. Marking an ROI complete says
its unlabeled voxels are background, which training then uses.
"""

import datetime
import uuid
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy import select

from .. import audit, labels
from ..auth.deps import CurrentAuth, DbSession
from ..db import Roi
from .projects import MemberProject

router = APIRouter(prefix="/api/projects/{project_id}/rois", tags=["rois"])

MAX_ROIS = 10_000
Status = Literal["open", "complete", "skipped"]
Split = Literal["train", "val"]


class RoiIn(BaseModel):
    bbox: tuple[int, int, int, int, int, int]
    kind: Literal["cube", "slice"] = "cube"
    split: Split = "train"


class RoiPatch(BaseModel):
    status: Status | None = None
    split: Split | None = None


class RoiOut(BaseModel):
    id: uuid.UUID
    bbox: list[int]
    kind: str
    status: str
    split: str
    origin: str
    score: float | None
    created_at: datetime.datetime


def _out(roi: Roi) -> RoiOut:
    return RoiOut(
        id=roi.id,
        bbox=roi.bbox,
        kind=roi.kind,
        status=roi.status,
        split=roi.split,
        origin=roi.origin,
        score=roi.score,
        created_at=roi.created_at,
    )


@router.get("")
async def list_rois(project: MemberProject, db: DbSession) -> list[RoiOut]:
    rois = (
        await db.scalars(
            select(Roi).where(Roi.project_id == project.id).order_by(Roi.created_at)
        )
    ).all()
    return [_out(roi) for roi in rois]


@router.post("", status_code=201)
async def add_roi(
    body: RoiIn,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> RoiOut:
    try:
        shape = await labels.volume_shape(db, project.id)
    except labels.NoImage:
        raise HTTPException(
            status_code=409, detail="This project has no image yet."
        ) from None
    start, stop = body.bbox[:3], body.bbox[3:]
    if any(
        a < 0 or b > n or a >= b for a, b, n in zip(start, stop, shape, strict=True)
    ):
        raise HTTPException(status_code=422, detail="The box must be inside the image.")
    if (
        body.kind == "slice"
        and min(b - a for a, b in zip(start, stop, strict=True)) != 1
    ):
        raise HTTPException(status_code=422, detail="A slice is one voxel thick.")
    count = len(
        (await db.scalars(select(Roi.id).where(Roi.project_id == project.id))).all()
    )
    if count >= MAX_ROIS:
        raise HTTPException(status_code=409, detail="This project has too many ROIs.")
    roi = Roi(
        project_id=project.id,
        created_by=auth.user.id,
        bbox=list(body.bbox),
        kind=body.kind,
        split=body.split,
        status="open",
        origin="user",
    )
    db.add(roi)
    await db.flush()
    audit.record(
        db,
        actor_id=auth.user.id,
        action="roi.add",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"roi_id": str(roi.id), "bbox": roi.bbox},
    )
    await db.commit()
    await db.refresh(roi)
    return _out(roi)


async def _roi(db, project, roi_id: uuid.UUID) -> Roi:
    roi = await db.scalar(
        select(Roi).where(Roi.id == roi_id, Roi.project_id == project.id)
    )
    if roi is None:
        raise HTTPException(status_code=404, detail="No such ROI.")
    return roi


@router.patch("/{roi_id}")
async def edit_roi(
    roi_id: uuid.UUID, body: RoiPatch, project: MemberProject, db: DbSession
) -> RoiOut:
    roi = await _roi(db, project, roi_id)
    if body.status is not None:
        roi.status = body.status
    if body.split is not None:
        roi.split = body.split
    await db.commit()
    await db.refresh(roi)
    return _out(roi)


@router.delete("/{roi_id}", status_code=204)
async def remove_roi(
    roi_id: uuid.UUID,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> None:
    """
    Remove an ROI (its labels stay).
    """
    roi = await _roi(db, project, roi_id)
    await db.delete(roi)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="roi.remove",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"roi_id": str(roi_id)},
    )
    await db.commit()
