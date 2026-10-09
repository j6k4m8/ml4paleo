"""
A project's data in Neuroglancer.

    GET /api/projects/{id}/neuroglancer    a link to the project's Neuroglancer view

The view has the image, the labels, the project's current prediction, and its
final segmentation as layers (see `neuroglancer_link`), read through the data
gateway as whoever opens the link, so it works only for members. It is for
looking: nothing in it can change the project. `url` is null when this server
has no Neuroglancer (`M4P_NEUROGLANCER_DIR`); a project with no image yet has
nothing to show (404).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from .. import artifacts
from ..auth.deps import DbSession, SettingsDep
from ..db import LabelClass
from ..viewer import neuroglancer_available, neuroglancer_link
from .gateway import zarr_path
from .projects import MemberProject

router = APIRouter(
    prefix="/api/projects/{project_id}/neuroglancer", tags=["neuroglancer"]
)


class NeuroglancerOut(BaseModel):
    # A link to this server's Neuroglancer with the view in it, relative to
    # the site (it starts with /neuroglancer/).
    url: str | None


@router.get("")
async def neuroglancer_view(
    project: MemberProject, db: DbSession, settings: SettingsDep
) -> NeuroglancerOut:
    if not neuroglancer_available(settings):
        return NeuroglancerOut(url=None)
    image = await artifacts.head(db, project.id, "image")
    if image is None or not image.manifest:
        raise HTTPException(status_code=404, detail="This project has no image yet.")
    classes = (
        await db.execute(
            select(LabelClass.value, LabelClass.color)
            .where(LabelClass.project_id == project.id, LabelClass.deleted_at.is_(None))
            .order_by(LabelClass.value)
        )
    ).all()
    # Only a prediction of this image, as `GET /prediction` has it (one of an
    # image since replaced doesn't fit the new one), and a final segmentation
    # of its shape, as `GET /segmentation` has it (it doesn't record the image).
    prediction = await artifacts.head(db, project.id, "prediction")
    segmentation = await artifacts.head(db, project.id, "segmentation")
    shape = image.manifest["shape_czyx"][1:]
    return NeuroglancerOut(
        url=neuroglancer_link(
            settings.public_url,
            zarr_path(project.id, image.id),
            image.manifest,
            labels_url=f"/api/projects/{project.id}/labels/zarr/",
            classes=[(value, color) for value, color in classes],
            prediction_url=zarr_path(project.id, prediction.id)
            if prediction is not None
            and prediction.manifest
            and prediction.inputs.get("image_artifact_id") == str(image.id)
            else None,
            segmentation_url=zarr_path(project.id, segmentation.id)
            if segmentation is not None
            and segmentation.manifest
            and segmentation.manifest.get("shape_zyx") == shape
            else None,
        )
    )
