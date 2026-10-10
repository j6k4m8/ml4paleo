"""
A project's data in Neuroglancer.

    GET /api/projects/{id}/neuroglancer    a link to the project's Neuroglancer view

The view has the image, the labels, the project's current prediction, and its
final segmentation and existing exported meshes as layers, read through the data
gateway as whoever opens the link, so it works only for members. It is for
looking: nothing in it can change the project. Neuroglancer is always bundled;
a project with no image yet has nothing to show (404).
"""

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import select

from .. import artifacts
from ..auth.deps import DbSession, SettingsDep
from ..db import Artifact, LabelClass
from ..viewer import neuroglancer_link
from .gateway import files_path, zarr_path
from .projects import MemberProject

router = APIRouter(
    prefix="/api/projects/{project_id}/neuroglancer", tags=["neuroglancer"]
)


class NeuroglancerOut(BaseModel):
    # A link to this server's Neuroglancer with the view in it, relative to
    # the site (it starts with /neuroglancer/).
    url: str


@router.get("")
async def neuroglancer_view(
    project: MemberProject, db: DbSession, settings: SettingsDep
) -> NeuroglancerOut:
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
    meshes = await artifacts.head(db, project.id, "meshes")
    if meshes is not None and not await _meshes_match_image(db, meshes, image):
        meshes = None
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
            meshes_url=files_path(project.id, meshes.id)
            if meshes is not None
            else None,
            meshes_manifest=meshes.manifest if meshes is not None else None,
        )
    )


async def _meshes_match_image(db, meshes: Artifact, image: Artifact) -> bool:
    """Follow provenance, never guess from shape after a same-size scan replacement."""
    image_id = (meshes.inputs or {}).get("image_artifact_id")
    if image_id is not None:
        return image_id == str(image.id)
    # Existing mesh exports predate the direct image reference. Their source
    # segmentation and prediction still identify the image, even when superseded.
    source = meshes
    for key, kind in (
        ("segmentation_artifact_id", "segmentation"),
        ("prediction_artifact_id", "prediction"),
    ):
        try:
            artifact_id = uuid.UUID(str((source.inputs or {}).get(key)))
        except ValueError:
            return False
        source = await db.scalar(
            select(Artifact).where(
                Artifact.id == artifact_id,
                Artifact.project_id == image.project_id,
                Artifact.kind == kind,
            )
        )
        if source is None:
            return False
    return (source.inputs or {}).get("image_artifact_id") == str(image.id)
