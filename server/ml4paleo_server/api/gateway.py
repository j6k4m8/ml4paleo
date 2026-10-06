"""
The data gateway: viewers (the web app, and Neuroglancer at /neuroglancer/)
read a project's artifacts through the API, on the same origin, signed in as
themselves.

    GET|HEAD /api/projects/{id}/artifacts/{artifact}/zarr/{key}

Only members of the project can read, and only committed artifacts (or ones
since replaced but not yet deleted). Committed artifacts never change, so
answers are cached as immutable. Byte ranges work as for any zarr store.
"""

import uuid

from fastapi import APIRouter, HTTPException, Request, Response
from sqlalchemy import select

from ml4paleo.storage import StorageGrant, object_store

from .. import artifacts, objects
from ..auth.deps import DbSession, SettingsDep
from ..db import Artifact
from ..storage import project_storage
from .projects import MemberProject

router = APIRouter(
    prefix="/api/projects/{project_id}/artifacts/{artifact_id}/zarr",
    tags=["data"],
)

READABLE = ("committed", "superseded")
IMMUTABLE = {"Cache-Control": "private, max-age=31536000, immutable"}
_KEY_CHECK = StorageGrant(url="s3://key-check")


def zarr_path(project_id: uuid.UUID, artifact_id: uuid.UUID) -> str:
    return f"/api/projects/{project_id}/artifacts/{artifact_id}/zarr/"


@router.head("/{key:path}", summary="Head an artifact file")
@router.get("/{key:path}", summary="Read an artifact file")
async def read(
    artifact_id: uuid.UUID,
    key: str,
    project: MemberProject,
    db: DbSession,
    settings: SettingsDep,
    request: Request,
) -> Response:
    if not key.strip("/"):
        raise HTTPException(status_code=400, detail="Name a file.")
    try:
        _KEY_CHECK.child(key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None
    artifact = await db.scalar(
        select(Artifact).where(
            Artifact.id == artifact_id,
            Artifact.project_id == project.id,
            Artifact.state.in_(READABLE),
        )
    )
    if artifact is None:
        raise HTTPException(status_code=404, detail="No such artifact.")
    path = artifacts.artifact_path(artifact)
    # Give the connection back before streaming.
    await db.rollback()
    store = object_store(project_storage(settings).child(path))
    return await objects.serve(store, key.strip("/"), request, headers=IMMUTABLE)
