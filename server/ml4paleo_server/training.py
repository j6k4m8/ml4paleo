"""
Training sets: what a model trains on, pinned.

A training set's manifest names the project's image artifact, its label
classes, its open and complete ROIs, and every labeled chunk by content hash,
all read from one consistent snapshot of the database. Label blobs never
change, so the manifest pins the labels exactly, however much people edit
afterwards. Its id is the SHA-256 of the manifest, so the same image,
classes, ROIs, and labels make the same training set. The manifest lives in
project storage at `projects/<project>/training/<id>/manifest.json`, where the
training job reads it.
"""

import hashlib
import json
import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from starlette.concurrency import run_in_threadpool

from ml4paleo.storage import put_bytes

from . import artifacts
from .db import LabelChunk, LabelClass, LabelOp, Roi, TrainingSet
from .settings import Settings
from .storage import project_storage

MANIFEST = "manifest.json"


class NotReady(Exception):
    """
    The project has nothing to train on yet (no image, classes, or labels).
    """


def training_path(project_id: uuid.UUID, set_id: str) -> str:
    return f"projects/{project_id}/training/{set_id}"


def manifest_id(manifest: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical(manifest)).hexdigest()


def _canonical(manifest: dict[str, Any]) -> bytes:
    return json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()


async def read_snapshot(
    sessionmaker: async_sessionmaker[AsyncSession], project_id: uuid.UUID
) -> tuple[dict[str, Any], int]:
    """
    The manifest of the project's training data now, and the last label op
    it includes.
    """
    async with sessionmaker() as db:
        # One snapshot for everything, so labels, ROIs, and classes agree.
        await db.connection(execution_options={"isolation_level": "REPEATABLE READ"})
        image = await artifacts.head(db, project_id, "image")
        if image is None or not image.manifest:
            raise NotReady("This project has no image yet.")
        classes = list(
            await db.scalars(
                select(LabelClass.value)
                .where(
                    LabelClass.project_id == project_id, LabelClass.deleted_at.is_(None)
                )
                .order_by(LabelClass.value)
            )
        )
        rois = (
            await db.scalars(
                select(Roi)
                .where(
                    Roi.project_id == project_id, Roi.status.in_(("open", "complete"))
                )
                .order_by(Roi.id)
            )
        ).all()
        chunks = (
            await db.execute(
                select(
                    LabelChunk.cz, LabelChunk.cy, LabelChunk.cx, LabelChunk.class_sha
                )
                .where(
                    LabelChunk.project_id == project_id,
                    LabelChunk.class_sha.is_not(None),
                    LabelChunk.labeled_voxels > 0,
                )
                .order_by(LabelChunk.cz, LabelChunk.cy, LabelChunk.cx)
            )
        ).all()
        label_seq = await db.scalar(
            select(func.max(LabelOp.seq)).where(LabelOp.project_id == project_id)
        )
        # Copy what's needed out of the rows before the snapshot ends.
        image_info = {
            "artifact_id": str(image.id),
            "shape_czyx": image.manifest["shape_czyx"],
            "window": image.manifest.get("window") or [0, 1],
        }
        roi_info = [
            {"bbox": list(roi.bbox), "status": roi.status, "split": roi.split}
            for roi in rois
        ]
        await db.rollback()
    if not classes:
        raise NotReady("Add a label class first.")
    if not chunks:
        raise NotReady("Label something first.")
    manifest = {
        "version": 1,
        "project_id": str(project_id),
        "image": image_info,
        "class_values": classes,
        "rois": roi_info,
        "chunks": [[cz, cy, cx, sha] for cz, cy, cx, sha in chunks],
    }
    return manifest, int(label_seq or 0)


async def snapshot(
    db: AsyncSession,
    sessionmaker: async_sessionmaker[AsyncSession],
    settings: Settings,
    project_id: uuid.UUID,
) -> TrainingSet:
    """
    Pin the project's training data now: store its manifest (once) and
    record the training set; the caller commits.
    """
    manifest, label_seq = await read_snapshot(sessionmaker, project_id)
    set_id = manifest_id(manifest)
    grant = project_storage(settings).child(training_path(project_id, set_id))
    # If training is refused after all (another training took the owner's
    # last model slot since the API checked), this manifest stays behind.
    # Nothing collects those yet; they're small, and the same labels reuse
    # them.
    await run_in_threadpool(put_bytes, grant, MANIFEST, _canonical(manifest))
    rois = manifest["rois"]
    summary = {
        "image_artifact_id": manifest["image"]["artifact_id"],
        "class_values": manifest["class_values"],
        "label_seq": label_seq,
        "labeled_chunks": len(manifest["chunks"]),
        "rois": {
            "complete": sum(r["status"] == "complete" for r in rois),
            "open": sum(r["status"] == "open" for r in rois),
            "validation": sum(r["split"] == "val" for r in rois),
        },
    }
    await db.execute(
        insert(TrainingSet)
        .values(id=set_id, project_id=project_id, summary=summary)
        .on_conflict_do_nothing()
    )
    training_set = await db.get(TrainingSet, set_id)
    assert training_set is not None
    return training_set
