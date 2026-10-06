"""
The final segmentation: the project's prediction, overruled by its labels,
with specks removed (see `ml4paleo.segmentation.compose`), into a
"segmentation" artifact that becomes the project's "segmentation" head.

    compose.prepare -> cc.block x N -> cc.merge -> cc.apply x N -> compose.finalize

The labels and complete ROIs are pinned when the pipeline starts: their
chunk hashes go to `compose.prepare` in its payload, and it writes them into
the artifact as `inputs.json`, so edits made while the pipeline runs don't
change the result, and a start that never commits leaves no files behind.
"""

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ml4paleo.segmentation.predict import SHARD_ZYX, shard_boxes

from .. import artifacts, jobs
from ..db import Artifact, Job, LabelChunk, LabelOp, Roi

WEIGHTS = {"prepare": 1.0, "blocks": 45.0, "merge": 4.0, "apply": 45.0, "finalize": 1.0}


async def pinned_labels(
    sessionmaker: async_sessionmaker[AsyncSession], project_id: uuid.UUID
) -> dict[str, Any]:
    """The project's labeled chunks and complete ROIs, from one snapshot."""
    async with sessionmaker() as db:
        await db.connection(execution_options={"isolation_level": "REPEATABLE READ"})
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
        complete = (
            await db.scalars(
                select(Roi.bbox)
                .where(Roi.project_id == project_id, Roi.status == "complete")
                .order_by(Roi.id)
            )
        ).all()
        label_seq = await db.scalar(
            select(func.max(LabelOp.seq)).where(LabelOp.project_id == project_id)
        )
        await db.rollback()
    return {
        "chunks": [[cz, cy, cx, sha] for cz, cy, cx, sha in chunks],
        "complete_rois": [list(bbox) for bbox in complete],
        "label_seq": int(label_seq or 0),
    }


async def start(
    db: AsyncSession,
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    prediction: Artifact,
    min_voxels: int,
    created_by: uuid.UUID,
) -> tuple[Job, Artifact]:
    """
    Start a final segmentation; the caller commits. Raises ValueError if
    jobs can't be given the prediction (for example it is being deleted).
    """
    assert prediction.manifest is not None
    shape = [int(n) for n in prediction.manifest["shape_zyx"]]
    inputs = await pinned_labels(sessionmaker, prediction.project_id)
    segmentation = await artifacts.create_staging(
        db,
        project_id=prediction.project_id,
        kind="segmentation",
        head_slot="segmentation",
        inputs={
            "prediction_artifact_id": str(prediction.id),
            "model_id": prediction.inputs.get("model_id"),
            "min_voxels": min_voxels,
            "label_seq": inputs["label_seq"],
        },
    )
    grants = [
        artifacts.grant_for(prediction, "r"),
        # The label root; blob keys start with "blobs/".
        {"path": f"projects/{prediction.project_id}/labels", "access": "r"},
        artifacts.grant_for(segmentation),
    ]
    common = {
        "project_id": prediction.project_id,
        "created_by": created_by,
        "grants": grants,
    }
    payload = {
        "shape_zyx": shape,
        "min_voxels": min_voxels,
        "model_id": prediction.inputs.get("model_id"),
        "prediction_artifact_id": str(prediction.id),
    }
    # The pinned labels (about 80 bytes a labeled chunk) go to the first job
    # only, which writes them out for the others.
    prepare = await jobs.enqueue(
        db,
        "compose.prepare",
        {**payload, "inputs": inputs},
        weight=WEIGHTS["prepare"],
        **common,
    )
    boxes = shard_boxes(shape, SHARD_ZYX)
    blocks = [
        await jobs.enqueue(
            db,
            "cc.block",
            {**payload, "shard": index, "box": list(box)},
            pipeline=prepare,
            depends_on=[prepare],
            weight=WEIGHTS["blocks"] / len(boxes),
            **common,
        )
        for index, box in enumerate(boxes)
    ]
    merge = await jobs.enqueue(
        db,
        "cc.merge",
        {**payload, "shards": len(boxes)},
        pipeline=prepare,
        depends_on=blocks,
        weight=WEIGHTS["merge"],
        **common,
    )
    applies = [
        await jobs.enqueue(
            db,
            "cc.apply",
            {**payload, "shard": index, "box": list(box)},
            pipeline=prepare,
            depends_on=[merge],
            weight=WEIGHTS["apply"] / len(boxes),
            **common,
        )
        for index, box in enumerate(boxes)
    ]
    finalize = await jobs.enqueue(
        db,
        "compose.finalize",
        {**payload, "shards": len(boxes), "label_seq": inputs["label_seq"]},
        pipeline=prepare,
        depends_on=applies,
        weight=WEIGHTS["finalize"],
        **common,
    )
    segmentation.produced_by_job = finalize.id
    await db.flush()
    return prepare, segmentation
