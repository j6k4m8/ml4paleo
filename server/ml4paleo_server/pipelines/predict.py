"""
Predicting with a model: the project's image through a ready model, into a
prediction artifact that becomes the project's "prediction" head.

    predict.prepare -> predict.shard x N -> prediction.finalize

`prepare` creates the prediction's arrays; each `predict.shard` job predicts
and writes one 512³ shard (they run in parallel, on as many workers as
there are); `finalize` writes the manifest, so its success commits the
artifact.

Starting a prediction cancels the project's other predictions that are
still running, so an older one can't finish later and take the head from
it.
"""

import uuid

from sqlalchemy import exists, not_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from ml4paleo.segmentation.predict import SHARD_ZYX, shard_boxes

from .. import artifacts, jobs
from ..db import Artifact, Job, Project, TrainedModel
from .train import RUNNING_JOB

WEIGHTS = {"prepare": 1.0, "shards": 95.0, "finalize": 1.0}


async def running(
    db: AsyncSession, project_id: uuid.UUID, model_id: uuid.UUID | None = None
) -> list[Job]:
    """
    The first jobs of a project's prediction pipelines that are still
    running (any of their jobs is), or only of those with `model_id`.
    """
    job = aliased(Job)
    query = select(Job).where(
        Job.project_id == project_id,
        Job.id == Job.root_id,
        Job.kind == "predict.prepare",
        not_(Job.cancel_requested),
        exists().where(job.root_id == Job.id, job.status.in_(RUNNING_JOB)),
    )
    if model_id is not None:
        query = query.where(Job.payload.contains({"model_id": str(model_id)}))
    return list((await db.scalars(query.order_by(Job.id))).all())


async def start(
    db: AsyncSession,
    *,
    model: TrainedModel,
    image: Artifact,
    created_by: uuid.UUID,
) -> tuple[Job, Artifact]:
    model_artifact = (
        await db.get(Artifact, model.artifact_id) if model.artifact_id else None
    )
    if model_artifact is None or model_artifact.state != "committed":
        raise ValueError("That model isn't ready.")
    # Predictions in a project start one at a time, so each sees the others.
    await db.scalar(
        select(Project.id)
        .where(Project.id == model.project_id)
        .with_for_update(key_share=True)
    )
    for root in await running(db, model.project_id):
        await jobs.cancel_pipeline(db, root.id)
    assert image.manifest is not None
    _, z, y, x = image.manifest["shape_czyx"]
    shape = (int(z), int(y), int(x))
    # Normalize the image as the model's training crops were; models that
    # don't keep their window get the image's.
    window = (
        (model_artifact.manifest or {}).get("window")
        or image.manifest.get("window")
        or [0, 1]
    )
    prediction = await artifacts.create_staging(
        db,
        project_id=model.project_id,
        kind="prediction",
        head_slot="prediction",
        inputs={
            "model_id": str(model.id),
            "image_artifact_id": str(image.id),
            "window": window,
        },
    )
    grants = [
        artifacts.grant_for(image, "r"),
        artifacts.grant_for(model_artifact, "r"),
        artifacts.grant_for(prediction),
    ]
    common = {
        "project_id": model.project_id,
        "created_by": created_by,
        "grants": grants,
    }
    payload = {
        "model_id": str(model.id),
        "plugin": model.plugin,
        "class_values": list(model.class_values),
        "window": window,
        "shape_zyx": list(shape),
    }
    prepare = await jobs.enqueue(
        db, "predict.prepare", payload, weight=WEIGHTS["prepare"], **common
    )
    boxes = shard_boxes(shape, SHARD_ZYX)
    shards = [
        await jobs.enqueue(
            db,
            "predict.shard",
            {**payload, "box": list(box)},
            pipeline=prepare,
            depends_on=[prepare],
            weight=WEIGHTS["shards"] / len(boxes),
            **common,
        )
        for box in boxes
    ]
    finalize = await jobs.enqueue(
        db,
        "prediction.finalize",
        payload,
        pipeline=prepare,
        depends_on=shards,
        weight=WEIGHTS["finalize"],
        **common,
    )
    prediction.produced_by_job = finalize.id
    await db.flush()
    return prepare, prediction
