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
it, unless one of them is already predicting the same image with the same
model; then the new one is refused.

A proposal (`propose`) predicts just one ROI, on demand and ahead of other
work, as a single `predict.region` job: a prediction artifact in the image's
grid with only that box filled, which becomes the head of the person's own
proposal slot (see `artifacts.proposal_slot`) for them to look over and
accept. Their newer proposal stops their older ones; other people's go on.
"""

import math
import uuid
from collections.abc import Sequence

from sqlalchemy import exists, not_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from ml4paleo.protocol import Tier
from ml4paleo.segmentation.dataset import clip_box
from ml4paleo.segmentation.plugin import get_plugin
from ml4paleo.segmentation.predict import SHARD_ZYX, shard_boxes

from .. import artifacts, jobs
from ..db import Artifact, Job, Project, Roi, TrainedModel
from .train import RUNNING_JOB

WEIGHTS = {"prepare": 1.0, "shards": 95.0, "finalize": 1.0}
# The most a proposal predicts: about as much as one shard job, quickly.
MAX_PROPOSAL_VOXELS = 256**3


async def running(
    db: AsyncSession,
    project_id: uuid.UUID,
    model_id: uuid.UUID | None = None,
    kinds: Sequence[str] = ("predict.prepare",),
    created_by: uuid.UUID | None = None,
) -> list[Job]:
    """
    The first jobs of a project's pipelines starting with one of `kinds`
    (its predictions by default; "predict.region" for its proposals) that
    are still running (any of their jobs is), or only of those with
    `model_id`, or only of those `created_by` someone.
    """
    job = aliased(Job)
    query = select(Job).where(
        Job.project_id == project_id,
        Job.id == Job.root_id,
        Job.kind.in_(kinds),
        not_(Job.cancel_requested),
        exists().where(job.root_id == Job.id, job.status.in_(RUNNING_JOB)),
    )
    if model_id is not None:
        query = query.where(Job.payload.contains({"model_id": str(model_id)}))
    if created_by is not None:
        query = query.where(Job.created_by == created_by)
    return list((await db.scalars(query.order_by(Job.id))).all())


async def start(
    db: AsyncSession,
    *,
    model: TrainedModel,
    image: Artifact,
    created_by: uuid.UUID,
) -> tuple[Job, Artifact]:
    model_artifact = await _start_with(db, model)
    plugin = get_plugin(model.plugin)
    others = await running(db, model.project_id)
    if any(
        root.payload.get("model_id") == str(model.id)
        and root.payload.get("image_artifact_id") == str(image.id)
        for root in others
    ):
        raise ValueError("A prediction with this model is already running.")
    for root in others:
        await jobs.cancel_pipeline(db, root.id)
    shape = _shape(image)
    window = _window(model_artifact, image)
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
        "image_artifact_id": str(image.id),
        "plugin": model.plugin,
        "class_values": list(model.class_values),
        "window": window,
        "shape_zyx": list(shape),
    }
    prepare = await jobs.enqueue(
        db, "predict.prepare", payload, weight=WEIGHTS["prepare"], **common
    )
    boxes = shard_boxes(shape, SHARD_ZYX)
    # Only the shards run the model, so only they need the plugin's GPU.
    shards = [
        await jobs.enqueue(
            db,
            "predict.shard",
            {**payload, "box": list(box)},
            pipeline=prepare,
            depends_on=[prepare],
            weight=WEIGHTS["shards"] / len(boxes),
            min_vram_gb=plugin.caps.min_vram_gb,
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


async def _start_with(db: AsyncSession, model: TrainedModel) -> Artifact:
    """
    Lock things for a prediction or proposal to start with `model`, and
    return the model's files.
    """
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
    # Nor can a model being deleted start one: deleting waits for this share
    # lock, then cancels the model's predictions and proposals still
    # running, this one included.
    alive = await db.scalar(
        select(TrainedModel.id)
        .where(TrainedModel.id == model.id, TrainedModel.deleted_at.is_(None))
        .with_for_update(read=True)
    )
    if alive is None:
        raise ValueError("That model was deleted.")
    return model_artifact


def _shape(image: Artifact) -> tuple[int, int, int]:
    assert image.manifest is not None
    _, z, y, x = image.manifest["shape_czyx"]
    return (int(z), int(y), int(x))


def _window(model_artifact: Artifact, image: Artifact) -> list[float]:
    """
    The window to normalize the image with: the one the model's training
    crops were normalized with, or for models that don't keep it, the image's.
    """
    assert image.manifest is not None
    return (
        (model_artifact.manifest or {}).get("window")
        or image.manifest.get("window")
        or [0, 1]
    )


async def propose(
    db: AsyncSession,
    *,
    model: TrainedModel,
    image: Artifact,
    roi: Roi,
    created_by: uuid.UUID,
) -> tuple[Job, Artifact]:
    """
    Predict one ROI with `model` into a new proposal for `created_by` (see
    the module docstring), stopping their other proposals still running.
    """
    shape = _shape(image)
    clipped = clip_box(roi.bbox, shape)
    if clipped is None:
        raise ValueError("That ROI is outside the image.")
    box = list(clipped)
    if math.prod(box[a + 3] - box[a] for a in range(3)) > MAX_PROPOSAL_VOXELS:
        raise ValueError(
            "Proposals are for ROIs up to 256³ voxels; predict the whole image "
            "on the Models page instead."
        )
    model_artifact = await _start_with(db, model)
    plugin = get_plugin(model.plugin)
    for root in await running(
        db, model.project_id, kinds=["predict.region"], created_by=created_by
    ):
        await jobs.cancel_pipeline(db, root.id)
    window = _window(model_artifact, image)
    proposal = await artifacts.create_staging(
        db,
        project_id=model.project_id,
        kind="prediction",
        head_slot=artifacts.proposal_slot(created_by),
        inputs={
            "model_id": str(model.id),
            "image_artifact_id": str(image.id),
            "window": window,
            "roi_id": str(roi.id),
            "box": box,
        },
    )
    job = await jobs.enqueue(
        db,
        "predict.region",
        {
            "model_id": str(model.id),
            "image_artifact_id": str(image.id),
            "plugin": model.plugin,
            "class_values": list(model.class_values),
            "window": window,
            "shape_zyx": list(shape),
            "box": box,
            "roi_id": str(roi.id),
        },
        project_id=model.project_id,
        created_by=created_by,
        tier=Tier.INTERACTIVE,
        min_vram_gb=plugin.caps.min_vram_gb,
        grants=[
            artifacts.grant_for(image, "r"),
            artifacts.grant_for(model_artifact, "r"),
            artifacts.grant_for(proposal),
        ],
    )
    proposal.produced_by_job = job.id
    await db.flush()
    return job, proposal
