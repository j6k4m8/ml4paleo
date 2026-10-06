"""
Training a segmentation model: one job, `model.train`, on a worker that has
the plugin, reading a training set's manifest, the image, and the label
blobs, and writing the model's files into a model artifact, which the job's
success commits.

A model holds one of the project owner's trained-model slots from the
moment training starts; a failed or cancelled training gives it back (see
`release_failed_slots`), as does deleting the model.
"""

import uuid
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ml4paleo.segmentation.plugin import SegmentationPlugin

from .. import artifacts, jobs, quotas
from ..db import Artifact, Job, Project, TrainedModel, TrainingSet, User
from ..settings import Settings
from ..training import training_path

FAILED_JOB = ("failed", "cancelled")


def model_status(
    model: TrainedModel, job_status: str | None, artifact_state: str | None
) -> str:
    if model.deleted_at is not None:
        return "deleted"
    if artifact_state == "committed":
        return "ready"
    if job_status in FAILED_JOB or artifact_state in ("failed", "deleting", "deleted"):
        return "failed"
    return "training"


async def release_failed_slots(db: AsyncSession, project_id: uuid.UUID) -> None:
    """
    Give back the trained-model slots of the project's trainings that failed.
    """
    rows = (
        await db.execute(
            select(TrainedModel, Project.owner_id)
            .join(Project, Project.id == TrainedModel.project_id)
            .outerjoin(Job, Job.id == TrainedModel.job_id)
            .where(
                TrainedModel.project_id == project_id,
                TrainedModel.holds_slot,
                (Job.status.in_(FAILED_JOB)) | (TrainedModel.job_id.is_(None)),
            )
            .with_for_update(of=TrainedModel)
        )
    ).all()
    for model, owner_id in rows:
        model.holds_slot = False
        await quotas.release_trained_model(db, owner_id)


async def start(
    db: AsyncSession,
    settings: Settings,
    *,
    project: Project,
    training_set: TrainingSet,
    plugin: type[SegmentationPlugin],
    params: BaseModel,
    name: str,
    created_by: uuid.UUID,
) -> tuple[Job, TrainedModel]:
    owner = await db.get(User, project.owner_id)
    assert owner is not None
    await release_failed_slots(db, project.id)
    await quotas.reserve_trained_model(db, settings, owner)
    image = await db.get(Artifact, uuid.UUID(training_set.summary["image_artifact_id"]))
    assert image is not None
    artifact = await artifacts.create_staging(
        db,
        project_id=project.id,
        kind="model",
        inputs={"training_set": training_set.id, "plugin": plugin.name},
    )
    model = TrainedModel(
        project_id=project.id,
        name=name,
        plugin=plugin.name,
        params=params.model_dump(),
        training_set_id=training_set.id,
        class_values=training_set.summary["class_values"],
        artifact_id=artifact.id,
        holds_slot=True,
        created_by=created_by,
    )
    db.add(model)
    await db.flush()
    job = await jobs.enqueue(
        db,
        "model.train",
        {
            "model_id": str(model.id),
            "plugin": plugin.name,
            "params": params.model_dump(),
            "training_set": training_set.id,
        },
        project_id=project.id,
        created_by=created_by,
        grants=[
            artifacts.grant_for(image, "r"),
            # The label root; blob keys start with "blobs/".
            {"path": f"projects/{project.id}/labels", "access": "r"},
            {"path": training_path(project.id, training_set.id), "access": "r"},
            artifacts.grant_for(artifact),
        ],
        min_vram_gb=plugin.caps.min_vram_gb,
    )
    artifact.produced_by_job = job.id
    model.job_id = job.id
    await db.flush()
    return job, model


def check_result(result: dict[str, Any]) -> None:
    if not isinstance(result.get("metrics"), dict):
        raise ValueError("metrics must be an object")
    if not isinstance(result.get("plugin_version"), str):
        raise ValueError("plugin_version must be a string")


async def after_train(db: AsyncSession, settings: Settings, job: Job) -> None:
    model = await db.scalar(select(TrainedModel).where(TrainedModel.job_id == job.id))
    if model is None:
        return
    result = job.result or {}
    model.metrics = result.get("metrics")
    model.plugin_version = result.get("plugin_version")
    await db.flush()
