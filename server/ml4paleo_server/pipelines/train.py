"""
Training a segmentation model: one job, `model.train`, on a worker that has
the plugin, reading a training set's manifest, the image, and the label
blobs, and writing the model's files into a model artifact, which the job's
success commits.

A model holds one of the project owner's trained-model slots from the
moment training starts; a failed or cancelled training gives it back (see
`release_failed_slots`), as does deleting the model or its project. Slots
are only ever
given back through `release_slots`, which clears `holds_slot` and counts
what it cleared in one UPDATE, so no slot is given back twice.
"""

import uuid
from typing import Any

from pydantic import BaseModel
from sqlalchemy import ColumnElement, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ml4paleo.segmentation.plugin import SegmentationPlugin

from .. import artifacts, jobs, quotas
from ..db import Artifact, Job, Project, TrainedModel, TrainingSet, User
from ..settings import Settings
from ..training import training_path

FAILED_JOB = ("failed", "cancelled")
RUNNING_JOB = ("blocked", "queued", "leased")


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


async def release_slots(
    db: AsyncSession, owner_id: uuid.UUID, *conditions: ColumnElement[bool]
) -> int:
    """
    Give back the trained-model slots that models matching `conditions` (all
    in projects `owner_id` owns) hold, and say how many that was.

    The UPDATE only takes rows that still hold a slot, and locks them, so of
    two transactions releasing the same model only one gets it back.
    """
    released = (
        await db.scalars(
            update(TrainedModel)
            .where(TrainedModel.holds_slot, *conditions)
            .values(holds_slot=False)
            .returning(TrainedModel.id)
        )
    ).all()
    if released:
        await quotas.release_trained_model(db, owner_id, len(released))
    return len(released)


async def release_failed_slots(
    db: AsyncSession, owner_id: uuid.UUID, project_id: uuid.UUID | None = None
) -> None:
    """
    Give back the trained-model slots of the owner's trainings that failed,
    in all of their projects, or only in `project_id`.
    """
    failed = (
        select(TrainedModel.id)
        .join(Project, Project.id == TrainedModel.project_id)
        .outerjoin(Job, Job.id == TrainedModel.job_id)
        .where(
            Project.owner_id == owner_id,
            (Job.status.in_(FAILED_JOB)) | (TrainedModel.job_id.is_(None)),
        )
    )
    if project_id is not None:
        failed = failed.where(TrainedModel.project_id == project_id)
    await release_slots(db, owner_id, TrainedModel.id.in_(failed))


async def stop_project(db: AsyncSession, project: Project) -> None:
    """
    For a project being deleted: cancel its trainings that are still running
    and give back the trained-model slots its models hold.
    """
    roots = (
        await db.scalars(
            select(Job.root_id)
            .join(TrainedModel, TrainedModel.job_id == Job.id)
            .where(TrainedModel.project_id == project.id, Job.status.in_(RUNNING_JOB))
        )
    ).all()
    for root_id in sorted(set(roots)):
        await jobs.cancel_pipeline(db, root_id)
    await release_slots(db, project.owner_id, TrainedModel.project_id == project.id)


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
    # Failed trainings anywhere in the owner's projects free their slots.
    await release_failed_slots(db, owner.id)
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


async def after_train(db: AsyncSession, job: Job) -> None:
    model = await db.scalar(select(TrainedModel).where(TrainedModel.job_id == job.id))
    if model is None:
        return
    result = job.result or {}
    model.metrics = result.get("metrics")
    model.plugin_version = result.get("plugin_version")
    await db.flush()
