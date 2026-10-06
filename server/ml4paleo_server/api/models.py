"""
Segmentation models.

    GET    /api/plugins                         the installed plugins
    GET    /api/projects/{id}/models
    POST   /api/projects/{id}/models            {plugin, params?, name?}
    GET    /api/projects/{id}/models/{model}
    DELETE /api/projects/{id}/models/{model}

Training pins the project's labels and ROIs as a training set and starts a
`model.train` pipeline (follow it under /pipelines). A model is "training"
until its job finishes, then "ready" or "failed". Each model training or
kept counts against the project owner's trained-model quota; deleting one
frees its slot.
"""

import datetime
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select

from ml4paleo.segmentation.plugin import get_plugin, plugins

from .. import audit, jobs, training
from ..auth.deps import CurrentAuth, DbSession, SettingsDep
from ..db import Artifact, Job, Project, TrainedModel, TrainingSet
from ..pipelines import train
from .projects import MemberProject

router = APIRouter(prefix="/api", tags=["models"])


class PluginOut(BaseModel):
    name: str
    version: str
    devices: list[str]
    params_schema: dict[str, Any]


@router.get("/plugins")
async def list_plugins(auth: CurrentAuth) -> list[PluginOut]:
    return [
        PluginOut(
            name=plugin.name,
            version=plugin.version,
            devices=list(plugin.caps.devices),
            params_schema=plugin.Params.model_json_schema(),
        )
        for plugin in plugins().values()
    ]


class TrainIn(BaseModel):
    plugin: str = "rf"
    params: dict[str, Any] = {}
    name: str | None = Field(default=None, max_length=100)


class ModelOut(BaseModel):
    id: uuid.UUID
    name: str
    plugin: str
    plugin_version: str | None
    params: dict[str, Any]
    status: str
    class_values: list[int]
    metrics: dict[str, Any] | None
    # The training pipeline (see /pipelines/{id}).
    pipeline_id: uuid.UUID | None
    training_set: dict[str, Any]
    created_at: datetime.datetime


async def _model_out(db, model: TrainedModel) -> ModelOut:
    job_status = (
        await db.scalar(select(Job.status).where(Job.id == model.job_id))
        if model.job_id
        else None
    )
    artifact_state = (
        await db.scalar(select(Artifact.state).where(Artifact.id == model.artifact_id))
        if model.artifact_id
        else None
    )
    training_set = await db.get(TrainingSet, model.training_set_id)
    return ModelOut(
        id=model.id,
        name=model.name,
        plugin=model.plugin,
        plugin_version=model.plugin_version,
        params=model.params,
        status=train.model_status(model, job_status, artifact_state),
        class_values=list(model.class_values),
        metrics=model.metrics,
        pipeline_id=model.job_id,
        training_set={
            "id": model.training_set_id,
            **(training_set.summary if training_set else {}),
        },
        created_at=model.created_at,
    )


async def _model(db, project: Project, model_id: uuid.UUID) -> TrainedModel:
    model = await db.scalar(
        select(TrainedModel).where(
            TrainedModel.id == model_id,
            TrainedModel.project_id == project.id,
            TrainedModel.deleted_at.is_(None),
        )
    )
    if model is None:
        raise HTTPException(status_code=404, detail="No such model.")
    return model


@router.get("/projects/{project_id}/models")
async def list_models(project: MemberProject, db: DbSession) -> list[ModelOut]:
    await train.release_failed_slots(db, project)
    await db.commit()
    models = (
        await db.scalars(
            select(TrainedModel)
            .where(
                TrainedModel.project_id == project.id, TrainedModel.deleted_at.is_(None)
            )
            .order_by(TrainedModel.created_at.desc())
        )
    ).all()
    return [await _model_out(db, model) for model in models]


@router.post("/projects/{project_id}/models", status_code=202)
async def train_model(
    body: TrainIn,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
    settings: SettingsDep,
) -> ModelOut:
    try:
        plugin = get_plugin(body.plugin)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    try:
        params = plugin.Params(**body.params)
    except ValidationError as exc:
        raise HTTPException(
            status_code=422, detail=exc.errors(include_url=False)
        ) from None
    try:
        training_set = await training.snapshot(
            db, request.app.state.sessionmaker, settings, project.id
        )
    except training.NotReady as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    count = len(
        (
            await db.scalars(
                select(TrainedModel.id).where(TrainedModel.project_id == project.id)
            )
        ).all()
    )
    name = body.name or f"{plugin.name} model {count + 1}"
    _, model = await train.start(
        db,
        settings,
        project=project,
        training_set=training_set,
        plugin=plugin,
        params=params,
        name=name,
        created_by=auth.user.id,
    )
    audit.record(
        db,
        actor_id=auth.user.id,
        action="model.train",
        target_type="project",
        target_id=project.id,
        request=request,
        details={
            "model_id": str(model.id),
            "plugin": plugin.name,
            "training_set": training_set.id,
        },
    )
    await db.commit()
    return await _model_out(db, model)


@router.get("/projects/{project_id}/models/{model_id}")
async def get_model(
    model_id: uuid.UUID, project: MemberProject, db: DbSession
) -> ModelOut:
    return await _model_out(db, await _model(db, project, model_id))


@router.delete("/projects/{project_id}/models/{model_id}", status_code=204)
async def delete_model(
    model_id: uuid.UUID,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> None:
    """
    Delete a model: stop its training if it's still running, free its slot,
    and let garbage collection remove its files.
    """
    model = await _model(db, project, model_id)
    model.deleted_at = datetime.datetime.now(datetime.UTC)
    if model.job_id:
        job = await db.get(Job, model.job_id)
        if job is not None and job.status in train.RUNNING_JOB:
            await jobs.cancel_pipeline(db, job.root_id)
    if model.artifact_id:
        artifact = await db.get(Artifact, model.artifact_id)
        if artifact is not None and artifact.state == "committed":
            artifact.expires_at = datetime.datetime.now(datetime.UTC)
    await train.release_slots(db, project.owner_id, TrainedModel.id == model.id)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="model.delete",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"model_id": str(model.id)},
    )
    await db.commit()
