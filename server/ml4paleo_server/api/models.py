"""
Segmentation models.

    GET    /api/plugins                         the installed plugins
    GET    /api/projects/{id}/models
    POST   /api/projects/{id}/models            {plugin, params?, name?}
    GET    /api/projects/{id}/models/{model}
    DELETE /api/projects/{id}/models/{model}
    POST   /api/projects/{id}/models/{model}/predict
    POST   /api/projects/{id}/models/{model}/propose  {roi_id}
    GET    /api/projects/{id}/prediction           the prediction of the current image
    GET    /api/projects/{id}/proposal             your newest proposal (one ROI predicted)

Training pins the project's labels and ROIs as a training set and starts a
`model.train` pipeline (follow it under /pipelines). A model is "training"
until its job finishes, then "ready" or "failed". Each model training or
kept counts against the project owner's trained-model quota; deleting one
frees its slot.
"""

import dataclasses
import datetime
import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator
from sqlalchemy import select

from ml4paleo.protocol import json_text
from ml4paleo.segmentation.plugin import get_plugin, plugins

from .. import artifacts, audit, jobs, pipelines, quotas, training
from ..auth.deps import CurrentAuth, DbSession, SettingsDep
from ..db import Artifact, Job, Project, Roi, TrainedModel, TrainingSet, User
from ..pipelines import predict, train
from .gateway import zarr_path
from .projects import MemberProject

router = APIRouter(prefix="/api", tags=["models"])


class PluginOut(BaseModel):
    name: str
    version: str
    devices: list[str]
    params_schema: dict[str, Any]
    capabilities: dict[str, Any]


@router.get("/plugins")
async def list_plugins(auth: CurrentAuth) -> list[PluginOut]:
    return [
        PluginOut(
            name=plugin.name,
            version=plugin.version,
            devices=list(plugin.caps.devices),
            params_schema=plugin.Params.model_json_schema(),
            capabilities=dataclasses.asdict(plugin.caps),
        )
        for plugin in plugins().values()
    ]


class TrainIn(BaseModel):
    plugin: str = "rf"
    params: dict[str, Any] = {}
    name: str | None = Field(default=None, max_length=100)
    live: bool = False

    @field_validator("params")
    @classmethod
    def _finite(cls, params: dict[str, Any]) -> dict[str, Any]:
        # A plugin's own fields bound what it takes, but what it keeps of them
        # is stored as JSON, which can't hold NaN or infinity.
        json_text(params, "params")
        return params


class ModelOut(BaseModel):
    id: uuid.UUID
    name: str
    plugin: str
    plugin_version: str | None
    params: dict[str, Any]
    status: str
    # Why a failed training failed, in a sentence (if it says).
    error: str | None = None
    class_values: list[int]
    metrics: dict[str, Any] | None
    # The training pipeline (see /pipelines/{id}).
    pipeline_id: uuid.UUID | None
    training_set: dict[str, Any]
    created_at: datetime.datetime
    live: bool = False
    created_by: uuid.UUID | None = None
    # Only on a live train request: whether this checkpoint pins the snapshot
    # just requested (including returning to an older state through undo).
    live_current: bool | None = None


async def _model_out(db, model: TrainedModel) -> ModelOut:
    job_status = (
        await db.scalar(select(Job.status).where(Job.id == model.job_id))
        if model.job_id
        else None
    )
    artifact = await db.get(Artifact, model.artifact_id) if model.artifact_id else None
    training_set = await db.get(TrainingSet, model.training_set_id)
    status = train.model_status(model, job_status, artifact.state if artifact else None)
    error = None
    if status == "failed" and model.job_id:
        error = await pipelines.failure(db, model.job_id)
        if error is None and job_status == "cancelled":
            error = "Stopped."
    return ModelOut(
        id=model.id,
        name=model.name,
        plugin=model.plugin,
        plugin_version=model.plugin_version,
        params=model.params,
        status=status,
        error=error,
        class_values=list(model.class_values),
        metrics=model.metrics,
        pipeline_id=model.job_id,
        training_set={
            "id": model.training_set_id,
            **(training_set.summary if training_set else {}),
        },
        created_at=model.created_at,
        live=bool(artifact and artifact.inputs.get("live")),
        created_by=model.created_by,
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
    await train.release_failed_slots(db, project.owner_id, project.id)
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
        # Answered as the rest of the body's errors are, which are safe to send
        # whatever the request held (see `app.invalid_request`).
        raise RequestValidationError(
            [
                {**error, "loc": ("body", "params", *error["loc"])}
                for error in exc.errors(include_url=False)
            ]
        ) from None
    # Look for a free model slot before pinning a training set, so a refused
    # training stores nothing; `train.start` reserves the slot. Commit the
    # slots of failed trainings at once, so the owner's usage row isn't
    # locked while the snapshot is taken.
    owner = await db.get(User, project.owner_id)
    assert owner is not None
    await train.release_failed_slots(db, owner.id)
    await db.commit()
    if not body.live:
        await quotas.check_trained_model(db, settings, owner)
    try:
        training_set = await training.snapshot(
            db, request.app.state.sessionmaker, settings, project.id
        )
    except training.NotReady as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    if body.live:
        if plugin.caps.learning == "manual":
            raise HTTPException(
                status_code=422,
                detail="This model requires explicit training on the Models page.",
            )
        # Serialize live requests across tabs. Keep one ready checkpoint and
        # at most one training, without touching manually saved models.
        await db.scalar(
            select(Project.id)
            .where(Project.id == project.id)
            .with_for_update(key_share=True)
        )
        previous = (
            await db.scalars(
                select(TrainedModel)
                .join(Artifact, Artifact.id == TrainedModel.artifact_id)
                .where(
                    TrainedModel.project_id == project.id,
                    TrainedModel.created_by == auth.user.id,
                    TrainedModel.deleted_at.is_(None),
                    Artifact.inputs.contains({"live": True}),
                )
                .order_by(TrainedModel.created_at.desc())
            )
        ).all()
        ready = None
        for old in previous:
            out = await _model_out(db, old)
            out.live_current = (
                old.training_set_id == training_set.id
                and old.plugin == plugin.name
                and old.params == params.model_dump()
            )
            if out.status == "training":
                await db.commit()
                return out
            if old.plugin == plugin.name and old.params == params.model_dump():
                # A failed snapshot is not retried in a tight loop either.
                if (
                    old.training_set_id == training_set.id
                    or (artifacts.now() - old.created_at).total_seconds() * 1000
                    < plugin.caps.min_train_interval_ms
                ):
                    await db.commit()
                    return out
            if out.status == "ready" and ready is None:
                ready = old.id
        for old in previous:
            if old.id == ready:
                continue
            old.deleted_at = artifacts.now()
            artifact = await db.get(Artifact, old.artifact_id)
            if artifact:
                artifact.expires_at = artifacts.now()
            await train.release_slots(db, owner.id, TrainedModel.id == old.id)
    count = len(
        (
            await db.scalars(
                select(TrainedModel.id).where(TrainedModel.project_id == project.id)
            )
        ).all()
    )
    name = body.name or (
        f"Live {plugin.caps.display_name}"
        if body.live
        else f"{plugin.name} model {count + 1}"
    )
    _, model = await train.start(
        db,
        settings,
        project=project,
        training_set=training_set,
        plugin=plugin,
        params=params,
        name=name,
        created_by=auth.user.id,
        live=body.live,
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
    out = await _model_out(db, model)
    out.live_current = True if body.live else None
    return out


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
    Delete a model: stop its training, predictions, and proposals if they're
    still running, free its slot, and let garbage collection remove its files.
    """
    model = await _model(db, project, model_id)
    model.deleted_at = datetime.datetime.now(datetime.UTC)
    # Waits for a prediction or proposal being started with it, so the
    # search below finds (and stops) that one too.
    await db.flush()
    if model.job_id:
        job = await db.get(Job, model.job_id)
        if job is not None and job.status in train.RUNNING_JOB:
            await jobs.cancel_pipeline(db, job.root_id)
    for root in await predict.running(
        db,
        project.id,
        model.id,
        kinds=["predict.prepare", "predict.region", "predict.live"],
    ):
        await jobs.cancel_pipeline(db, root.id)
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


class PredictionStarted(BaseModel):
    pipeline_id: uuid.UUID
    artifact_id: uuid.UUID


class LiveChunkIn(BaseModel):
    image_artifact_id: uuid.UUID
    key: tuple[int, int, int]


class LiveChunkOut(BaseModel):
    artifact_id: uuid.UUID
    pipeline_id: uuid.UUID | None
    ready: bool
    box: list[int]
    zarr_url: str


@router.post("/projects/{project_id}/models/{model_id}/live-chunk")
async def live_chunk(
    model_id: uuid.UUID,
    body: LiveChunkIn,
    project: MemberProject,
    auth: CurrentAuth,
    db: DbSession,
) -> LiveChunkOut:
    model = await _model(db, project, model_id)
    image = await artifacts.head(db, project.id, "image")
    if image is None or image.id != body.image_artifact_id:
        raise HTTPException(
            status_code=409, detail="The image changed; reload the annotator."
        )
    try:
        result = await predict.live_chunk(
            db, model=model, image=image, key=body.key, created_by=auth.user.id
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    await db.commit()
    return LiveChunkOut(
        artifact_id=result.id,
        pipeline_id=result.produced_by_job,
        ready=result.state == "committed",
        box=result.inputs["box"],
        zarr_url=zarr_path(project.id, result.id),
    )


@router.post("/projects/{project_id}/models/{model_id}/predict", status_code=202)
async def predict_with(
    model_id: uuid.UUID,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> PredictionStarted:
    """
    Run a ready model over the project's image; the result becomes the
    project's prediction when the pipeline succeeds. Other predictions still
    running are cancelled, and if this model is already predicting this
    image, the answer is 409.
    """
    model = await _model(db, project, model_id)
    image = await artifacts.head(db, project.id, "image")
    if image is None or not image.manifest:
        raise HTTPException(status_code=409, detail="This project has no image yet.")
    try:
        root, artifact = await predict.start(
            db, model=model, image=image, created_by=auth.user.id
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    audit.record(
        db,
        actor_id=auth.user.id,
        action="model.predict",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"model_id": str(model.id), "pipeline_id": str(root.id)},
    )
    await db.commit()
    return PredictionStarted(pipeline_id=root.id, artifact_id=artifact.id)


class ProposeIn(BaseModel):
    roi_id: uuid.UUID


@router.post("/projects/{project_id}/models/{model_id}/propose", status_code=202)
async def propose_with(
    model_id: uuid.UUID,
    body: ProposeIn,
    project: MemberProject,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
) -> PredictionStarted:
    """
    Predict one ROI (up to 256³ voxels) with a ready model, ahead of other
    work; the result becomes your proposal to look over and accept. Your
    proposals still running stop; other people's go on.
    """
    model = await _model(db, project, model_id)
    image = await artifacts.head(db, project.id, "image")
    if image is None or not image.manifest:
        raise HTTPException(status_code=409, detail="This project has no image yet.")
    roi = await db.scalar(
        select(Roi).where(Roi.id == body.roi_id, Roi.project_id == project.id)
    )
    if roi is None:
        raise HTTPException(status_code=404, detail="No such ROI.")
    try:
        job, artifact = await predict.propose(
            db, model=model, image=image, roi=roi, created_by=auth.user.id
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from None
    audit.record(
        db,
        actor_id=auth.user.id,
        action="model.propose",
        target_type="project",
        target_id=project.id,
        request=request,
        details={"model_id": str(model.id), "roi_id": str(roi.id)},
    )
    await db.commit()
    return PredictionStarted(pipeline_id=job.id, artifact_id=artifact.id)


class PredictionOut(BaseModel):
    artifact_id: uuid.UUID
    model_id: uuid.UUID | None
    model_name: str | None
    class_values: list[int]
    # The image it was predicted from (always the current one), and its
    # (z, y, x) shape.
    image_artifact_id: uuid.UUID
    shape_zyx: list[int]
    # The prediction's zarr group (arrays `class` and `uncertainty`), through
    # the data gateway.
    zarr_url: str
    # When it was asked for, and when it was done.
    started_at: datetime.datetime
    committed_at: datetime.datetime


@router.get("/projects/{project_id}/prediction")
async def current_prediction(project: MemberProject, db: DbSession) -> PredictionOut:
    """
    The project's prediction, if it is of the current image (a prediction of
    an image that has since been replaced doesn't fit the new one).
    """
    head = await _current(db, project.id, "prediction", "prediction")
    return await _prediction_out(db, project.id, head, PredictionOut)


class ProposalOut(PredictionOut):
    roi_id: uuid.UUID | None
    # The box it covers, (z0, y0, x0, z1, y1, x1); it's empty elsewhere.
    box: list[int]


@router.get("/projects/{project_id}/proposal")
async def current_proposal(
    project: MemberProject, auth: CurrentAuth, db: DbSession
) -> ProposalOut:
    """
    Your newest proposal (one ROI predicted on demand) in the project, if it
    is of the current image. Each person has their own.
    """
    slot = artifacts.proposal_slot(auth.user.id)
    head = await _current(db, project.id, slot, "proposal")
    return await _prediction_out(
        db,
        project.id,
        head,
        ProposalOut,
        roi_id=head.inputs.get("roi_id"),
        box=list(head.inputs.get("box", [])),
    )


async def _current(db, project_id: uuid.UUID, slot: str, what: str) -> Artifact:
    """The head of `slot`, if it is of the current image; `what` names it."""
    head = await artifacts.head(db, project_id, slot)
    if head is None or not head.manifest:
        raise HTTPException(status_code=404, detail=f"This project has no {what} yet.")
    image = await artifacts.head(db, project_id, "image")
    if image is None or head.inputs.get("image_artifact_id") != str(image.id):
        raise HTTPException(
            status_code=404, detail=f"This project has no {what} of its image."
        )
    return head


async def _prediction_out(db, project_id: uuid.UUID, head: Artifact, out, **extra):
    assert head.manifest is not None
    model_id = head.inputs.get("model_id")
    model = (
        await db.get(TrainedModel, uuid.UUID(model_id))
        if model_id is not None
        else None
    )
    return out(
        artifact_id=head.id,
        model_id=model.id if model else None,
        model_name=model.name if model else None,
        class_values=list(head.manifest.get("class_values", [])),
        image_artifact_id=head.inputs["image_artifact_id"],
        shape_zyx=list(head.manifest.get("shape_zyx", [])),
        zarr_url=zarr_path(project_id, head.id),
        started_at=head.created_at,
        committed_at=head.state_changed_at,
        **extra,
    )
