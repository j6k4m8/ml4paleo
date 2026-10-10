"""Live keeps immutable provenance without unbounded training/inference queues."""

import datetime
import uuid

import numpy as np
from helpers import run_db
from ml4paleo_server.db import Artifact, Job, TrainedModel
from sqlalchemy import select
from test_models import ada as ada
from test_models import (
    add_image,
    labeled_project,
    paint,
    ready_model,
)
from test_models import settings as settings


def finish_model(database_url, model, status="succeeded"):
    async def finish(db):
        row = await db.get(TrainedModel, uuid.UUID(model["id"]))
        job = await db.get(Job, row.job_id)
        job.status = status
        artifact = await db.get(Artifact, row.artifact_id)
        artifact.state = "committed" if status == "succeeded" else "failed"
        artifact.manifest = {"kind": "model", "window": [200, 800]}
        row.created_at -= datetime.timedelta(minutes=1)

    run_db(database_url, finish)


def test_plugins_declare_learning_semantics(ada):
    [plugin] = ada.get("/api/plugins").json()
    assert plugin["name"] == "rf"
    assert plugin["capabilities"]["display_name"] == "Random forest"
    assert plugin["capabilities"]["learning"] == "debounced"
    assert plugin["capabilities"]["family"] == "classical"


def test_live_coalesces_training_and_keeps_only_two_checkpoints(
    ada, settings, migrated_database_url
):
    project = labeled_project(ada, settings, migrated_database_url)
    base = f"/api/projects/{project}/models"
    saved = ready_model(ada, migrated_database_url, project)
    first = ada.post(base, json={"live": True}).json()
    assert first["live"] and first["live_current"]
    paint(settings, migrated_database_url, project, (10, 10, 10), np.ones((1, 2, 2)), 1)
    waiting = ada.post(base, json={"live": True}).json()
    assert waiting["id"] == first["id"] and not waiting["live_current"]
    finish_model(migrated_database_url, first)
    second = ada.post(base, json={"live": True}).json()
    assert second["id"] != first["id"]
    finish_model(migrated_database_url, second)
    # Unchanged annotations reuse a checkpoint rather than charging another slot.
    assert ada.post(base, json={"live": True}).json()["id"] == second["id"]
    paint(settings, migrated_database_url, project, (12, 12, 12), np.ones((1, 2, 2)), 1)
    third = ada.post(base, json={"live": True}).json()
    models = ada.get(base).json()
    assert {m["id"] for m in models} == {saved["id"], second["id"], third["id"]}
    assert ada.get("/api/me/quota").json()["trained_models_used"] == 3


def test_live_failed_snapshot_is_not_retried_forever(
    ada, settings, migrated_database_url
):
    project = labeled_project(ada, settings, migrated_database_url)
    base = f"/api/projects/{project}/models"
    first = ada.post(base, json={"live": True}).json()
    finish_model(migrated_database_url, first, "failed")
    again = ada.post(base, json={"live": True}).json()
    assert again["id"] == first["id"] and again["status"] == "failed"
    assert ada.get("/api/me/quota").json()["trained_models_used"] == 0


def test_live_training_throttle_preserves_pending_intent(
    ada, settings, migrated_database_url
):
    project = labeled_project(ada, settings, migrated_database_url)
    base = f"/api/projects/{project}/models"
    first = ada.post(base, json={"live": True}).json()
    finish_model(migrated_database_url, first)

    async def just_finished(db):
        row = await db.get(TrainedModel, uuid.UUID(first["id"]))
        row.created_at = datetime.datetime.now(datetime.UTC)

    run_db(migrated_database_url, just_finished)
    paint(settings, migrated_database_url, project, (10, 10, 10), np.ones((1, 2, 2)), 1)
    throttled = ada.post(base, json={"live": True}).json()
    assert throttled["id"] == first["id"] and not throttled["live_current"]


def test_live_chunk_is_private_to_no_head_bounded_and_cacheable(
    ada, settings, migrated_database_url
):
    project = labeled_project(ada, settings, migrated_database_url)
    model = ready_model(ada, migrated_database_url, project)
    other = ready_model(ada, migrated_database_url, project)
    base = f"/api/projects/{project}/models/{model['id']}/live-chunk"
    body = {
        "image_artifact_id": model["training_set"]["image_artifact_id"],
        "key": [0, 0, 0],
    }
    response = ada.post(base, json=body)
    assert response.status_code == 200, response.text
    first = response.json()
    assert first["box"] == [0, 0, 0, 40, 48, 56] and not first["ready"]
    assert ada.post(base, json=body).json()["artifact_id"] == first["artifact_id"]
    # Another checkpoint cannot queue behind this one in a second tab.
    assert (
        ada.post(
            f"/api/projects/{project}/models/{other['id']}/live-chunk", json=body
        ).status_code
        == 409
    )
    assert ada.post(base, json={**body, "key": [-1, 0, 0]}).status_code == 409
    assert ada.post(base, json={**body, "key": [1, 0, 0]}).status_code == 409
    assert ada.get(f"/api/projects/{project}/prediction").status_code == 404
    assert ada.get(f"/api/projects/{project}/proposal").status_code == 404

    async def finish(db):
        artifact = await db.get(Artifact, uuid.UUID(first["artifact_id"]))
        assert artifact.head_slot is None and artifact.expires_at is not None
        artifact.state = "committed"
        artifact.manifest = {"class_values": [2], "shape_zyx": [40, 48, 56]}
        job = await db.get(Job, artifact.produced_by_job)
        assert job.kind == "predict.live" and job.tier == 0
        job.status = "succeeded"

    run_db(migrated_database_url, finish)
    cached = ada.post(base, json=body).json()
    assert cached["ready"] and cached["artifact_id"] == first["artifact_id"]
    add_image(settings, migrated_database_url, project)
    assert ada.post(base, json=body).status_code == 409


def test_deleting_model_cancels_live_chunks(ada, settings, migrated_database_url):
    project = labeled_project(ada, settings, migrated_database_url)
    model = ready_model(ada, migrated_database_url, project)
    base = f"/api/projects/{project}/models/{model['id']}"
    chunk = ada.post(
        f"{base}/live-chunk",
        json={
            "image_artifact_id": model["training_set"]["image_artifact_id"],
            "key": [0, 0, 0],
        },
    ).json()
    assert ada.request("DELETE", base).status_code == 204

    async def check(db):
        job = await db.scalar(
            select(Job).where(Job.id == uuid.UUID(chunk["pipeline_id"]))
        )
        assert job.status == "cancelled"

    run_db(migrated_database_url, check)
