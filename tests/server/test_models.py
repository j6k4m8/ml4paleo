"""
Models: training sets pinned from labels and ROIs, the models API and its
quota, and a random forest trained end to end by a worker.
"""

import asyncio
import json
import threading
import time
import uuid

import numpy as np
import pytest
from helpers import SECRET_KEY, add_worker, run_db, signup
from ml4paleo_server import artifacts, labels
from ml4paleo_server.db import (
    Artifact,
    Job,
    Project,
    TrainedModel,
    create_engine,
    create_sessionmaker,
)
from ml4paleo_server.pipelines import train
from ml4paleo_server.settings import Settings
from ml4paleo_server.storage import project_storage
from ml4paleo_server.training import training_path
from ml4paleo_worker.client import ServerClient
from ml4paleo_worker.handlers import HANDLERS
from ml4paleo_worker.main import Worker
from sqlalchemy import update

from ml4paleo.labels.deltas import split_into_deltas
from ml4paleo.ome import OmeImage
from ml4paleo.protocol import WorkerCaps
from ml4paleo.storage import get_bytes

SHAPE = (40, 48, 56)
BONE = 2


def ball_image():
    rng = np.random.default_rng(0)
    z, y, x = np.indices(SHAPE)
    truth = (z - 20) ** 2 + (y - 24) ** 2 + (x - 28) ** 2 <= 12**2
    image = np.where(truth, 800, 200) + rng.normal(0, 40, SHAPE)
    return image.astype(np.uint16), truth


def add_image(settings, database_url, project: str, with_data: bool = False):
    async def create(db):
        artifact = await artifacts.create_staging(
            db, project_id=uuid.UUID(project), kind="image", head_slot="image"
        )
        if with_data:
            image = OmeImage.create(
                project_storage(settings).child(artifacts.artifact_path(artifact)),
                shape_czyx=(1, *SHAPE),
                dtype=np.uint16,
            )
            image.array(0)[0] = ball_image()[0]
        artifact.state = "committed"
        artifact.manifest = {"shape_czyx": [1, *SHAPE], "window": [200, 800]}
        await artifacts.set_head(db, artifact)

    run_db(database_url, create)


def paint(settings, database_url, project: str, origin, mask, value):
    async def apply(db):
        await labels.apply_edit(
            db,
            settings,
            uuid.UUID(project),
            client_op_id=uuid.uuid4(),
            deltas=split_into_deltas(np.asarray(mask, dtype=bool), origin, value=value),
        )

    run_db(database_url, apply)


@pytest.fixture
def ada(new_browser):
    browser = new_browser()
    signup(browser)
    return browser


def make_project(browser) -> str:
    return browser.post("/api/projects", json={"name": "Skull"}).json()["id"]


def add_class(browser, project):
    return browser.post(
        f"/api/projects/{project}/labels/classes",
        json={"name": "bone", "color": "#ffffff"},
    ).json()["value"]


def test_training_sets_pin_labels_and_rois(ada, settings, migrated_database_url):
    project = make_project(ada)
    base = f"/api/projects/{project}/models"
    assert ada.post(base, json={}).status_code == 409  # no image
    add_image(settings, migrated_database_url, project)
    assert "label class" in ada.post(base, json={}).json()["detail"]
    add_class(ada, project)
    assert "Label something" in ada.post(base, json={}).json()["detail"]
    paint(settings, migrated_database_url, project, (5, 5, 5), np.ones((1, 3, 3)), BONE)
    ada.post(
        f"/api/projects/{project}/rois",
        json={"bbox": [0, 0, 0, 8, 8, 8], "kind": "cube", "split": "val"},
    )
    assert ada.post(base, json={"plugin": "nope"}).status_code == 422
    assert ada.post(base, json={"params": {"n_estimators": 0}}).status_code == 422

    first = ada.post(base, json={"params": {"n_estimators": 5}})
    assert first.status_code == 202, first.text
    model = first.json()
    assert model["status"] == "training"
    assert model["params"]["n_estimators"] == 5
    summary = model["training_set"]
    assert summary["labeled_chunks"] == 1 and summary["rois"]["validation"] == 1
    assert summary["class_values"] == [BONE]
    raw = get_bytes(
        project_storage(settings).child(
            training_path(uuid.UUID(project), summary["id"])
        ),
        "manifest.json",
    )
    assert raw is not None
    manifest = json.loads(raw)
    assert manifest["rois"] == [
        {"bbox": [0, 0, 0, 8, 8, 8], "status": "open", "split": "val"}
    ]
    assert [c[:3] for c in manifest["chunks"]] == [[0, 0, 0]]

    # The same labels make the same training set; new labels a new one.
    again = ada.post(base, json={}).json()
    assert again["training_set"]["id"] == summary["id"]
    paint(settings, migrated_database_url, project, (20, 20, 20), np.ones((1, 2, 2)), 1)
    changed = ada.post(base, json={}).json()
    assert changed["training_set"]["id"] != summary["id"]
    assert [m["id"] for m in ada.get(base).json()] == [
        changed["id"],
        again["id"],
        model["id"],
    ]


def test_model_slots_follow_training_and_deletion(
    new_browser, settings, migrated_database_url
):
    limited = settings.model_copy(
        update={"quota": settings.quota.model_copy(update={"trained_models": 1})}
    )
    ada = new_browser(limited)
    signup(ada)
    project = make_project(ada)
    add_image(limited, migrated_database_url, project)
    add_class(ada, project)
    paint(limited, migrated_database_url, project, (5, 5, 5), np.ones((1, 3, 3)), BONE)
    base = f"/api/projects/{project}/models"
    first = ada.post(base, json={}).json()
    second = ada.post(base, json={})
    assert second.status_code == 403
    assert second.json()["detail"] == "trained_model_quota_exceeded"

    # A failed training gives its slot back.
    async def fail(db):
        await db.execute(
            update(Job)
            .where(Job.id == uuid.UUID(first["pipeline_id"]))
            .values(status="failed")
        )

    run_db(migrated_database_url, fail)
    assert ada.get(f"{base}/{first['id']}").json()["status"] == "failed"
    third = ada.post(base, json={})
    assert third.status_code == 202
    # So does deleting a model (and it stops its training).
    assert ada.request("DELETE", f"{base}/{third.json()['id']}").status_code == 204
    assert ada.get(f"{base}/{third.json()['id']}").status_code == 404
    assert ada.post(base, json={}).status_code == 202


def test_a_model_slot_is_given_back_once(ada, settings, migrated_database_url):
    project = make_project(ada)
    add_image(settings, migrated_database_url, project)
    add_class(ada, project)
    paint(settings, migrated_database_url, project, (5, 5, 5), np.ones((1, 3, 3)), BONE)
    base = f"/api/projects/{project}/models"
    first = ada.post(base, json={}).json()
    ada.post(base, json={})
    assert ada.get("/api/me/quota").json()["trained_models_used"] == 2

    async def race():
        engine = create_engine(migrated_database_url)
        sessions = create_sessionmaker(engine)
        try:
            async with sessions() as one, sessions() as two:
                model = await one.get(TrainedModel, uuid.UUID(first["id"]))
                assert model is not None
                project_row = await one.get(Project, model.project_id)
                assert project_row is not None
                owner, this = project_row.owner_id, TrainedModel.id == model.id
                assert await train.release_slots(one, owner, this) == 1
                # The other waits for the first to finish, then finds the
                # slot already given back.
                other = asyncio.create_task(train.release_slots(two, owner, this))
                await asyncio.sleep(0.3)
                assert not other.done()
                await one.commit()
                assert await other == 0
                await two.commit()
        finally:
            await engine.dispose()

    asyncio.run(race())
    assert ada.get("/api/me/quota").json()["trained_models_used"] == 1
    assert ada.request("DELETE", f"{base}/{first['id']}").status_code == 204
    assert ada.get("/api/me/quota").json()["trained_models_used"] == 1


def test_others_cant_reach_models(new_browser, settings, migrated_database_url):
    ada = new_browser()
    signup(ada)
    project = make_project(ada)
    bob = new_browser()
    signup(bob, username="bob")
    assert bob.get(f"/api/projects/{project}/models").status_code == 404
    assert bob.post(f"/api/projects/{project}/models", json={}).status_code == 404
    assert bob.get("/api/plugins").json()[0]["name"] == "rf"


@pytest.fixture
def settings(migrated_database_url, tmp_path, s3_endpoint, s3_bucket):
    """S3 storage, so the worker reaches it as it would in production."""
    return Settings(
        database_url=migrated_database_url,
        secret_key=SECRET_KEY,
        storage={
            "url": f"s3://{s3_bucket}/{tmp_path.name}",
            "endpoint": s3_endpoint,
            "public_endpoint": s3_endpoint,
            "access_key_id": "test",
            "secret_access_key": "test",
            "region": "us-east-1",
        },
    )


def test_a_worker_trains_a_random_forest(
    new_browser, settings, migrated_database_url, live_server
):
    ada = new_browser()
    signup(ada)
    project = make_project(ada)
    add_image(settings, migrated_database_url, project, with_data=True)
    add_class(ada, project)
    _, truth = ball_image()
    # Training strokes go outside the validation ROI, whose labels are held out.
    paint(
        settings, migrated_database_url, project, (27, 22, 26), np.ones((1, 4, 4)), BONE
    )
    paint(settings, migrated_database_url, project, (2, 2, 2), np.ones((1, 6, 6)), 1)
    val = (14, 18, 22, 26, 30, 34)
    region = tuple(slice(val[a], val[a + 3]) for a in range(3))
    paint(settings, migrated_database_url, project, val[:3], truth[region], BONE)
    ada.post(
        f"/api/projects/{project}/rois",
        json={"bbox": list(val), "kind": "cube", "split": "val"},
    )
    roi = ada.get(f"/api/projects/{project}/rois").json()[0]
    ada.patch(f"/api/projects/{project}/rois/{roi['id']}", json={"status": "complete"})
    params = {
        "n_estimators": 10,
        "max_depth": 8,
        "samples_per_class": 2000,
        "sigma_max": 1.0,
    }
    model = ada.post(f"/api/projects/{project}/models", json={"params": params}).json()

    token = add_worker(migrated_database_url)
    client = ServerClient(token, base_url=live_server)
    caps = WorkerCaps(version="test", kinds=sorted(HANDLERS))
    worker = Worker(client, caps, claim_wait_seconds=0.5, heartbeat_seconds=0.2)
    thread = threading.Thread(target=worker.run, kwargs={"max_jobs": None})
    thread.start()
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status = ada.get(f"/api/projects/{project}/models/{model['id']}").json()
            if status["status"] != "training":
                break
            time.sleep(0.3)
    finally:
        worker.stop()
        thread.join(timeout=30)
        client.close()
    assert status["status"] == "ready", status
    assert status["plugin_version"] == "1"
    assert status["metrics"]["validation_crops"] == 1
    assert status["metrics"]["classes"][str(BONE)]["dice"] > 0.8

    async def model_manifest(db):
        trained = await db.get(TrainedModel, uuid.UUID(model["id"]))
        return (await db.get(Artifact, trained.artifact_id)).manifest

    # The model keeps the window its training crops were normalized with.
    assert run_db(migrated_database_url, model_manifest)["window"] == [200.0, 800.0]
