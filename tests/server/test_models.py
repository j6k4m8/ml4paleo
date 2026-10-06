"""
Models: training sets pinned from labels and ROIs, the models API and its
quota, and a random forest trained end to end by a worker.
"""

import asyncio
import datetime
import json
import threading
import time
import uuid

import numpy as np
import obstore
import pytest
from helpers import SECRET_KEY, add_worker, run_db, signup
from ml4paleo_server import artifacts, jobs, labels
from ml4paleo_server.db import (
    Artifact,
    Job,
    Project,
    Roi,
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
from sqlalchemy import select, update

from ml4paleo.labels.deltas import split_into_deltas
from ml4paleo.ome import OmeImage
from ml4paleo.protocol import WorkerCaps
from ml4paleo.storage import get_bytes, object_store

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


def test_training_sets_cut_rois_to_the_image(ada, settings, migrated_database_url):
    project = make_project(ada)
    add_image(settings, migrated_database_url, project)
    add_class(ada, project)
    paint(settings, migrated_database_url, project, (5, 5, 5), np.ones((1, 3, 3)), BONE)
    rois = f"/api/projects/{project}/rois"
    for bbox in ([0, 0, 0, 8, 8, 8], [8, 8, 8, 16, 16, 16]):
        ada.post(rois, json={"bbox": bbox, "kind": "cube"})
    first, second = (roi["id"] for roi in ada.get(rois).json())

    # As if the image had been replaced by a smaller one since.
    async def move(db):
        for roi_id, bbox in (
            (first, [30, 40, 50, 60, 60, 60]),
            (second, [45, 0, 0, 50, 5, 5]),
        ):
            await db.execute(
                update(Roi).where(Roi.id == uuid.UUID(roi_id)).values(bbox=bbox)
            )

    run_db(migrated_database_url, move)
    model = ada.post(f"/api/projects/{project}/models", json={}).json()
    raw = get_bytes(
        project_storage(settings).child(
            training_path(uuid.UUID(project), model["training_set"]["id"])
        ),
        "manifest.json",
    )
    assert raw is not None
    assert json.loads(raw)["rois"] == [
        {"bbox": [30, 40, 50, 40, 48, 56], "status": "open", "split": "train"}
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
    # New labels would make a new training set, but none is stored for a
    # training that can't start.
    paint(limited, migrated_database_url, project, (9, 9, 9), np.ones((1, 2, 2)), 1)
    second = ada.post(base, json={})
    assert second.status_code == 403
    assert second.json()["detail"] == "trained_model_quota_exceeded"
    stored = object_store(
        project_storage(limited).child(f"projects/{project}/training")
    )
    manifests = [item["path"] for batch in obstore.list(stored) for item in batch]
    assert manifests == [f"{first['training_set']['id']}/manifest.json"]

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


def test_failed_trainings_give_back_slots_across_projects(
    new_browser, settings, migrated_database_url
):
    limited = settings.model_copy(
        update={"quota": settings.quota.model_copy(update={"trained_models": 1})}
    )
    ada = new_browser(limited)
    signup(ada)
    first, second = make_project(ada), make_project(ada)
    for project in (first, second):
        add_image(limited, migrated_database_url, project)
        add_class(ada, project)
        paint(
            limited, migrated_database_url, project, (5, 5, 5), np.ones((1, 3, 3)), BONE
        )
    model = ada.post(f"/api/projects/{first}/models", json={}).json()
    assert ada.post(f"/api/projects/{second}/models", json={}).status_code == 403

    async def fail(db):
        await db.execute(
            update(Job)
            .where(Job.id == uuid.UUID(model["pipeline_id"]))
            .values(status="failed")
        )

    run_db(migrated_database_url, fail)
    # Without anyone listing the first project's models.
    assert ada.post(f"/api/projects/{second}/models", json={}).status_code == 202
    assert ada.get("/api/me/quota").json()["trained_models_used"] == 1


def test_deleting_a_project_gives_back_its_model_slots(
    new_browser, settings, migrated_database_url
):
    limited = settings.model_copy(
        update={"quota": settings.quota.model_copy(update={"trained_models": 1})}
    )
    ada = new_browser(limited)
    signup(ada)
    first, second = make_project(ada), make_project(ada)
    for project in (first, second):
        add_image(limited, migrated_database_url, project)
        add_class(ada, project)
        paint(
            limited, migrated_database_url, project, (5, 5, 5), np.ones((1, 3, 3)), BONE
        )
    model = ada.post(f"/api/projects/{first}/models", json={}).json()
    assert ada.post(f"/api/projects/{second}/models", json={}).status_code == 403

    # Other work of the project, which would run to the end only to be refused.
    async def other_work(db):
        job = await jobs.enqueue(
            db, "noop", {"seconds": 0}, project_id=uuid.UUID(first)
        )
        return job.id

    other = run_db(migrated_database_url, other_work)
    assert ada.request("DELETE", f"/api/projects/{first}").status_code == 204

    async def statuses(db):
        return [
            (await db.get(Job, job_id)).status
            for job_id in (uuid.UUID(model["pipeline_id"]), other)
        ]

    assert run_db(migrated_database_url, statuses) == ["cancelled", "cancelled"]
    assert ada.get("/api/me/quota").json()["trained_models_used"] == 0
    assert ada.post(f"/api/projects/{second}/models", json={}).status_code == 202


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


def add_prediction(database_url, project: str) -> str:
    """A prediction of the project's current image, made its head."""

    async def create(db):
        image = await artifacts.head(db, uuid.UUID(project), "image")
        assert image is not None
        artifact = await artifacts.create_staging(
            db,
            project_id=uuid.UUID(project),
            kind="prediction",
            head_slot="prediction",
            inputs={"image_artifact_id": str(image.id)},
        )
        artifact.state = "committed"
        artifact.manifest = {"class_values": [BONE], "shape_zyx": list(SHAPE)}
        await artifacts.set_head(db, artifact)
        return str(artifact.id)

    return run_db(database_url, create)


def test_a_prediction_of_a_replaced_image_is_hidden(
    ada, settings, migrated_database_url
):
    project = make_project(ada)
    add_image(settings, migrated_database_url, project)
    artifact = add_prediction(migrated_database_url, project)
    prediction = ada.get(f"/api/projects/{project}/prediction").json()
    assert prediction["artifact_id"] == artifact
    assert prediction["shape_zyx"] == list(SHAPE)
    # A new image leaves the prediction in place, but it no longer fits.
    add_image(settings, migrated_database_url, project)
    assert ada.get(f"/api/projects/{project}/prediction").status_code == 404


def labeled_project(browser, settings, database_url) -> str:
    project = make_project(browser)
    add_image(settings, database_url, project)
    add_class(browser, project)
    paint(settings, database_url, project, (5, 5, 5), np.ones((1, 3, 3)), BONE)
    return project


def ready_model(browser, database_url, project: str, window=None) -> dict:
    """A model whose training has finished, as far as the server knows."""
    model = browser.post(f"/api/projects/{project}/models", json={}).json()

    async def finish(db):
        trained = await db.get(TrainedModel, uuid.UUID(model["id"]))
        artifact = await db.get(Artifact, trained.artifact_id)
        artifact.state = "committed"
        artifact.manifest = {"kind": "model"} | ({"window": window} if window else {})
        await db.execute(
            update(Job).where(Job.id == trained.job_id).values(status="succeeded")
        )

    run_db(database_url, finish)
    return model


def predict(browser, project: str, model: dict):
    return browser.post(f"/api/projects/{project}/models/{model['id']}/predict")


def propose(browser, project: str, model: dict, bbox=(0, 0, 0, 16, 16, 16)):
    """Draw an ROI and propose a prediction for it."""
    roi = browser.post(
        f"/api/projects/{project}/rois", json={"bbox": list(bbox), "kind": "cube"}
    ).json()
    return browser.post(
        f"/api/projects/{project}/models/{model['id']}/propose",
        json={"roi_id": roi["id"]},
    )


def windows(database_url, started: dict):
    """The window a prediction recorded, and the one its jobs were given."""

    async def read(db):
        artifact = await db.get(Artifact, uuid.UUID(started["artifact_id"]))
        root = await db.get(Job, uuid.UUID(started["pipeline_id"]))
        return artifact.inputs["window"], root.payload["window"]

    return run_db(database_url, read)


def test_predictions_normalize_like_the_models_training(
    ada, settings, migrated_database_url
):
    project = labeled_project(ada, settings, migrated_database_url)
    kept = ready_model(ada, migrated_database_url, project, window=[100.0, 900.0])
    older = ready_model(ada, migrated_database_url, project)
    started = predict(ada, project, kept).json()
    assert windows(migrated_database_url, started) == ([100.0, 900.0],) * 2
    # A model that doesn't keep its window gets the image's.
    started = predict(ada, project, older).json()
    assert windows(migrated_database_url, started) == ([200, 800],) * 2


def pipeline_status(browser, project: str, pipeline_id: str) -> str:
    return browser.get(f"/api/projects/{project}/pipelines/{pipeline_id}").json()[
        "status"
    ]


def test_a_new_prediction_cancels_older_ones(ada, settings, migrated_database_url):
    project = labeled_project(ada, settings, migrated_database_url)
    first, second = (ready_model(ada, migrated_database_url, project) for _ in "ab")
    older = predict(ada, project, first).json()["pipeline_id"]

    # Its first job is done, so only its shards and finalize are left.
    async def prepared(db):
        await db.execute(
            update(Job).where(Job.id == uuid.UUID(older)).values(status="succeeded")
        )

    run_db(migrated_database_url, prepared)
    assert pipeline_status(ada, project, older) == "running"
    newer = predict(ada, project, second).json()["pipeline_id"]
    assert pipeline_status(ada, project, older) == "cancelled"
    assert pipeline_status(ada, project, newer) == "waiting"


def test_a_running_prediction_isnt_started_again(ada, settings, migrated_database_url):
    project = labeled_project(ada, settings, migrated_database_url)
    model = ready_model(ada, migrated_database_url, project)
    first = predict(ada, project, model)
    assert first.status_code == 202
    again = predict(ada, project, model)
    assert again.status_code == 409
    assert again.json()["detail"] == "A prediction with this model is already running."
    started = first.json()["pipeline_id"]
    assert pipeline_status(ada, project, started) == "waiting"
    # The pipelines say which model they're for, and who started them.
    me = ada.get("/api/auth/session").json()["user"]["id"]
    listed = ada.get(f"/api/projects/{project}/pipelines").json()
    assert [
        (p["id"], p["model_id"], p["created_by"])
        for p in listed
        if p["kind"] == "prediction"
    ] == [(started, model["id"], me)]
    # A new image is a new prediction, and the old one stops.
    add_image(settings, migrated_database_url, project)
    assert predict(ada, project, model).status_code == 202
    assert pipeline_status(ada, project, started) == "cancelled"


def test_deleting_a_model_stops_its_predictions(ada, settings, migrated_database_url):
    project = labeled_project(ada, settings, migrated_database_url)
    model, other = (ready_model(ada, migrated_database_url, project) for _ in "ab")
    started = predict(ada, project, model).json()["pipeline_id"]
    base = f"/api/projects/{project}/models"
    assert ada.request("DELETE", f"{base}/{other['id']}").status_code == 204
    assert pipeline_status(ada, project, started) == "waiting"
    assert ada.request("DELETE", f"{base}/{model['id']}").status_code == 204
    assert pipeline_status(ada, project, started) == "cancelled"


def test_deleting_a_model_stops_its_proposals(ada, settings, migrated_database_url):
    project = labeled_project(ada, settings, migrated_database_url)
    model, other = (ready_model(ada, migrated_database_url, project) for _ in "ab")
    proposed = propose(ada, project, model)
    assert proposed.status_code == 202, proposed.text
    started = proposed.json()["pipeline_id"]
    base = f"/api/projects/{project}/models"
    assert ada.request("DELETE", f"{base}/{other['id']}").status_code == 204
    assert pipeline_status(ada, project, started) == "waiting"
    # So it can't finish later and become the proposal of a deleted model.
    assert ada.request("DELETE", f"{base}/{model['id']}").status_code == 204
    assert pipeline_status(ada, project, started) == "cancelled"


def made(database_url, started: dict) -> None:
    """Finish a proposal as its job would: commit it, which moves its head."""

    async def commit(db):
        artifact = await db.get(Artifact, uuid.UUID(started["artifact_id"]))
        artifact.state = "committed"
        artifact.manifest = {"class_values": [BONE], "shape_zyx": list(SHAPE)}
        await artifacts.set_head(db, artifact)
        await db.execute(
            update(Job)
            .where(Job.id == uuid.UUID(started["pipeline_id"]))
            .values(status="succeeded")
        )

    run_db(database_url, commit)


def test_each_person_has_their_own_proposal(
    ada, new_browser, settings, migrated_database_url
):
    project = labeled_project(ada, settings, migrated_database_url)
    bob = new_browser()
    signup(bob, username="bob")
    ada.post(f"/api/projects/{project}/members", json={"username": "bob"})
    model = ready_model(ada, migrated_database_url, project)
    first = propose(ada, project, model).json()
    theirs = propose(bob, project, model, bbox=(8, 8, 8, 24, 24, 24)).json()
    # Bob's proposal leaves Ada's running; her next one stops only hers.
    assert pipeline_status(ada, project, first["pipeline_id"]) == "waiting"
    second = propose(ada, project, model).json()
    assert pipeline_status(ada, project, first["pipeline_id"]) == "cancelled"
    assert pipeline_status(ada, project, theirs["pipeline_id"]) == "waiting"

    url = f"/api/projects/{project}/proposal"
    assert ada.get(url).status_code == 404
    made(migrated_database_url, theirs)
    assert ada.get(url).status_code == 404
    assert bob.get(url).json()["artifact_id"] == theirs["artifact_id"]
    made(migrated_database_url, second)
    assert ada.get(url).json()["artifact_id"] == second["artifact_id"]
    assert bob.get(url).json()["artifact_id"] == theirs["artifact_id"]
    # Ada's newer proposal replaces her own, not Bob's.
    third = propose(ada, project, model).json()
    made(migrated_database_url, third)
    assert ada.get(url).json()["artifact_id"] == third["artifact_id"]
    assert bob.get(url).json()["artifact_id"] == theirs["artifact_id"]

    async def states(db):
        return [
            (await db.get(Artifact, uuid.UUID(started["artifact_id"]))).state
            for started in (second, theirs)
        ]

    assert run_db(migrated_database_url, states) == ["superseded", "committed"]


def test_proposals_cut_rois_to_the_image(ada, settings, migrated_database_url):
    project = labeled_project(ada, settings, migrated_database_url)
    model = ready_model(ada, migrated_database_url, project)
    roi = ada.post(
        f"/api/projects/{project}/rois",
        json={"bbox": [0, 0, 0, 8, 8, 8], "kind": "cube"},
    ).json()["id"]
    url = f"/api/projects/{project}/models/{model['id']}/propose"

    # As if the image had been replaced by a smaller one since.
    async def move(db, bbox):
        await db.execute(update(Roi).where(Roi.id == uuid.UUID(roi)).values(bbox=bbox))

    run_db(migrated_database_url, lambda db: move(db, [30, 40, 50, 60, 60, 60]))
    started = ada.post(url, json={"roi_id": roi}).json()

    async def box(db):
        return (await db.get(Artifact, uuid.UUID(started["artifact_id"]))).inputs["box"]

    assert run_db(migrated_database_url, box) == [30, 40, 50, 40, 48, 56]
    run_db(migrated_database_url, lambda db: move(db, [45, 0, 0, 50, 5, 5]))
    outside = ada.post(url, json={"roi_id": roi})
    assert outside.status_code == 409
    assert outside.json()["detail"] == "That ROI is outside the image."


def test_prediction_shards_ask_for_the_plugins_gpu(
    ada, settings, migrated_database_url, monkeypatch
):
    from ml4paleo.segmentation.plugin import PluginCaps
    from ml4paleo.segmentation.plugins.rf import RandomForestPlugin

    project = labeled_project(ada, settings, migrated_database_url)
    model = ready_model(ada, migrated_database_url, project)
    caps = PluginCaps(devices=("cuda",), min_vram_gb=6.0)
    monkeypatch.setattr(RandomForestPlugin, "caps", caps)
    started = predict(ada, project, model).json()["pipeline_id"]

    async def needs(db):
        rows = await db.execute(
            select(Job.kind, Job.min_vram_gb).where(Job.root_id == uuid.UUID(started))
        )
        return {kind: vram for kind, vram in rows}

    assert run_db(migrated_database_url, needs) == {
        "predict.prepare": 0,
        "predict.shard": 6.0,
        "prediction.finalize": 0,
    }


def test_missing_label_blobs_fail_training_for_good(tmp_path):
    from ml4paleo_worker.context import JobContext, PermanentError
    from ml4paleo_worker.handlers import train as train_handler

    from ml4paleo.protocol import JobLease
    from ml4paleo.storage import StorageGrant, put_bytes

    def grant(name, access="r"):
        (tmp_path / name).mkdir()
        return StorageGrant(url=f"file://{tmp_path}/{name}", access=access)

    image = grant("image", "rw")
    OmeImage.create(image, shape_czyx=(1, 16, 16, 16), dtype=np.uint16)
    training = grant("training", "rw")
    manifest = {
        "image": {"window": [0, 1]},
        "class_values": [BONE],
        "rois": [],
        "chunks": [[0, 0, 0, "ab" * 32]],
    }
    put_bytes(training, "manifest.json", json.dumps(manifest).encode())
    lease = JobLease(
        job_id=uuid.uuid4(),
        kind="model.train",
        payload={"plugin": "rf", "params": {"sigma_max": 1.0}, "training_set": "x"},
        lease_token="token",
        lease_expires_at=datetime.datetime.now(datetime.UTC),
        attempt=1,
        grants=[image, grant("labels"), training, grant("model", "rw")],
    )
    with pytest.raises(PermanentError, match="missing"):
        train_handler.run(JobContext(lease))


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

    def wait(url, done):
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            response = ada.get(url)
            if done(response):
                return response
            time.sleep(0.3)
        raise AssertionError(f"{url} never finished")

    try:
        status = wait(
            f"/api/projects/{project}/models/{model['id']}",
            lambda r: r.json()["status"] != "training",
        ).json()
        assert status["status"] == "ready", status
        assert status["plugin_version"] == "1"
        assert status["metrics"]["validation_crops"] == 1
        assert status["metrics"]["classes"][str(BONE)]["dice"] > 0.8
        # The worker's memory budget leaves room for every sample asked for.
        assert status["metrics"]["samples_per_class_used"] == 2000

        # The model predicts the whole image, which becomes the prediction.
        assert ada.get(f"/api/projects/{project}/prediction").status_code == 404
        started = ada.post(f"/api/projects/{project}/models/{model['id']}/predict")
        assert started.status_code == 202, started.text
        pipeline = wait(
            f"/api/projects/{project}/pipelines/{started.json()['pipeline_id']}",
            lambda r: r.json()["status"] in ("succeeded", "failed", "cancelled"),
        ).json()
        assert pipeline["status"] == "succeeded", pipeline
        assert pipeline["kind"] == "prediction"

        # A proposal predicts just one ROI, on demand.
        assert ada.get(f"/api/projects/{project}/proposal").status_code == 404
        proposed = ada.post(
            f"/api/projects/{project}/models/{model['id']}/propose",
            json={"roi_id": roi["id"]},
        )
        assert proposed.status_code == 202, proposed.text
        proposal_pipeline = wait(
            f"/api/projects/{project}/pipelines/{proposed.json()['pipeline_id']}",
            lambda r: r.json()["status"] in ("succeeded", "failed", "cancelled"),
        ).json()
        assert proposal_pipeline["status"] == "succeeded", proposal_pipeline
        assert proposal_pipeline["kind"] == "proposal"
    finally:
        worker.stop()
        thread.join(timeout=30)
        client.close()
    proposal = ada.get(f"/api/projects/{project}/proposal").json()
    assert proposal["roi_id"] == roi["id"]
    assert proposal["box"] == list(val)
    prediction = ada.get(f"/api/projects/{project}/prediction").json()
    assert prediction["model_id"] == model["id"]
    assert prediction["class_values"] == [BONE]
    assert prediction["shape_zyx"] == list(SHAPE)
    metadata = ada.get(prediction["zarr_url"] + "class/zarr.json").json()
    assert metadata["shape"] == list(SHAPE)
    import zarr

    from ml4paleo.storage import zarr_store

    group = zarr.open_group(
        store=zarr_store(
            project_storage(settings).child(
                f"projects/{project}/artifacts/{prediction['artifact_id']}"
            )
        ),
        mode="r",
    )
    predicted = np.asarray(group["class"][:])
    assert ((predicted == BONE) == truth).mean() > 0.95
    # The proposal holds the same prediction inside its ROI, and nothing else.
    proposed_classes = np.asarray(
        zarr.open_group(
            store=zarr_store(
                project_storage(settings).child(
                    f"projects/{project}/artifacts/{proposal['artifact_id']}"
                )
            ),
            mode="r",
        )["class"][:]
    )
    inside = tuple(slice(val[a], val[a + 3]) for a in range(3))
    np.testing.assert_array_equal(proposed_classes[inside], predicted[inside])
    proposed_classes[inside] = 0
    assert not proposed_classes.any()

    async def model_manifest(db):
        trained = await db.get(TrainedModel, uuid.UUID(model["id"]))
        return (await db.get(Artifact, trained.artifact_id)).manifest

    async def prediction_window(db):
        artifact = await db.get(Artifact, uuid.UUID(prediction["artifact_id"]))
        return artifact.manifest["window"]

    # The model keeps the window its training crops were normalized with,
    # and its prediction normalized the image with it too.
    assert run_db(migrated_database_url, model_manifest)["window"] == [200.0, 800.0]
    assert run_db(migrated_database_url, prediction_window) == [200.0, 800.0]
