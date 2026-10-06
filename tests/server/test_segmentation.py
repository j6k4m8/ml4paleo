"""
The final segmentation, end to end: a worker merges the prediction with the
labels and complete ROIs, and removes specks.
"""

import datetime
import threading
import time
import tracemalloc
import uuid

import numpy as np
import obstore
import pytest
import zarr
from helpers import SECRET_KEY, add_worker, run_db, signup
from ml4paleo_server import artifacts, labels
from ml4paleo_server.db import Artifact, Job, LabelOp, TrainedModel, TrainingSet
from ml4paleo_server.settings import Settings
from ml4paleo_server.storage import project_storage
from ml4paleo_worker.client import ServerClient
from ml4paleo_worker.context import Cancelled, JobContext, PermanentError
from ml4paleo_worker.handlers import HANDLERS
from ml4paleo_worker.handlers import compose as jobs
from ml4paleo_worker.main import Worker
from scipy import ndimage
from sqlalchemy import select

from ml4paleo.labels.deltas import split_into_deltas
from ml4paleo.protocol import JobLease, WorkerCaps
from ml4paleo.segmentation.predict import (
    SHARD_ZYX,
    create_prediction,
    open_prediction,
    shard_boxes,
)
from ml4paleo.storage import StorageGrant, get_bytes, object_store, zarr_store

SHAPE = (20, 24, 28)
BONE, TOOTH = 2, 3


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


def predicted() -> np.ndarray:
    classes = np.ones(SHAPE, dtype=np.uint8)
    classes[2:12, 2:12, 2:12] = BONE  # a big piece
    classes[15, 20, 20] = BONE  # a speck
    classes[15, 5, 20:22] = BONE  # a speck someone labeled part of
    classes[2:12, 14:22, 2:12] = BONE  # a big piece inside a complete ROI
    return classes


async def new_image(db, project: str, shape=SHAPE):
    image = await artifacts.create_staging(
        db, project_id=uuid.UUID(project), kind="image", head_slot="image"
    )
    image.state = "committed"
    image.manifest = {"shape_czyx": [1, *shape], "window": [0, 1]}
    await artifacts.set_head(db, image)
    return image


def add_prediction(settings, database_url, project: str, classes=None):
    classes = predicted() if classes is None else classes

    async def create(db):
        image = await new_image(db, project, classes.shape)
        artifact = await artifacts.create_staging(
            db,
            project_id=uuid.UUID(project),
            kind="prediction",
            head_slot="prediction",
            inputs={"model_id": None, "image_artifact_id": str(image.id)},
        )
        grant = project_storage(settings).child(artifacts.artifact_path(artifact))
        group = create_prediction(grant, classes.shape)
        group["class"][:] = classes  # type: ignore[index]
        artifact.state = "committed"
        artifact.manifest = {
            "kind": "prediction",
            "shape_zyx": list(classes.shape),
            "class_values": [BONE],
        }
        await artifacts.set_head(db, artifact)

    run_db(database_url, create)


def run_worker(database_url, live_server, browser, project: str, pipeline_id: str):
    """Run a worker until the pipeline finishes, and return the pipeline."""
    token = add_worker(database_url)
    client = ServerClient(token, base_url=live_server)
    worker = Worker(
        client,
        WorkerCaps(version="test", kinds=sorted(HANDLERS)),
        claim_wait_seconds=0.5,
        heartbeat_seconds=0.2,
    )
    thread = threading.Thread(target=worker.run, kwargs={"max_jobs": None})
    thread.start()
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            pipeline = browser.get(
                f"/api/projects/{project}/pipelines/{pipeline_id}"
            ).json()
            if pipeline["status"] in ("succeeded", "failed", "cancelled"):
                return pipeline
            time.sleep(0.3)
    finally:
        worker.stop()
        thread.join(timeout=30)
        client.close()
    raise AssertionError("The pipeline didn't finish.")


def made(settings, project: str, artifact_id: str) -> np.ndarray:
    grant = project_storage(settings).child(
        f"projects/{project}/artifacts/{artifact_id}"
    )
    return np.asarray(zarr.open_group(store=zarr_store(grant), mode="r")["class"][:])


def paint(settings, database_url, project, origin, mask, value):
    async def apply(db):
        await labels.apply_edit(
            db,
            settings,
            uuid.UUID(project),
            client_op_id=uuid.uuid4(),
            deltas=split_into_deltas(np.asarray(mask, dtype=bool), origin, value=value),
        )

    run_db(database_url, apply)


def test_a_worker_composes_the_final_segmentation(
    new_browser, settings, migrated_database_url, live_server
):
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    base = f"/api/projects/{project}/segmentation"
    assert ada.post(base, json={}).status_code == 409  # nothing to compose yet
    for name in ("bone", "tooth"):
        ada.post(
            f"/api/projects/{project}/labels/classes",
            json={"name": name, "color": "#ffffff"},
        )
    add_prediction(settings, migrated_database_url, project)
    paint(
        settings, migrated_database_url, project, (2, 2, 2), np.ones((2, 2, 2)), TOOTH
    )
    paint(
        settings, migrated_database_url, project, (15, 5, 20), np.ones((1, 1, 1)), BONE
    )
    ada.post(
        f"/api/projects/{project}/rois",
        json={"bbox": [0, 13, 0, 14, 24, 14], "kind": "cube"},
    )
    roi = ada.get(f"/api/projects/{project}/rois").json()[0]
    ada.patch(f"/api/projects/{project}/rois/{roi['id']}", json={"status": "complete"})
    assert ada.get(base).status_code == 404

    async def newest_edit(db):
        return await db.scalar(
            select(LabelOp.created_at)
            .where(LabelOp.project_id == uuid.UUID(project))
            .order_by(LabelOp.seq.desc())
            .limit(1)
        )

    pinned = run_db(migrated_database_url, newest_edit)
    started = ada.post(base, json={"min_voxels": 10})
    assert started.status_code == 202, started.text
    # Labeling the speck now is too late for this one.
    paint(
        settings,
        migrated_database_url,
        project,
        (15, 20, 20),
        np.ones((1, 1, 1)),
        TOOTH,
    )
    pipeline = run_worker(
        migrated_database_url,
        live_server,
        ada,
        project,
        started.json()["pipeline_id"],
    )
    assert pipeline["status"] == "succeeded", pipeline
    assert pipeline["kind"] == "segmentation"

    segmentation = ada.get(base).json()
    assert segmentation["min_voxels"] == 10
    assert datetime.datetime.fromisoformat(segmentation["labels_as_of"]) == pinned
    final = made(settings, project, segmentation["artifact_id"])
    # Labels overrule the prediction.
    assert (final[2:4, 2:4, 2:4] == TOOTH).all()
    assert (final[4:12, 4:12, 4:12] == BONE).all()
    # Unlabeled voxels in the complete ROI are background.
    assert (final[2:12, 14:22, 2:12] == 1).all()
    # The speck is gone; the one someone labeled stays.
    assert final[15, 20, 20] == 1
    assert (final[15, 5, 20:22] == BONE).all()
    # Scratch files are cleaned up; the pinned labels stay, for the record.
    grant = project_storage(settings).child(
        f"projects/{project}/artifacts/{segmentation['artifact_id']}"
    )
    assert get_bytes(grant, "scratch/0.npz") is None
    assert get_bytes(grant, "inputs.json") is not None


def test_shards_decide_their_own_specks_and_seams_join_once_per_pair(
    new_browser, settings, migrated_database_url, live_server
):
    # Three shards along x, with seams after x 511 and x 1023.
    classes = np.ones((4, 8, 1030), dtype=np.uint8)
    classes[:, :, 500:530] = BONE  # a slab through the first seam: 32 voxels touch
    classes[0, 0, 511:513] = TOOTH  # a speck across the first seam
    classes[3, 7, 1020:1030] = BONE  # a bar across the second, too short to keep
    classes[1, 1, 100] = BONE  # specks inside the first and last shards
    classes[2, 2, 1027] = TOOTH
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    add_prediction(settings, migrated_database_url, project, classes)
    started = ada.post(f"/api/projects/{project}/segmentation", json={"min_voxels": 20})
    assert started.status_code == 202, started.text
    pipeline_id = started.json()["pipeline_id"]
    pipeline = run_worker(migrated_database_url, live_server, ada, project, pipeline_id)
    assert pipeline["status"] == "succeeded", pipeline

    async def results(db):
        rows = await db.execute(
            select(Job.kind, Job.payload, Job.result).where(
                Job.root_id == uuid.UUID(pipeline_id)
            )
        )
        return {(kind, payload.get("shard")): result for kind, payload, result in rows}

    found = run_db(migrated_database_url, results)
    # Each shard decides the specks inside it, and passes on only its pieces
    # on seams: the slab, the tooth speck, and the bar.
    assert [found["cc.block", shard] for shard in range(3)] == [
        {"specks": 1, "seam_pieces": 2},
        {"specks": 0, "seam_pieces": 3},
        {"specks": 1, "seam_pieces": 1},
    ]
    # The merge joins each pair of touching pieces once, however many voxels
    # touch, and finds the halves of the tooth speck and the bar too small.
    assert found["cc.merge", None] == {"specks": 4, "pairs": 3}
    want = classes.copy()
    for speck in [
        (0, 0, slice(511, 513)),
        (3, 7, slice(1020, 1030)),
        (1, 1, 100),
        (2, 2, 1027),
    ]:
        want[speck] = 1
    segmentation = ada.get(f"/api/projects/{project}/segmentation").json()
    assert np.array_equal(made(settings, project, segmentation["artifact_id"]), want)


def test_a_retried_finalize_succeeds_on_local_disk(tmp_path):
    noisy = Noisy(tmp_path)
    jobs.prepare(noisy.job("compose.prepare"))
    for index in range(4):
        jobs.block(noisy.job("cc.block", index))
    jobs.merge(noisy.job("cc.merge"))
    for index in range(4):
        jobs.apply(noisy.job("cc.apply", index))
    jobs.finalize(noisy.job("compose.finalize"))
    # Again, as after a success the server never heard about: the scratch
    # files are gone already.
    jobs.finalize(noisy.job("compose.finalize"))
    assert get_bytes(noisy.grants[2], "_MANIFEST.json") is not None
    assert get_bytes(noisy.grants[2], "scratch/0.npz") is None
    assert np.array_equal(noisy.made(), noisy.want())


def test_others_cant_reach_the_final_segmentation(
    new_browser, settings, migrated_database_url
):
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    add_prediction(settings, migrated_database_url, project)
    bob = new_browser()
    signup(bob, username="bob")
    base = f"/api/projects/{project}/segmentation"
    assert bob.get(base).status_code == 404
    assert bob.post(base, json={}).status_code == 404
    assert ada.post(base, json={}).status_code == 202


def test_the_final_segmentation_is_described_by_what_the_server_recorded(
    new_browser, settings, migrated_database_url
):
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    bob = new_browser()
    signup(bob, username="bob")
    theirs = bob.post("/api/projects", json={"name": "Jaw"}).json()["id"]

    async def setup(db):
        db.add(TrainingSet(id="a" * 64, project_id=uuid.UUID(theirs), summary={}))
        await db.flush()
        model = TrainedModel(
            project_id=uuid.UUID(theirs),
            name="Bob's secret model",
            plugin="rf",
            params={},
            training_set_id="a" * 64,
            class_values=[BONE],
        )
        db.add(model)
        await db.flush()
        artifact = await artifacts.create_staging(
            db,
            project_id=uuid.UUID(project),
            kind="segmentation",
            head_slot="segmentation",
            inputs={"model_id": str(model.id), "min_voxels": 7, "label_seq": 0},
        )
        artifact.state = "committed"
        # What a worker wrote doesn't count.
        artifact.manifest = {
            "kind": "segmentation",
            "shape_zyx": list(SHAPE),
            "model_id": "not a model",
            "min_voxels": 99,
        }
        await artifacts.set_head(db, artifact)

    run_db(migrated_database_url, setup)
    out = ada.get(f"/api/projects/{project}/segmentation")
    assert out.status_code == 200, out.text
    # Another project's model stays unnamed.
    assert out.json()["model_name"] is None
    assert out.json()["min_voxels"] == 7
    assert out.json()["labels_as_of"] is None


def test_one_final_segmentation_at_a_time(new_browser, settings, migrated_database_url):
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    add_prediction(settings, migrated_database_url, project)
    base = f"/api/projects/{project}/segmentation"
    first = ada.post(base, json={})
    assert first.status_code == 202, first.text
    pipeline = first.json()["pipeline_id"]
    again = ada.post(base, json={"min_voxels": 5})
    assert again.status_code == 409
    assert again.json()["detail"]["pipeline_id"] == pipeline
    # Once it has stopped, another can start.
    cancel = ada.post(f"/api/projects/{project}/pipelines/{pipeline}/cancel")
    assert cancel.status_code == 204
    assert ada.post(base, json={}).status_code == 202


def test_a_prediction_from_an_older_image_is_refused(
    new_browser, settings, migrated_database_url
):
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    add_prediction(settings, migrated_database_url, project)
    # A new image replaces the one the prediction was made from.
    run_db(migrated_database_url, lambda db: new_image(db, project))
    started = ada.post(f"/api/projects/{project}/segmentation", json={})
    assert started.status_code == 409
    assert started.json()["detail"] == (
        "The prediction is from an older image; predict again."
    )


def test_a_start_that_fails_leaves_no_files(
    new_browser, settings, migrated_database_url
):
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    add_prediction(settings, migrated_database_url, project)

    async def take_prediction(db):
        # As if garbage collection took it just as composing started.
        prediction = await artifacts.head(db, uuid.UUID(project), "prediction")
        prediction.state = "deleting"
        return prediction.id

    prediction_id = run_db(migrated_database_url, take_prediction)
    started = ada.post(f"/api/projects/{project}/segmentation", json={})
    assert started.status_code == 409, started.text
    assert "can't use them" in started.json()["detail"]
    files = object_store(project_storage(settings).child(f"projects/{project}"))
    assert all(
        meta["path"].startswith(f"artifacts/{prediction_id}/")
        for batch in obstore.list(files)
        for meta in batch
    )

    async def kinds(db):
        return list(
            await db.scalars(
                select(Artifact.kind).where(Artifact.project_id == uuid.UUID(project))
            )
        )

    assert "segmentation" not in run_db(migrated_database_url, kinds)


class Noisy:
    """
    A final segmentation's jobs over a noisy prediction on local disk: every
    voxel background, bone, or tooth at random, over four shards, so hundreds
    of thousands of pieces, most of them specks.
    """

    shape = (4, 520, 520)

    def __init__(self, root):
        rng = np.random.default_rng(3)
        self.classes = rng.choice(
            np.array([1, BONE, TOOTH], dtype=np.uint8),
            size=self.shape,
            p=[0.5, 0.25, 0.25],
        )
        prediction = StorageGrant(url=f"file://{root}/prediction", access="rw")
        group = create_prediction(prediction, self.shape)
        group["class"][:] = self.classes  # type: ignore[index]
        self.grants = [
            prediction.model_copy(update={"access": "r"}),
            StorageGrant(url=f"file://{root}/labels"),
            StorageGrant(url=f"file://{root}/segmentation", access="rw"),
        ]
        self.boxes = shard_boxes(self.shape, SHARD_ZYX)
        assert len(self.boxes) == 4
        self.payload = {
            "shape_zyx": list(self.shape),
            "min_voxels": 20,
            "model_id": None,
            "prediction_artifact_id": str(uuid.uuid4()),
            "label_seq": 0,
            "shards": len(self.boxes),
        }

    def job(self, kind, shard=None, budget=32 * 1024**2, context=JobContext):
        """A job as a worker would get it, with `budget` bytes to use."""
        payload = dict(self.payload)
        if kind == "compose.prepare":
            payload["inputs"] = {"chunks": [], "complete_rois": [], "label_seq": 0}
        if shard is not None:
            payload.update(shard=shard, box=list(self.boxes[shard]))
        return context(
            JobLease(
                job_id=uuid.uuid4(),
                kind=kind,
                payload=payload,
                lease_token="token",
                lease_expires_at=datetime.datetime.now(datetime.UTC)
                + datetime.timedelta(minutes=5),
                attempt=1,
                grants=self.grants,
            ),
            memory_budget_bytes=budget,
        )

    def want(self) -> np.ndarray:
        """The final segmentation, from labeling the whole volume at once."""
        want = self.classes.copy()
        for value in (BONE, TOOTH):
            ids, count = ndimage.label(self.classes == value)  # type: ignore[misc]
            speck = np.bincount(ids.ravel(), minlength=count + 1) < 20
            speck[0] = False
            want[speck[ids]] = 1
        return want

    def made(self) -> np.ndarray:
        return open_prediction(self.grants[2])["class"][:]  # type: ignore[index]


def run_within(ctx: JobContext, handler) -> dict:
    tracemalloc.start()
    try:
        result = handler(ctx)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak <= ctx.memory_budget_bytes, (ctx.lease.kind, peak)
    return result


def test_noisy_compose_jobs_stay_within_their_budget_or_fail_for_good(tmp_path):
    noisy = Noisy(tmp_path)
    jobs.prepare(noisy.job("compose.prepare"))
    # Too little memory to label a 4×512×512 shard: it fails for good, at once.
    with pytest.raises(PermanentError, match="a job may use on this worker"):
        jobs.block(noisy.job("cc.block", 0, budget=8 * 1024**2))
    blocks = [run_within(noisy.job("cc.block", i), jobs.block) for i in range(4)]
    assert sum(found["specks"] for found in blocks) > 100_000
    with pytest.raises(PermanentError, match="Joining the pieces"):
        jobs.merge(noisy.job("cc.merge", budget=64 * 1024))
    run_within(noisy.job("cc.merge"), jobs.merge)
    for index in range(4):
        run_within(noisy.job("cc.apply", index), jobs.apply)
    jobs.finalize(noisy.job("compose.finalize"))
    assert np.array_equal(noisy.made(), noisy.want())


class StopsAtOnce(JobContext):
    """A job that is asked to stop as soon as it reports progress."""

    def progress(self, fraction: float, message: str | None = None) -> None:
        super().progress(fraction, message)
        self.stop("cancelled")


def test_compose_jobs_report_progress_and_stop_partway(tmp_path):
    noisy = Noisy(tmp_path)
    jobs.prepare(noisy.job("compose.prepare"))
    steps = [
        ("cc.block", jobs.block, range(4)),
        ("cc.merge", jobs.merge, [None]),
        ("cc.apply", jobs.apply, range(4)),
        ("compose.finalize", jobs.finalize, [None]),
    ]
    for kind, handler, shards in steps:
        # Each step checks in as it goes, not just once.
        stopping = noisy.job(kind, next(iter(shards)), context=StopsAtOnce)
        with pytest.raises(Cancelled):
            handler(stopping)
        done, _ = stopping.take_progress()
        assert done is not None and done < 1, kind
        for shard in shards:
            ctx = noisy.job(kind, shard)
            handler(ctx)
            done, _ = ctx.take_progress()
            assert done is not None and done >= 0.8, kind
    assert np.array_equal(noisy.made(), noisy.want())
