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
from ml4paleo_server.db import Artifact
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
from ml4paleo.storage import StorageGrant, object_store, zarr_store

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


def add_prediction(settings, database_url, project: str):
    async def create(db):
        image = await artifacts.create_staging(
            db, project_id=uuid.UUID(project), kind="image", head_slot="image"
        )
        image.state = "committed"
        image.manifest = {"shape_czyx": [1, *SHAPE], "window": [0, 1]}
        await artifacts.set_head(db, image)
        artifact = await artifacts.create_staging(
            db,
            project_id=uuid.UUID(project),
            kind="prediction",
            head_slot="prediction",
            inputs={"model_id": None},
        )
        grant = project_storage(settings).child(artifacts.artifact_path(artifact))
        group = create_prediction(grant, SHAPE)
        group["class"][:] = predicted()  # type: ignore[index]
        artifact.state = "committed"
        artifact.manifest = {
            "kind": "prediction",
            "shape_zyx": list(SHAPE),
            "class_values": [BONE],
        }
        await artifacts.set_head(db, artifact)

    run_db(database_url, create)


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

    started = ada.post(base, json={"min_voxels": 10})
    assert started.status_code == 202, started.text
    token = add_worker(migrated_database_url)
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
        pipeline_url = (
            f"/api/projects/{project}/pipelines/{started.json()['pipeline_id']}"
        )
        while time.monotonic() < deadline:
            pipeline = ada.get(pipeline_url).json()
            if pipeline["status"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.3)
    finally:
        worker.stop()
        thread.join(timeout=30)
        client.close()
    assert pipeline["status"] == "succeeded", pipeline
    assert pipeline["kind"] == "segmentation"

    segmentation = ada.get(base).json()
    assert segmentation["min_voxels"] == 10
    group = zarr.open_group(
        store=zarr_store(
            project_storage(settings).child(
                f"projects/{project}/artifacts/{segmentation['artifact_id']}"
            )
        ),
        mode="r",
    )
    final = np.asarray(group["class"][:])
    # Labels overrule the prediction.
    assert (final[2:4, 2:4, 2:4] == TOOTH).all()
    assert (final[4:12, 4:12, 4:12] == BONE).all()
    # Unlabeled voxels in the complete ROI are background.
    assert (final[2:12, 14:22, 2:12] == 1).all()
    # The speck is gone; the one someone labeled stays.
    assert final[15, 20, 20] == 1
    assert (final[15, 5, 20:22] == BONE).all()
    # Scratch files are cleaned up; the pinned labels stay, for the record.
    from ml4paleo.storage import get_bytes

    grant = project_storage(settings).child(
        f"projects/{project}/artifacts/{segmentation['artifact_id']}"
    )
    assert get_bytes(grant, "scratch/0.npz") is None
    assert get_bytes(grant, "inputs.json") is not None


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
