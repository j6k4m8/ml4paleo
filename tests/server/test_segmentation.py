"""
The final segmentation, end to end: a worker merges the prediction with the
labels and complete ROIs, and removes specks.
"""

import threading
import time
import uuid

import numpy as np
import pytest
import zarr
from helpers import SECRET_KEY, add_worker, run_db, signup
from ml4paleo_server import artifacts, labels
from ml4paleo_server.settings import Settings
from ml4paleo_server.storage import project_storage
from ml4paleo_worker.client import ServerClient
from ml4paleo_worker.handlers import HANDLERS
from ml4paleo_worker.main import Worker

from ml4paleo.labels.deltas import split_into_deltas
from ml4paleo.protocol import WorkerCaps
from ml4paleo.segmentation.predict import create_prediction
from ml4paleo.storage import zarr_store

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
