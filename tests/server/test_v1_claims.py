"""
Importing v1 jobs, end to end: claiming one by its id makes a project, and a
worker with the v1 volume brings over the image (in z, y, x), the placed
annotation samples as labels with complete slice ROIs, and the finished
segmentation as the prediction. The first claim wins; admins can release.
"""

import datetime
import shutil
import threading
import time
import uuid

import numpy as np
import pytest
import v1_volume
import zarr
from helpers import SECRET_KEY, add_worker, bearer, make_admin, run_db, signup
from ml4paleo_server import jobs
from ml4paleo_server.db import Job
from ml4paleo_server.db import Worker as WorkerRow
from ml4paleo_server.settings import Settings
from ml4paleo_server.storage import project_storage
from ml4paleo_worker.client import ServerClient
from ml4paleo_worker.context import JobContext, PermanentError
from ml4paleo_worker.handlers import HANDLERS, V1_HANDLERS, v1import
from ml4paleo_worker.main import Worker
from sqlalchemy import select

from ml4paleo.labels import BACKGROUND, LABEL_CHUNK_ZYX
from ml4paleo.labels.codec import decode_chunk
from ml4paleo.protocol import JobLease, WorkerCaps
from ml4paleo.storage import StorageGrant, zarr_store


@pytest.fixture
def volume(tmp_path):
    return v1_volume.make(tmp_path / "v1")


@pytest.fixture
def settings(migrated_database_url, tmp_path, s3_endpoint, s3_bucket, volume):
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
        v1={"volume_path": volume},
    )


def context(volume, kind, payload, grants=(), memory=4 * 1024**3) -> JobContext:
    """A job's context, to run one handler by itself."""
    lease = JobLease(
        job_id=uuid.uuid4(),
        kind=kind,
        payload=payload,
        lease_token="t",
        lease_expires_at=datetime.datetime.now(datetime.UTC),
        attempt=1,
        grants=list(grants),
    )
    return JobContext(lease, memory, v1_volume=volume)


def run_import(database_url, live_server, browser, project, pipeline, volume):
    """
    Run an import as the v1 override does: a worker with the v1 volume for
    the import's own jobs, and a plain worker for the rest.
    """
    workers = [
        Worker(
            ServerClient(add_worker(database_url, name=name), base_url=live_server),
            WorkerCaps(version="test", kinds=sorted(handlers), labels=labels),
            handlers=handlers,
            claim_wait_seconds=0.5,
            heartbeat_seconds=0.2,
            v1_volume=v1_volume,
        )
        for name, handlers, labels, v1_volume in (
            ("worker-v1", V1_HANDLERS, ["v1-volume"], volume),
            ("worker-cpu", HANDLERS, [], None),
        )
    ]
    threads = [
        threading.Thread(target=worker.run, kwargs={"max_jobs": None})
        for worker in workers
    ]
    for thread in threads:
        thread.start()
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status = browser.get(f"/api/projects/{project}/pipelines/{pipeline}").json()
            if status["status"] in ("succeeded", "failed", "cancelled"):
                return status
            time.sleep(0.3)
        raise AssertionError("The import never finished")
    finally:
        for worker in workers:
            worker.stop()
        for thread in threads:
            thread.join(timeout=30)
        for worker in workers:
            worker.client.close()


def ran_on(database_url, pipeline) -> dict[str, set[str]]:
    """The workers that ran each kind of job in a pipeline."""

    async def look(db):
        rows = await db.execute(
            select(Job.kind, WorkerRow.name)
            .join(WorkerRow, WorkerRow.id == Job.lease_worker_id)
            .where(Job.root_id == uuid.UUID(pipeline))
        )
        found: dict[str, set[str]] = {}
        for kind, name in rows:
            found.setdefault(kind, set()).add(name)
        return found

    return run_db(database_url, look)


def label_volume(browser, project) -> np.ndarray:
    z, y, x = reversed(v1_volume.SHAPE_XYZ)
    volume = np.zeros((z, y, x), dtype=np.uint8)
    for cz in range(-(-z // LABEL_CHUNK_ZYX[0])):
        for cy in range(-(-y // LABEL_CHUNK_ZYX[1])):
            for cx in range(-(-x // LABEL_CHUNK_ZYX[2])):
                response = browser.get(
                    f"/api/projects/{project}/labels/zarr/class/c/{cz}/{cy}/{cx}"
                )
                if response.status_code == 404:
                    continue
                chunk = decode_chunk(response.content)
                origin = [
                    k * s for k, s in zip((cz, cy, cx), LABEL_CHUNK_ZYX, strict=True)
                ]
                part = tuple(
                    slice(o, min(o + s, n))
                    for o, s, n in zip(origin, LABEL_CHUNK_ZYX, (z, y, x), strict=True)
                )
                volume[part] = chunk[tuple(slice(0, p.stop - p.start) for p in part)]
    return volume


def test_a_worker_imports_a_claimed_v1_job(
    new_browser, settings, migrated_database_url, live_server, volume
):
    ada = new_browser()
    signup(ada)
    claimed = ada.post("/api/v1-jobs/abc123/claim")
    assert claimed.status_code == 201, claimed.text
    project = claimed.json()["project_id"]
    assert ada.get(f"/api/projects/{project}").json()["name"] == "Burrow"

    pipeline = run_import(
        migrated_database_url,
        live_server,
        ada,
        project,
        claimed.json()["pipeline_id"],
        volume,
    )
    assert pipeline["status"] == "succeeded", pipeline
    assert pipeline["kind"] == "import"
    # The worker with the v1 volume ran the import's own jobs and no others.
    for kind, names in ran_on(migrated_database_url, pipeline["id"]).items():
        assert names == {"worker-v1" if kind.startswith("v1.") else "worker-cpu"}

    # The image, in (z, y, x), with v1's voxel size.
    image = ada.get(f"/api/projects/{project}/image").json()
    x, y, z = v1_volume.SHAPE_XYZ
    assert image["manifest"]["shape_czyx"] == [1, z, y, x]
    assert image["manifest"]["voxel_size_zyx"] == list(
        reversed(v1_volume.VOXEL_SIZE_XYZ_MM)
    )
    group = zarr.open_group(
        store=zarr_store(
            project_storage(settings).child(
                f"projects/{project}/artifacts/{image['artifact_id']}"
            )
        ),
        mode="r",
    )
    stored = np.asarray(group["0"][0])  # type: ignore[index]
    np.testing.assert_array_equal(stored, v1_volume.image().transpose(2, 1, 0))

    # One class, "Foreground", and each placed sample as labels.
    classes = ada.get(f"/api/projects/{project}/labels/classes").json()
    assert [(c["value"], c["name"]) for c in classes] == [(2, "Foreground")]
    labels = label_volume(ada, project)
    expected = np.zeros_like(labels)
    for z_index, stamp in ((9, "1745400000"), (11, "1745400100-z07")):
        y0, y1, x0, x1 = v1_volume.FOREGROUND[stamp]
        expected[z_index] = BACKGROUND
        expected[z_index, y0:y1, x0:x1] = 2
    np.testing.assert_array_equal(labels, expected)

    # ... each a complete slice ROI.
    rois = ada.get(f"/api/projects/{project}/rois").json()
    assert sorted((r["bbox"], r["kind"], r["status"]) for r in rois) == [
        ([9, 0, 0, 10, y, x], "slice", "complete"),
        ([11, 0, 0, 12, y, x], "slice", "complete"),
    ]

    # The finished segmentation is the prediction: 255 is the foreground.
    prediction = ada.get(f"/api/projects/{project}/prediction").json()
    assert prediction["class_values"] == [2]
    group = zarr.open_group(
        store=zarr_store(
            project_storage(settings).child(
                f"projects/{project}/artifacts/{prediction['artifact_id']}"
            )
        ),
        mode="r",
    )
    predicted = np.asarray(group["class"][:])  # type: ignore[index]
    segmented = v1_volume.segmentation().transpose(2, 1, 0)
    np.testing.assert_array_equal(predicted, np.where(segmented > 0, 2, BACKGROUND))

    # Claiming it again gives the same project.
    again = ada.post("/api/v1-jobs/ABC123/claim")
    assert again.status_code == 200
    assert again.json() == {"project_id": project, "pipeline_id": None}


def test_the_first_claim_wins_until_an_admin_releases_it(
    new_browser, settings, migrated_database_url
):
    ada = new_browser()
    signup(ada)
    bob = new_browser()
    signup(bob, username="bob")
    assert ada.post("/api/v1-jobs/ABC123/claim").status_code == 201
    taken = bob.post("/api/v1-jobs/abc123/claim")
    assert taken.status_code == 409
    assert "admin" in taken.json()["detail"]
    assert bob.post("/api/v1-jobs/nothing/claim").status_code == 404
    assert bob.post("/api/v1-jobs/ABCDEF/claim").status_code == 404
    assert bob.post("/api/v1-jobs/DEAD00/claim").status_code == 409

    # Releasing deletes the claimer's project, so the owner can claim it.
    admin, _ = make_admin(new_browser, migrated_database_url)
    assert bob.post("/api/v1-jobs/ABC123/release").status_code == 403
    assert admin.post("/api/v1-jobs/ABC123/release").status_code == 204
    assert admin.post("/api/v1-jobs/ABC123/release").status_code == 404
    assert [p["name"] for p in ada.get("/api/projects").json()] == []
    assert bob.post("/api/v1-jobs/ABC123/claim").status_code == 201

    # So does deleting your own project.
    project = ada.post("/api/v1-jobs/FEED01/claim").json()["project_id"]
    assert ada.get(f"/api/projects/{project}").json()["name"] == "v1 job FEED01"
    assert ada.request("DELETE", f"/api/projects/{project}").status_code == 204
    assert bob.post("/api/v1-jobs/FEED01/claim").status_code == 201


def test_claims_are_limited_and_need_a_v1_volume(
    new_browser, settings, migrated_database_url
):
    limited = settings.model_copy(
        update={"v1": settings.v1.model_copy(update={"claims_per_hour": 2})}
    )
    ada = new_browser(limited)
    signup(ada)
    assert ada.post("/api/v1-jobs/000000/claim").status_code == 404
    assert ada.post("/api/v1-jobs/000001/claim").status_code == 404
    assert ada.post("/api/v1-jobs/ABC123/claim").status_code == 429

    unset = settings.model_copy(
        update={"v1": settings.v1.model_copy(update={"volume_path": None})}
    )
    bob = new_browser(unset)
    signup(bob, username="bob")
    missing = bob.post("/api/v1-jobs/ABC123/claim")
    assert missing.status_code == 404
    assert missing.json()["detail"] == "This server has no v1 jobs to import."


def test_only_import_jobs_write_labels(new_browser, settings, migrated_database_url):
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    token = add_worker(migrated_database_url)

    async def add(db):
        await jobs.enqueue(db, "noop", {"seconds": 0}, project_id=uuid.UUID(project))

    run_db(migrated_database_url, add)
    worker = new_browser()
    caps = {"version": "test", "kinds": ["noop"]}
    worker.post("/api/worker/v1/hello", json={"caps": caps}, headers=bearer(token))
    lease = worker.post(
        "/api/worker/v1/claim",
        json={"caps": caps, "wait_seconds": 0},
        headers=bearer(token),
    ).json()["job"]
    url = f"/api/worker/v1/jobs/{lease['job_id']}/label-ops"
    op = {"client_op_id": str(uuid.uuid4()), "deltas": [{"key": [0, 0, 0]}]}
    refused = worker.post(
        url, json={"lease_token": lease["lease_token"], **op}, headers=bearer(token)
    )
    assert refused.status_code == 403
    lost = worker.post(url, json={"lease_token": "nope", **op}, headers=bearer(token))
    assert lost.status_code == 409


def test_jobs_that_never_converted_cant_be_claimed(new_browser, settings, tmp_path):
    # The API reads only jobs.json.
    folder = tmp_path / "jobs-only"
    folder.mkdir()
    assert settings.v1.volume_path is not None
    shutil.copy(settings.v1.volume_path / "jobs.json", folder / "jobs.json")
    ada = new_browser(
        settings.model_copy(
            update={"v1": settings.v1.model_copy(update={"volume_path": folder})}
        )
    )
    signup(ada)
    for job_id in ("DEAD00", "DEAD01"):
        refused = ada.post(f"/api/v1-jobs/{job_id}/claim")
        assert refused.status_code == 409
        assert "never finished converting" in refused.json()["detail"]
    assert ada.post("/api/v1-jobs/ABC123/claim").status_code == 201


def test_the_probe_refuses_jobs_that_never_converted(volume, tmp_path):
    # DEAD01's conversion left part of an array behind.
    assert (volume / "chunks" / "DEAD01" / ".zarray").is_file()
    image = StorageGrant(url=(tmp_path / "image").as_uri(), access="rw")
    for job_id in ("DEAD00", "DEAD01"):
        ctx = context(volume, "v1.probe", {"job_id": job_id}, [image])
        with pytest.raises(PermanentError, match="never finished converting"):
            v1import.probe(ctx)
