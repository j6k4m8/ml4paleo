"""
The ingest pipeline end to end: a browser uploads a zip of slices straight to
storage (an in-process S3 server) and starts ingest, a worker runs every job
through the storage proxy, progress streams to the browser, and the project
ends up with its image.
"""

import io
import json
import threading
import time
import urllib.request
import zipfile

import numpy as np
import pytest
from helpers import SECRET_KEY, add_worker, signup
from ml4paleo_server.settings import Settings
from ml4paleo_server.storage import project_storage
from ml4paleo_worker.client import ServerClient
from ml4paleo_worker.handlers import HANDLERS
from ml4paleo_worker.main import Worker
from PIL import Image

from ml4paleo.ome import OmeImage
from ml4paleo.protocol import WorkerCaps

CAPS = WorkerCaps(version="test", kinds=sorted(HANDLERS))


@pytest.fixture
def settings(migrated_database_url, tmp_path, s3_endpoint, s3_bucket):
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


def _zip(entries: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def _png(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def upload(browser, project: str, data: bytes, filename="scan.zip") -> str:
    base = f"/api/projects/{project}/uploads"
    created = browser.post(base, json={"filename": filename, "size": len(data)}).json()
    url = browser.post(f"{base}/{created['id']}/part-urls", json={"parts": [1]})
    request = urllib.request.Request(
        url.json()["urls"]["1"],
        data=data,
        method="PUT",
        headers={"Content-Type": "application/octet-stream"},
    )
    urllib.request.urlopen(request).close()
    assert (
        browser.post(f"{base}/{created['id']}/complete").json()["state"] == "complete"
    )
    return created["id"]


def run_until_finished(browser, project, pipeline_id, token, live_server):
    client = ServerClient(token, base_url=live_server)
    worker = Worker(client, CAPS, claim_wait_seconds=0.5, heartbeat_seconds=0.2)
    thread = threading.Thread(target=worker.run, kwargs={"max_jobs": None})
    thread.start()
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            status = browser.get(f"/api/projects/{project}/pipelines/{pipeline_id}")
            if status.json()["status"] in ("succeeded", "failed", "cancelled"):
                return status.json()
            time.sleep(0.2)
        raise AssertionError("the pipeline did not finish")
    finally:
        worker.stop()
        thread.join(timeout=30)
        client.close()


def events(browser, project, pipeline_id) -> list[dict]:
    """
    Read the pipeline's server-sent events until the stream ends.
    """
    statuses = []
    with browser.client.stream(
        "GET", f"/api/projects/{project}/pipelines/{pipeline_id}/events"
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                statuses.append(json.loads(line[len("data: ") :]))
    return statuses


def test_an_upload_becomes_the_projects_image(
    new_browser, settings, migrated_database_url, live_server
):
    browser = new_browser()
    signup(browser)
    project = browser.post("/api/projects", json={"name": "Skull"}).json()["id"]
    rng = np.random.default_rng(0)
    # 600 slices: two shard-deep slabs and several pyramid levels.
    slices = [rng.integers(0, 60000, (5, 7), dtype=np.uint16) for _ in range(600)]
    archive = _zip({f"stack/z{z}.png": _png(s) for z, s in enumerate(slices)})
    upload_id = upload(browser, project, archive)
    assert browser.get(f"/api/projects/{project}/image").status_code == 404

    started = browser.post(
        f"/api/projects/{project}/ingest", json={"upload_id": upload_id}
    )
    assert started.status_code == 202
    pipeline = started.json()
    assert (pipeline["kind"], pipeline["status"]) == ("ingest", "waiting")

    # Watch progress while the pipeline runs.
    watched: list[dict] = []
    watcher = threading.Thread(
        target=lambda: watched.extend(events(browser, project, pipeline["id"]))
    )
    watcher.start()
    token = add_worker(migrated_database_url)
    done = run_until_finished(browser, project, pipeline["id"], token, live_server)
    assert done["status"] == "succeeded", done
    assert done["progress"] == 1.0
    watcher.join(timeout=30)
    assert len(watched) >= 2 and watched[-1]["status"] == "succeeded"
    progress = [status["progress"] for status in watched]
    assert progress == sorted(progress)
    # probe, 2 slabs, the pyramid levels, and finalize
    image = browser.get(f"/api/projects/{project}/image").json()
    manifest = image["manifest"]
    assert manifest["shape_czyx"] == [1, 600, 5, 7]
    assert done["jobs"] == 1 + 2 + (manifest["levels"] - 1) + 1
    assert manifest["levels"] >= 2
    assert manifest["source"]["filename"] == "scan.zip"
    assert manifest["window"][0] < manifest["window"][1]

    stored = OmeImage.open(
        project_storage(settings).child(
            f"projects/{project}/artifacts/{image['artifact_id']}"
        )
    )
    np.testing.assert_array_equal(np.asarray(stored.array(0)[0]), np.stack(slices))
    assert stored.num_levels == manifest["levels"]

    # A finished pipeline has no events left: 204 tells EventSource to stop.
    finished = browser.get(f"/api/projects/{project}/pipelines/{pipeline['id']}/events")
    assert finished.status_code == 204
    listed = browser.get(f"/api/projects/{project}/pipelines").json()
    assert [p["id"] for p in listed] == [pipeline["id"]]


def test_a_bad_upload_fails_with_a_reason(
    new_browser, migrated_database_url, live_server
):
    browser = new_browser()
    signup(browser)
    project = browser.post("/api/projects", json={"name": "Oops"}).json()["id"]
    upload_id = upload(browser, project, b"this is not a zip file", "notes.txt")
    pipeline = browser.post(
        f"/api/projects/{project}/ingest", json={"upload_id": upload_id}
    ).json()
    token = add_worker(migrated_database_url)
    done = run_until_finished(browser, project, pipeline["id"], token, live_server)
    assert done["status"] == "failed"
    assert done["error"].startswith("The upload is not a zip archive")
    # Not retried: the upload won't change.
    assert browser.get(f"/api/projects/{project}/image").status_code == 404


def test_ingest_needs_a_finished_upload_in_this_project(new_browser):
    browser = new_browser()
    signup(browser)
    project = browser.post("/api/projects", json={"name": "Skull"}).json()["id"]
    unfinished = browser.post(
        f"/api/projects/{project}/uploads", json={"filename": "a.zip", "size": 10}
    ).json()
    for upload_id in [unfinished["id"], "00000000-0000-0000-0000-000000000000"]:
        response = browser.post(
            f"/api/projects/{project}/ingest", json={"upload_id": upload_id}
        )
        assert response.status_code == 404


def test_a_malformed_probe_result_fails_the_pipeline(
    new_browser, migrated_database_url, live_server, monkeypatch
):
    browser = new_browser()
    signup(browser)
    project = browser.post("/api/projects", json={"name": "Skull"}).json()["id"]
    upload_id = upload(
        browser, project, _zip({"a.png": _png(np.zeros((4, 4), np.uint8))})
    )
    pipeline = browser.post(
        f"/api/projects/{project}/ingest", json={"upload_id": upload_id}
    ).json()
    # A probe that reports slabs that don't cover the volume.
    monkeypatch.setitem(
        HANDLERS,
        "ingest.probe",
        lambda ctx: {
            "kind": "images",
            "slices": 1,
            "shape_zyx": [3, 4, 4],
            "dtype": "|u1",
            "levels": 1,
            "slabs": [[0, 2]],
        },
    )
    token = add_worker(migrated_database_url)
    done = run_until_finished(browser, project, pipeline["id"], token, live_server)
    assert done["status"] == "failed"
    assert done["error"].startswith("The job's result is malformed")
    assert done["jobs"] == 1  # nothing was built on it


def test_the_event_stream_stops_when_access_ends(new_browser, migrated_database_url):
    ada = new_browser()
    signup(ada)
    bob = new_browser()
    signup(bob, username="bob")
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    members = ada.post(f"/api/projects/{project}/members", json={"username": "bob"})
    bob_id = next(m["user_id"] for m in members.json() if m["username"] == "bob")
    upload_id = upload(ada, project, _zip({"a.png": _png(np.zeros((4, 4), np.uint8))}))
    pipeline = ada.post(
        f"/api/projects/{project}/ingest", json={"upload_id": upload_id}
    ).json()
    # No worker runs, so the pipeline stays waiting; bob watches it...
    watched: list[dict] = []
    watcher = threading.Thread(
        target=lambda: watched.extend(events(bob, project, pipeline["id"]))
    )
    watcher.start()
    time.sleep(1.5)
    # ...until he is removed from the project.
    removed = ada.request("DELETE", f"/api/projects/{project}/members/{bob_id}")
    assert removed.status_code == 204
    watcher.join(timeout=10)
    assert not watcher.is_alive()
    assert watched and watched[0]["status"] == "waiting"


def test_an_image_is_not_abandoned_while_its_probe_waits(
    new_browser, settings, migrated_database_url
):
    import datetime

    from helpers import run_db
    from ml4paleo_server import artifacts
    from ml4paleo_server.db import Artifact
    from sqlalchemy import select, update

    browser = new_browser()
    signup(browser)
    project = browser.post("/api/projects", json={"name": "Skull"}).json()["id"]
    upload_id = upload(
        browser, project, _zip({"a.png": _png(np.zeros((4, 4), np.uint8))})
    )
    browser.post(f"/api/projects/{project}/ingest", json={"upload_id": upload_id})

    async def age_and_sweep(db):
        old = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=3)
        await db.execute(update(Artifact).values(created_at=old))
        await artifacts.abandon_staging(db)
        return await db.scalar(select(Artifact.state))

    assert run_db(migrated_database_url, age_and_sweep) == "staging"
