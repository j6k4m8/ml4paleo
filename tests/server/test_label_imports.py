"""
Labels from a file, end to end: a browser uploads a label file straight to
storage and has it checked; a worker checks it against the image and counts
its values; the person says what each value becomes; and a worker brings
the labels in, chunk by chunk, as edits through the label writer. The upload
counts against the owner's storage until it goes, once the import ends.
"""

import base64
import datetime
import io
import threading
import time
import urllib.request
import uuid

import numpy as np
import pytest
from helpers import SECRET_KEY, add_worker, run_db, signup
from ml4paleo_server import artifacts, jobs, uploads
from ml4paleo_server.db import Job, LabelClass, UserUsage, create_sessionmaker
from ml4paleo_server.pipelines import labelimport
from ml4paleo_server.settings import Settings
from ml4paleo_worker.client import ServerClient
from ml4paleo_worker.handlers import HANDLERS
from ml4paleo_worker.handlers import labelimport as handlers
from ml4paleo_worker.main import Worker
from PIL import Image
from sqlalchemy import select, update

from ml4paleo.labels import BACKGROUND, LABEL_CHUNK_ZYX, Source
from ml4paleo.labels.codec import decode_chunk
from ml4paleo.labels.deltas import split_into_deltas
from ml4paleo.protocol import WorkerCaps

CAPS = WorkerCaps(version="test", kinds=sorted(HANDLERS))
FINISHED = ("succeeded", "failed", "cancelled")


@pytest.fixture
def settings(migrated_database_url, tmp_path, s3_endpoint, s3_bucket):
    """S3 storage, which uploads need."""
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


def make_project(browser, database_url, shape_zyx) -> str:
    """A project with an image of `shape_zyx` (only its manifest)."""
    project = browser.post("/api/projects", json={"name": "Skull"}).json()["id"]

    async def add_image(db):
        image = await artifacts.create_staging(
            db, project_id=uuid.UUID(project), kind="image", head_slot="image"
        )
        image.state = "committed"
        image.manifest = {"shape_czyx": [1, *shape_zyx]}
        await artifacts.set_head(db, image)

    run_db(database_url, add_image)
    return project


def tiff(volume: np.ndarray) -> bytes:
    """A TIFF stack, a page per slice."""
    pages = [Image.fromarray(plane) for plane in volume]
    buffer = io.BytesIO()
    pages[0].save(
        buffer,
        format="TIFF",
        save_all=True,
        append_images=pages[1:],
        compression="tiff_lzw",
    )
    return buffer.getvalue()


def upload(browser, project, data: bytes, filename="labels.tif") -> str:
    base = f"/api/projects/{project}/uploads"
    created = browser.post(base, json={"filename": filename, "size": len(data)}).json()
    urls = browser.post(f"{base}/{created['id']}/part-urls", json={"parts": [1]})
    request = urllib.request.Request(
        urls.json()["urls"]["1"],
        data=data,
        method="PUT",
        headers={"Content-Type": "application/octet-stream"},
    )
    urllib.request.urlopen(request).close()
    assert browser.post(f"{base}/{created['id']}/complete").status_code == 200
    return created["id"]


def run_worker(browser, project, import_id, database_url, live_server) -> dict:
    """Run a worker until the import's current pipeline ends; return the import."""
    url = f"/api/projects/{project}/labels/imports/{import_id}"
    client = ServerClient(
        add_worker(database_url, name=uuid.uuid4().hex[:8]), base_url=live_server
    )
    worker = Worker(client, CAPS, claim_wait_seconds=0.5, heartbeat_seconds=0.2)
    thread = threading.Thread(target=worker.run, kwargs={"max_jobs": None})
    thread.start()
    try:
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            found = browser.get(url).json()
            if found["pipeline"]["status"] in FINISHED:
                return found
            time.sleep(0.2)
        raise AssertionError("the pipeline did not finish")
    finally:
        worker.stop()
        thread.join(timeout=30)
        client.close()


def label_volume(browser, project, shape_zyx) -> np.ndarray:
    """The project's labels, read through the labels API."""
    volume = np.zeros(shape_zyx, dtype=np.uint8)
    counts = [-(-n // s) for n, s in zip(shape_zyx, LABEL_CHUNK_ZYX, strict=True)]
    for key in np.ndindex(*counts):
        response = browser.get(
            "/api/projects/{}/labels/zarr/class/c/{}/{}/{}".format(project, *key)
        )
        if response.status_code == 404:
            continue
        chunk = decode_chunk(response.content)
        part = tuple(
            slice(k * s, min((k + 1) * s, n))
            for k, s, n in zip(key, LABEL_CHUNK_ZYX, shape_zyx, strict=True)
        )
        volume[part] = chunk[tuple(slice(0, p.stop - p.start) for p in part)]
    return volume


def paint(browser, project, zyx, value):
    """Label one voxel, as the annotator would."""
    [delta] = split_into_deltas(np.ones((1, 1, 1), dtype=bool), zyx, value=value)
    op = {
        "client_op_id": str(uuid.uuid4()),
        "deltas": [
            {
                "key": list(delta.key),
                "box": list(delta.box),
                "mask": base64.b64encode(delta.mask).decode(),
                "value": value,
            }
        ],
    }
    assert (
        browser.post(f"/api/projects/{project}/labels/ops", json=op).status_code == 201
    )


def collect_uploads(settings, database_url) -> int:
    """Run garbage collection's pass over uploads, as the housekeeper does."""

    async def collect(db):
        return await uploads.collect_garbage(create_sessionmaker(db.bind), settings)

    return run_db(database_url, collect)


def usage(database_url) -> int:
    async def read(db):
        return await db.scalar(select(UserUsage.storage_bytes))

    return run_db(database_url, read)


def test_a_label_file_is_checked_and_imported_as_edits(
    new_browser, settings, migrated_database_url, live_server
):
    ada = new_browser()
    signup(ada)
    me = ada.get("/api/auth/session").json()["user"]["id"]
    shape = (70, 40, 130)
    project = make_project(ada, migrated_database_url, shape)
    classes = f"/api/projects/{project}/labels/classes"
    bone = ada.post(classes, json={"name": "Bone", "color": "#e5484d"}).json()["value"]
    rng = np.random.default_rng(0)
    volume = rng.choice(np.array([0, 0, 0, 1, 7, 300], np.uint16), size=shape)
    volume[0, 0, 0] = 7
    # Someone labeled a voxel before the import: it keeps their label.
    paint(ada, project, (0, 0, 0), BACKGROUND)
    data = tiff(volume)
    upload_id = upload(ada, project, data)
    assert usage(migrated_database_url) == len(data)

    started = ada.post(
        f"/api/projects/{project}/labels/imports", json={"upload_id": upload_id}
    )
    assert started.status_code == 202
    assert started.json()["state"] == "checking"
    assert started.json()["pipeline"]["kind"] == "label check"
    # Checking the same upload again gives the same check.
    again = ada.post(
        f"/api/projects/{project}/labels/imports", json={"upload_id": upload_id}
    )
    assert again.json()["id"] == started.json()["id"]
    import_id = started.json()["id"]

    checked = run_worker(ada, project, import_id, migrated_database_url, live_server)
    assert checked["state"] == "ready", checked
    assert checked["shape_zyx"] == list(shape)
    values, voxels = np.unique(volume, return_counts=True)
    assert checked["values"] == [
        {"value": int(v), "voxels": int(n)} for v, n in zip(values, voxels, strict=True)
    ]

    # 7 becomes Bone, 300 a new class, 1 background; 0 stays unlabeled.
    url = f"/api/projects/{project}/labels/imports/{import_id}"
    mapping = [
        {"value": 7, "label": bone},
        {"value": 300, "new_class": {"name": "Shell", "color": "#46a758"}},
        {"value": 1, "label": BACKGROUND},
    ]
    importing = ada.post(f"{url}/start", json={"mapping": mapping})
    assert importing.status_code == 202, importing.text
    assert importing.json()["state"] == "importing"
    assert importing.json()["pipeline"]["kind"] == "label import"
    shell = next(c["value"] for c in ada.get(classes).json() if c["name"] == "Shell")
    assert importing.json()["lookup"] == [[1, BACKGROUND], [7, bone], [300, shell]]
    # Once is enough.
    assert ada.post(f"{url}/start", json={"mapping": mapping}).status_code == 409

    done = run_worker(ada, project, import_id, migrated_database_url, live_server)
    assert done["state"] == "done", done
    expected = np.select(
        [volume == 7, volume == 300, volume == 1], [bone, shell, BACKGROUND], 0
    ).astype(np.uint8)
    expected[0, 0, 0] = BACKGROUND
    np.testing.assert_array_equal(label_volume(ada, project, shape), expected)

    # One edit a chunk, from the file, by ada.
    history = ada.get(f"/api/projects/{project}/labels/ops?limit=500").json()
    imported = [op for op in history if op["tool"].get("name") == "label-import"]
    assert len(imported) == 2 * 1 * 3
    assert all(op["source"] == Source.IMPORTED for op in imported)
    assert {op["user_id"] for op in imported} == {me}
    # The history names who imported them, and the job that brought them in.
    assert {(op["username"], op["job_kind"]) for op in imported} == {
        ("ada", "labels.import")
    }
    assert imported[0]["tool"] == {
        "name": "label-import",
        "import": import_id,
        "file": "labels.tif",
    }

    # The upload goes once the import is done, and its storage with it.
    assert collect_uploads(settings, migrated_database_url) == 1
    assert usage(migrated_database_url) == 0
    assert ada.get(url).json()["state"] == "done"


def test_labels_of_another_size_are_refused(
    new_browser, settings, migrated_database_url, live_server
):
    ada = new_browser()
    signup(ada)
    project = make_project(ada, migrated_database_url, (5, 6, 7))
    upload_id = upload(ada, project, tiff(np.zeros((4, 6, 7), np.uint8)))
    import_id = ada.post(
        f"/api/projects/{project}/labels/imports", json={"upload_id": upload_id}
    ).json()["id"]
    failed = run_worker(ada, project, import_id, migrated_database_url, live_server)
    assert failed["state"] == "failed"
    assert failed["error"].startswith(
        "The labels are 7 × 6 × 4 voxels, but the image is 7 × 6 × 5 (x × y × z)."
    )
    url = f"/api/projects/{project}/labels/imports/{import_id}/start"
    refused = ada.post(url, json={"mapping": [{"value": 0, "label": 1}]})
    assert refused.status_code == 409


def test_a_retried_import_sends_each_chunk_once(
    new_browser, settings, migrated_database_url, live_server, monkeypatch
):
    ada = new_browser()
    signup(ada)
    shape = (3, 70, 130)
    project = make_project(ada, migrated_database_url, shape)
    volume = np.full(shape, 5, dtype=np.uint8)
    upload_id = upload(ada, project, tiff(volume))
    base = f"/api/projects/{project}/labels/imports"
    import_id = ada.post(base, json={"upload_id": upload_id}).json()["id"]
    run_worker(ada, project, import_id, migrated_database_url, live_server)

    # The first try fails after two of the six chunks; the second finishes.
    monkeypatch.setattr(jobs.queue, "FIRST_RETRY", datetime.timedelta(0))
    tries = []

    def flaky(ctx):
        tries.append(ctx.lease.attempt)
        if len(tries) == 1:
            send, sent = ctx.apply_label_op, []

            def apply(op):
                if len(sent) == 2:
                    raise RuntimeError("The network went away.")
                sent.append(op)
                return send(op)

            ctx.apply_label_op = apply
        return handlers.run(ctx)

    monkeypatch.setitem(HANDLERS, "labels.import", flaky)
    mapping = [{"value": 5, "new_class": {"name": "Bone", "color": "#e5484d"}}]
    ada.post(f"{base}/{import_id}/start", json={"mapping": mapping, "overwrite": True})
    done = run_worker(ada, project, import_id, migrated_database_url, live_server)
    assert done["state"] == "done", done
    assert tries == [1, 2]
    history = ada.get(f"/api/projects/{project}/labels/ops").json()
    assert len(history) == 6
    assert (label_volume(ada, project, shape) == 2).all()


def test_overwriting_replaces_labels_already_there(
    new_browser, settings, migrated_database_url, live_server
):
    ada = new_browser()
    signup(ada)
    project = make_project(ada, migrated_database_url, (2, 3, 4))
    paint(ada, project, (1, 2, 3), BACKGROUND)
    paint(ada, project, (0, 0, 0), BACKGROUND)
    volume = np.zeros((2, 3, 4), np.uint8)
    volume[1, 2, 3] = 9
    upload_id = upload(ada, project, tiff(volume))
    base = f"/api/projects/{project}/labels/imports"
    import_id = ada.post(base, json={"upload_id": upload_id}).json()["id"]
    run_worker(ada, project, import_id, migrated_database_url, live_server)
    mapping = [{"value": 9, "new_class": {"name": "Tooth", "color": "#f2c14e"}}]
    ada.post(f"{base}/{import_id}/start", json={"mapping": mapping, "overwrite": True})
    done = run_worker(ada, project, import_id, migrated_database_url, live_server)
    assert done["state"] == "done" and done["overwrite"] is True
    labels = label_volume(ada, project, (2, 3, 4))
    # The file's 0 is left alone; its 9 replaces the background there.
    assert labels[1, 2, 3] == 2 and labels[0, 0, 0] == BACKGROUND


def test_what_values_can_become(new_browser, settings, migrated_database_url):
    ada = new_browser()
    signup(ada)
    project = make_project(ada, migrated_database_url, (2, 3, 4))
    upload_id = upload(ada, project, tiff(np.ones((2, 3, 4), np.uint8)))
    base = f"/api/projects/{project}/labels/imports"
    import_id = ada.post(base, json={"upload_id": upload_id}).json()["id"]
    url = f"{base}/{import_id}/start"
    # Not until the file is checked.
    assert (
        ada.post(url, json={"mapping": [{"value": 1, "label": 1}]}).status_code == 409
    )

    async def checked(db):
        await db.execute(
            update(Job)
            .where(Job.id == uuid.UUID(import_id))
            .values(
                status="succeeded",
                result={
                    "format": "tiff",
                    "shape_zyx": [2, 3, 4],
                    "values": [[1, 12], [2, 12]],
                },
            )
        )

    run_db(migrated_database_url, checked)
    assert ada.get(f"{base}/{import_id}").json()["state"] == "ready"
    for mapping in [
        [{"value": 3, "label": 1}],  # not in the file
        [{"value": 1, "label": 5}],  # not a class here
        [{"value": 1, "label": 0}],  # leave it out instead
        [{"value": 1, "label": 1}, {"value": 1, "label": 1}],
        [{"value": 1}],
        [{"value": 1, "label": 1, "new_class": {"name": "A", "color": "#000000"}}],
        [],
    ]:
        assert ada.post(url, json={"mapping": mapping}).status_code == 422, mapping

    # A project with room for one more class can't make two.
    async def fill(db):
        for value in range(2, 254):
            db.add(
                LabelClass(
                    project_id=uuid.UUID(project),
                    value=value,
                    name=str(value),
                    color="#000000",
                )
            )

    run_db(migrated_database_url, fill)
    new = {"name": "Bone", "color": "#e5484d"}
    two = [{"value": 1, "new_class": new}, {"value": 2, "new_class": new}]
    refused = ada.post(url, json={"mapping": two})
    assert refused.status_code == 409
    assert refused.json()["detail"] == "This project has room for only 1 more class."
    assert ada.post(url, json={"mapping": two[:1]}).status_code == 202
    assert ada.post(url, json={"mapping": two[:1]}).status_code == 409

    # Discarding lets the file go; a discarded file can't import.
    other = upload(ada, project, tiff(np.ones((2, 3, 4), np.uint8)), "b.tif")
    second = ada.post(base, json={"upload_id": other}).json()["id"]
    assert ada.request("DELETE", f"{base}/{second}").status_code == 204
    found = ada.get(f"{base}/{second}").json()
    assert found["state"] == "cancelled"
    assert collect_uploads(settings, migrated_database_url) == 1
    bob = new_browser()
    signup(bob, username="bob")
    assert bob.get(base).status_code == 404
    assert [i["id"] for i in ada.get(base).json()] == [second, import_id]


def test_check_results_are_checked():
    good = {"format": "zip", "shape_zyx": [1, 2, 3], "values": [[0, 4], [3, 2]]}
    labelimport.check_probe_result(good)
    for bad in [
        {**good, "format": "nrrd"},
        {**good, "shape_zyx": [1, 2]},
        {**good, "shape_zyx": [0, 2, 3]},
        {**good, "values": [[3, 2], [0, 4]]},
        {**good, "values": [[0, 4], [0, 4]]},
        {**good, "values": [[0, 0]]},
        {**good, "values": [[0.5, 1]]},
        {**good, "values": [[v, 1] for v in range(300)]},
    ]:
        with pytest.raises(ValueError):
            labelimport.check_probe_result(bad)


def test_an_import_says_why_the_server_refused_its_labels(
    new_browser, settings, migrated_database_url, live_server
):
    ada = new_browser()
    signup(ada)
    project = make_project(ada, migrated_database_url, (2, 3, 4))
    upload_id = upload(ada, project, tiff(np.full((2, 3, 4), 4, np.uint8)))
    base = f"/api/projects/{project}/labels/imports"
    import_id = ada.post(base, json={"upload_id": upload_id}).json()["id"]
    run_worker(ada, project, import_id, migrated_database_url, live_server)
    mapping = [{"value": 4, "new_class": {"name": "Bone", "color": "#e5484d"}}]
    ada.post(f"{base}/{import_id}/start", json={"mapping": mapping})
    # The class goes before the labels come in.
    classes = f"/api/projects/{project}/labels/classes"
    assert ada.request("DELETE", f"{classes}/2").status_code == 204
    failed = run_worker(ada, project, import_id, migrated_database_url, live_server)
    assert failed["state"] == "failed"
    assert failed["error"] == (
        "The server refused the labels at z 0, y 0, x 0: label values [2] are not "
        "classes here"
    )
