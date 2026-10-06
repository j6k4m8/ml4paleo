"""
Exports, end to end: a worker zips the image's slices, the final
segmentation's zarr group, and the meshes; the API serves each archive from
its parts with byte ranges, hands out the same one when asked again, and
lets it go when deleted.
"""

import datetime
import io
import json
import pathlib
import tempfile
import threading
import time
import uuid
import zipfile

import numpy as np
import pytest
import zarr
from helpers import SECRET_KEY, add_worker, bearer, run_db, signup
from ml4paleo_server import artifacts
from ml4paleo_server.db import Artifact, create_sessionmaker
from ml4paleo_server.settings import Settings
from ml4paleo_server.storage import project_storage
from ml4paleo_worker.client import ServerClient
from ml4paleo_worker.context import JobContext
from ml4paleo_worker.handlers import HANDLERS
from ml4paleo_worker.handlers.export import _finish, _mesh_names
from ml4paleo_worker.main import Worker
from PIL import Image
from sqlalchemy import select, update

import ml4paleo.export
from ml4paleo.ome import OmeImage
from ml4paleo.protocol import JobLease, WorkerCaps
from ml4paleo.segmentation.predict import create_prediction
from ml4paleo.storage import StorageGrant, get_bytes, put_bytes, write_manifest

SHAPE = (6, 10, 14)
BONE = 2


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


def image() -> np.ndarray:
    return (np.arange(np.prod(SHAPE)).reshape(SHAPE) * 37 % 4000).astype(np.uint16)


def classes() -> np.ndarray:
    values = np.ones(SHAPE, dtype=np.uint8)
    values[1:4, 2:8, 3:9] = BONE
    return values


def add_heads(settings, database_url, project: str):
    async def create(db):
        pid = uuid.UUID(project)

        def grant(artifact):
            return project_storage(settings).child(artifacts.artifact_path(artifact))

        head = await artifacts.create_staging(
            db, project_id=pid, kind="image", head_slot="image"
        )
        OmeImage.create(grant(head), shape_czyx=(1, *SHAPE), dtype=np.uint16).array(0)[
            0
        ] = image()
        head.state = "committed"
        head.bytes = 5000
        head.manifest = {
            "shape_czyx": [1, *SHAPE],
            "dtype": "<u2",
            "window": [0, 4000],
        }
        await artifacts.set_head(db, head)

        head = await artifacts.create_staging(
            db, project_id=pid, kind="segmentation", head_slot="segmentation"
        )
        group = create_prediction(
            grant(head), SHAPE, arrays=("class",), kind="segmentation"
        )
        group["class"][:] = classes()  # type: ignore[index]
        put_bytes(grant(head), "inputs.json", b"{}")
        write_manifest(grant(head), {"kind": "segmentation"})
        head.state = "committed"
        head.manifest = {"kind": "segmentation", "shape_zyx": list(SHAPE)}
        await artifacts.set_head(db, head)

        head = await artifacts.create_staging(
            db, project_id=pid, kind="meshes", head_slot="meshes"
        )
        files = {"stl": "2.stl", "obj": "2.obj", "glb": "2.glb"}
        for extension, key in files.items():
            put_bytes(grant(head), key, f"a {extension} mesh".encode())
        # What a join leaves for finalize, which isn't exported.
        put_bytes(grant(head), "2.json", b"{}")
        info = {
            "axis_order": "xyz",
            "classes": [
                {"value": BONE, "name": "Bone (left)", "color": "#fff", "files": files}
            ],
        }
        put_bytes(grant(head), "mesh_info.json", json.dumps(info).encode())
        head.state = "committed"
        head.manifest = {"kind": "meshes", **info}
        await artifacts.set_head(db, head)

    run_db(database_url, create)


def run_worker(database_url, live_server, browser, project, pipeline_ids):
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
            statuses = [
                browser.get(f"/api/projects/{project}/pipelines/{p}").json()["status"]
                for p in pipeline_ids
            ]
            if all(s in ("succeeded", "failed", "cancelled") for s in statuses):
                return statuses
            time.sleep(0.3)
        raise AssertionError("The exports never finished")
    finally:
        worker.stop()
        thread.join(timeout=30)
        client.close()


def test_a_worker_exports_volumes_and_meshes(
    new_browser, settings, migrated_database_url, live_server, monkeypatch
):
    # Small parts, so downloads cross from one part to the next.
    monkeypatch.setattr(ml4paleo.export, "PART_BYTES", 1000)
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    base = f"/api/projects/{project}/exports"
    assert ada.post(base, json={"source": "image", "format": "png"}).status_code == 409
    add_heads(settings, migrated_database_url, project)
    assert ada.post(base, json={"source": "meshes", "format": "png"}).status_code == 422
    assert ada.post(base, json={"source": "nope", "format": "png"}).status_code == 422
    assert (
        ada.post(base, json={"source": "prediction", "format": "zarr"}).status_code
        == 409
    )

    asked = [
        {"source": "image", "format": "png"},
        {"source": "segmentation", "format": "zarr"},
        {"source": "meshes", "format": "zip"},
    ]
    started = [ada.post(base, json=body) for body in asked]
    assert [r.status_code for r in started] == [202, 202, 202], started[0].text
    exports = [r.json() for r in started]
    assert [e["status"] for e in exports] == ["making"] * 3
    # Asking again while one is being made gives that one.
    again = ada.post(base, json=asked[0])
    assert (again.status_code, again.json()["id"]) == (202, exports[0]["id"])

    statuses = run_worker(
        migrated_database_url,
        live_server,
        ada,
        project,
        [e["pipeline_id"] for e in exports],
    )
    assert statuses == ["succeeded"] * 3

    listed = {e["id"]: e for e in ada.get(base).json()}
    assert all(listed[e["id"]]["status"] == "ready" for e in exports)
    png, segmentation, meshes = (listed[e["id"]] for e in exports)
    assert png["filename"] == "skull-image-png.zip"
    assert segmentation["filename"] == "skull-segmentation.zarr.zip"
    assert meshes["filename"] == "skull-meshes.zip"

    # One PNG per z, as the stack was uploaded: y down the rows.
    whole = ada.get(png["download_url"])
    assert whole.status_code == 200
    assert whole.headers["content-type"] == "application/zip"
    assert 'filename="skull-image-png.zip"' in whole.headers["content-disposition"]
    assert int(whole.headers["content-length"]) == png["bytes"] == len(whole.content)
    with zipfile.ZipFile(io.BytesIO(whole.content)) as zipped:
        names = zipped.namelist()
        assert names == [f"skull-image-png/z{z:05d}.png" for z in range(SHAPE[0])]
        plane = np.asarray(Image.open(io.BytesIO(zipped.read(names[4]))))
        np.testing.assert_array_equal(plane, image()[4])

    # Byte ranges, across parts (of 1000 bytes here).
    content = whole.content
    assert len(content) > 1200
    for header, expected in (
        ("bytes=10-1199", content[10:1200]),
        ("bytes=950-", content[950:]),
        ("bytes=-22", content[-22:]),
    ):
        part = ada.get(png["download_url"], headers={"Range": header})
        assert part.status_code == 206, header
        assert part.content == expected, header
    beyond = ada.get(png["download_url"], headers={"Range": f"bytes={len(content)}-"})
    assert beyond.status_code == 416

    # The zarr group sits at the archive's root, so zarr reads the zip.
    raw = ada.get(segmentation["download_url"]).content
    with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
        assert "inputs.json" not in zipped.namelist()
        assert "_MANIFEST.json" not in zipped.namelist()
    with tempfile.TemporaryDirectory() as scratch:
        archive = pathlib.Path(scratch) / "segmentation.zarr.zip"
        archive.write_bytes(raw)
        store = zarr.storage.ZipStore(archive, mode="r")
        group = zarr.open_group(store=store, mode="r")
        np.testing.assert_array_equal(np.asarray(group["class"][:]), classes())
        store.close()

    # Meshes are named for their classes.
    with zipfile.ZipFile(io.BytesIO(ada.get(meshes["download_url"]).content)) as zipped:
        assert sorted(zipped.namelist()) == [
            "skull-meshes/bone-left.glb",
            "skull-meshes/bone-left.obj",
            "skull-meshes/bone-left.stl",
            "skull-meshes/mesh_info.json",
        ]
        info = json.loads(zipped.read("skull-meshes/mesh_info.json"))
        assert info["classes"][0]["files"]["stl"] == "bone-left.stl"
        assert zipped.read("skull-meshes/bone-left.stl") == b"a stl mesh"

    # Asking again gives the kept one, for another week.
    again = ada.post(base, json=asked[0])
    assert (again.status_code, again.json()["id"]) == (200, png["id"])
    assert again.json()["expires_at"] >= png["expires_at"]

    # Deleting lets it go.
    assert ada.request("DELETE", f"{base}/{png['id']}").status_code == 204
    assert png["id"] not in {e["id"] for e in ada.get(base).json()}
    assert ada.get(png["download_url"]).status_code == 404
    # Until garbage collection deletes it, asking again brings it back.
    revived = ada.post(base, json=asked[0])
    assert (revived.status_code, revived.json()["id"]) == (200, png["id"])


def test_others_cant_reach_exports(new_browser, settings, migrated_database_url):
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    add_heads(settings, migrated_database_url, project)
    export = ada.post(
        f"/api/projects/{project}/exports", json={"source": "image", "format": "tiff"}
    ).json()
    bob = new_browser()
    signup(bob, username="bob")
    base = f"/api/projects/{project}/exports"
    assert bob.get(base).status_code == 404
    assert bob.post(base, json={"source": "image", "format": "tiff"}).status_code == 404
    assert bob.request("DELETE", f"{base}/{export['id']}").status_code == 404
    assert bob.get(f"{base}/{export['id']}/download").status_code == 404


def project_with_heads(
    new_browser, settings, database_url, browser_settings=None, username="ada"
):
    ada = new_browser(browser_settings) if browser_settings else new_browser()
    signup(ada, username=username)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    add_heads(settings, database_url, project)
    return ada, project


def test_stopping_an_export_then_asking_again_starts_over(
    new_browser, settings, migrated_database_url
):
    ada, project = project_with_heads(new_browser, settings, migrated_database_url)
    base = f"/api/projects/{project}/exports"
    first = ada.post(base, json={"source": "image", "format": "tiff"}).json()
    # A worker has it when it's stopped, so it's still running for a moment.
    token = add_worker(migrated_database_url)
    worker = new_browser()
    caps = {"version": "test", "kinds": ["export.images"]}
    worker.post("/api/worker/v1/hello", json={"caps": caps}, headers=bearer(token))
    lease = worker.post(
        "/api/worker/v1/claim",
        json={"caps": caps, "wait_seconds": 0},
        headers=bearer(token),
    ).json()["job"]
    assert lease["job_id"] == first["pipeline_id"]
    assert ada.request("DELETE", f"{base}/{first['id']}").status_code == 204
    again = ada.post(base, json={"source": "image", "format": "tiff"})
    assert again.status_code == 202
    assert again.json()["id"] != first["id"]


def test_exports_that_cant_work_are_refused_up_front(
    new_browser, settings, migrated_database_url
):
    ada, project = project_with_heads(new_browser, settings, migrated_database_url)
    base = f"/api/projects/{project}/exports"

    async def float_image(db):
        image = await artifacts.head(db, uuid.UUID(project), "image")
        assert image is not None
        image.manifest = {**(image.manifest or {}), "dtype": "<f4"}

    run_db(migrated_database_url, float_image)
    refused = ada.post(base, json={"source": "image", "format": "png"})
    assert refused.status_code == 422
    assert "TIFF" in refused.json()["detail"]

    full = settings.model_copy(
        update={"quota": settings.quota.model_copy(update={"storage_gb": 0})}
    )
    bob, other = project_with_heads(
        new_browser,
        settings,
        migrated_database_url,
        browser_settings=full,
        username="bob",
    )
    too_big = bob.post(
        f"/api/projects/{other}/exports", json={"source": "image", "format": "zarr"}
    )
    assert too_big.status_code == 403
    assert "storage left" in too_big.json()["detail"]


def test_downloads_keep_their_export_and_deletes_wait_a_little(
    new_browser, settings, migrated_database_url, live_server
):
    ada, project = project_with_heads(new_browser, settings, migrated_database_url)
    base = f"/api/projects/{project}/exports"
    export = ada.post(base, json={"source": "segmentation", "format": "zarr"}).json()
    run_worker(
        migrated_database_url, live_server, ada, project, [export["pipeline_id"]]
    )
    export_id = uuid.UUID(export["id"])

    async def expire_soon(db):
        await db.execute(
            update(Artifact)
            .where(Artifact.id == export_id)
            .values(expires_at=artifacts.now() + datetime.timedelta(minutes=1))
        )

    async def expiry(db):
        return await db.scalar(
            select(Artifact.expires_at).where(Artifact.id == export_id)
        )

    run_db(migrated_database_url, expire_soon)
    ready = next(e for e in ada.get(base).json() if e["id"] == export["id"])
    assert ada.get(ready["download_url"]).status_code == 200
    kept = run_db(migrated_database_url, expiry)
    assert kept > artifacts.now() + datetime.timedelta(hours=5)

    # Deleted, it's gone from the list at once, but its files stay a while.
    assert ada.request("DELETE", f"{base}/{export['id']}").status_code == 204

    async def collect(db):
        return await artifacts.collect_garbage(create_sessionmaker(db.bind), settings)

    async def state(db):
        return await db.scalar(select(Artifact.state).where(Artifact.id == export_id))

    run_db(migrated_database_url, collect)
    assert run_db(migrated_database_url, state) == "committed"

    async def long_ago(db):
        await db.execute(
            update(Artifact)
            .where(Artifact.id == export_id)
            .values(expires_at=artifacts.now() - datetime.timedelta(hours=1))
        )

    run_db(migrated_database_url, long_ago)
    run_db(migrated_database_url, collect)
    assert run_db(migrated_database_url, state) == "deleted"


def test_mesh_names_never_collide(tmp_path):
    grant = StorageGrant(url=tmp_path.as_uri(), access="rw")
    classes = [
        ("bone-3", 1),
        ("Bone", 2),
        ("bone", 3),
        ("Class 7", 4),
        ("???", 7),
        ("", 8),
        ("हड्डी", 9),
    ]
    info = {
        "classes": [
            {"value": v, "name": n, "files": {"stl": f"{v}.stl"}} for n, v in classes
        ]
    }
    put_bytes(grant, "mesh_info.json", json.dumps(info).encode())
    names, rewritten = _mesh_names(grant)
    assert len(set(names.values())) == len(classes)
    # Names in other scripts keep their letters and marks.
    assert names["9.stl"] == "हड्डी.stl"
    files = [c["files"]["stl"] for c in json.loads(rewritten)["classes"]]
    assert files == [names[f"{v}.stl"] for _, v in classes]


def test_a_shorter_retry_removes_the_parts_past_its_end(tmp_path):
    export = StorageGrant(url=(tmp_path / "export").as_uri(), access="rw")
    for index in range(5):
        put_bytes(export, ml4paleo.export.part_key(index), b"old")
    lease = JobLease(
        job_id=uuid.uuid4(),
        kind="export.files",
        payload={"format": "zarr", "source": "segmentation"},
        lease_token="t",
        lease_expires_at=artifacts.now(),
        attempt=2,
        grants=[StorageGrant(url=(tmp_path / "source").as_uri(), access="r"), export],
    )
    out = ml4paleo.export.Parts(
        lambda index, data: put_bytes(export, ml4paleo.export.part_key(index), data),
        part_bytes=4,
    )
    out.write(b"0123456789")
    _finish(JobContext(lease), out, entries=1)
    kept = [get_bytes(export, ml4paleo.export.part_key(i)) for i in range(5)]
    assert kept == [b"0123", b"4567", b"89", None, None]
