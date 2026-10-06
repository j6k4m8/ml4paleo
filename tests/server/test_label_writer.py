"""
The label writer and its API: classes, edits, strict edits, undo and redo,
the change feed, the labels as zarr, and ROIs.
"""

import asyncio
import base64
import json
import threading
import time
import uuid

import numpy as np
import pytest
from helpers import run_db, signup
from ml4paleo_server import artifacts, labels
from ml4paleo_server.db import LabelOp, create_engine, create_sessionmaker
from sqlalchemy import func, select

from ml4paleo.labels import LABEL_CHUNK_ZYX, Source
from ml4paleo.labels.codec import decode_chunk
from ml4paleo.labels.deltas import split_into_deltas

SHAPE = (70, 130, 100)  # (z, y, x): chunks along each edge are partial


def make_project(browser, settings, database_url, name="Skull") -> str:
    project = browser.post("/api/projects", json={"name": name}).json()["id"]

    async def add_image(db):
        artifact = await artifacts.create_staging(
            db, project_id=uuid.UUID(project), kind="image", head_slot="image"
        )
        artifact.state = "committed"
        artifact.manifest = {"shape_czyx": [1, *SHAPE]}
        await artifacts.set_head(db, artifact)

    run_db(database_url, add_image)
    return project


def deltas_for(
    mask, origin, value=None, values=None, base_versions=None, only_if="any"
):
    out = []
    for delta in split_into_deltas(
        mask,
        origin,
        value=value,
        values=values,
        only_if=only_if,
        base_versions=base_versions,
    ):
        out.append(
            {
                "key": list(delta.key),
                "base_version": delta.base_version,
                "box": list(delta.box),
                "mask": base64.b64encode(delta.mask).decode(),
                "value": delta.value,
                "values": base64.b64encode(delta.values).decode()
                if delta.values
                else None,
                "only_if": delta.only_if,
            }
        )
    return out


def edit(browser, project, mask, origin, value, *, strict=False, op_id=None, **kw):
    return browser.post(
        f"/api/projects/{project}/labels/ops",
        json={
            "client_op_id": str(op_id or uuid.uuid4()),
            "deltas": deltas_for(mask, origin, value=value, **kw),
            "strict": strict,
            "tool": {"name": "brush"},
        },
    )


def chunk(browser, project, key, array="class") -> np.ndarray:
    response = browser.get(
        f"/api/projects/{project}/labels/zarr/{array}/c/{key[0]}/{key[1]}/{key[2]}"
    )
    if response.status_code == 404:
        return np.zeros(LABEL_CHUNK_ZYX, dtype=np.uint8)
    assert response.status_code == 200, response.text
    return decode_chunk(response.content)


@pytest.fixture
def ada(new_browser):
    browser = new_browser()
    signup(browser)
    return browser


@pytest.fixture
def project(ada, settings, migrated_database_url):
    project = make_project(ada, settings, migrated_database_url)
    for name, color in [("bone", "#ffffff"), ("matrix", "#884400")]:
        ada.post(
            f"/api/projects/{project}/labels/classes",
            json={"name": name, "color": color},
        )
    return project


def test_class_values_are_never_reused(ada, project):
    base = f"/api/projects/{project}/labels/classes"
    assert [c["value"] for c in ada.get(base).json()] == [2, 3]
    assert ada.request("DELETE", f"{base}/3").status_code == 204
    added = ada.post(base, json={"name": "tooth", "color": "#ffeeaa"}).json()
    assert added["value"] == 4
    assert (
        ada.patch(f"{base}/2", json={"name": "cortical bone"}).json()["name"]
        == "cortical bone"
    )
    assert ada.post(base, json={"name": "x", "color": "red"}).status_code == 422


def test_an_edit_shows_up_in_the_label_zarr(ada, project):
    mask = np.zeros((4, 70, 4), dtype=bool)
    mask[:, :, :] = True  # crosses from chunk y=0 into y=1
    response = edit(ada, project, mask, (10, 30, 20), 2)
    assert response.status_code == 201, response.text
    result = response.json()
    assert sorted(c["key"] for c in result["chunks"]) == [[0, 0, 0], [0, 1, 0]]
    assert all(c["version"] == 1 for c in result["chunks"])

    first = chunk(ada, project, (0, 0, 0))
    assert (first[10:14, 30:64, 20:24] == 2).all() and first.sum() == 2 * 4 * 34 * 4
    assert (chunk(ada, project, (0, 0, 0), "source")[10, 30, 20]) == Source.HUMAN
    url = f"/api/projects/{project}/labels/zarr/class/c/0/0/0"
    response = ada.get(url)
    assert response.headers["x-chunk-version"] == "1"
    revalidated = ada.get(url, headers={"If-None-Match": response.headers["etag"]})
    assert revalidated.status_code == 304
    # Chunks nobody labeled read as zeros, with their version.
    empty = ada.get(f"/api/projects/{project}/labels/zarr/class/c/1/1/1")
    assert (empty.status_code, empty.headers["x-chunk-version"]) == (404, "0")

    metadata = ada.get(f"/api/projects/{project}/labels/zarr/class/zarr.json").json()
    assert metadata["shape"] == list(SHAPE)
    assert metadata["chunk_grid"]["configuration"]["chunk_shape"] == [64, 64, 64]
    changes = ada.get(f"/api/projects/{project}/labels/changes?after=0").json()
    assert [c["op"]["seq"] for c in changes] == [result["seq"]]


def test_a_retried_edit_applies_once(ada, project, migrated_database_url):
    op_id = uuid.uuid4()
    mask = np.ones((2, 2, 2), dtype=bool)
    first = edit(ada, project, mask, (0, 0, 0), 2, op_id=op_id).json()
    again = edit(ada, project, mask, (0, 0, 0), 2, op_id=op_id).json()
    assert first == again

    async def count(db):
        return await db.scalar(select(func.count()).select_from(LabelOp))

    assert run_db(migrated_database_url, count) == 1


def test_a_strict_edit_refuses_chunks_that_changed(ada, project):
    mask = np.ones((3, 3, 3), dtype=bool)
    assert edit(ada, project, mask, (5, 5, 5), 2).status_code == 201
    # This client still thinks the chunk is at version 0.
    stale = edit(
        ada, project, mask, (8, 8, 8), 3, strict=True, base_versions={(0, 0, 0): 0}
    )
    assert stale.status_code == 409
    assert stale.json()["detail"]["chunks"] == [[0, 0, 0]]
    fresh = edit(
        ada, project, mask, (8, 8, 8), 3, strict=True, base_versions={(0, 0, 0): 1}
    )
    assert fresh.status_code == 201


def test_undo_and_redo_follow_the_overlay_of_live_edits(ada, project):
    full = np.ones((4, 4, 4), dtype=bool)
    left = np.zeros((4, 4, 4), dtype=bool)
    left[:, :, :2] = True
    a = edit(ada, project, full, (0, 0, 0), 2).json()["seq"]
    edit(ada, project, left, (0, 0, 0), 3)

    def region():
        return chunk(ada, project, (0, 0, 0))[:4, :4, :4]

    undo = ada.post(
        f"/api/projects/{project}/labels/ops/{a}/undo",
        json={"client_op_id": str(uuid.uuid4())},
    )
    assert undo.status_code == 201
    # The later edit keeps its half; the undone edit's other half is gone.
    assert (region()[:, :, :2] == 3).all() and (region()[:, :, 2:] == 0).all()
    again = ada.post(
        f"/api/projects/{project}/labels/ops/{a}/undo",
        json={"client_op_id": str(uuid.uuid4())},
    )
    assert again.status_code == 409
    redo = ada.post(
        f"/api/projects/{project}/labels/ops/{a}/redo",
        json={"client_op_id": str(uuid.uuid4())},
    )
    assert redo.status_code == 201
    # Redone underneath the later edit.
    assert (region()[:, :, :2] == 3).all() and (region()[:, :, 2:] == 2).all()
    missing = ada.post(
        f"/api/projects/{project}/labels/ops/9999/undo",
        json={"client_op_id": str(uuid.uuid4())},
    )
    assert missing.status_code == 404
    history = ada.get(f"/api/projects/{project}/labels/ops").json()
    assert [h["kind"] for h in history] == ["redo", "undo", "edit", "edit"]


def test_edits_must_use_classes_and_stay_inside(
    ada, project, new_browser, settings, migrated_database_url
):
    mask = np.ones((2, 2, 2), dtype=bool)
    assert edit(ada, project, mask, (0, 0, 0), 7).status_code == 422
    values = np.full((2, 2, 2), 2, dtype=np.uint8)
    values[0, 0, 0] = 9
    bad_values = ada.post(
        f"/api/projects/{project}/labels/ops",
        json={
            "client_op_id": str(uuid.uuid4()),
            "deltas": deltas_for(mask, (0, 0, 0), values=values),
        },
    )
    assert bad_values.status_code == 422
    # Erasing (0) and background (1) are always allowed.
    assert edit(ada, project, mask, (0, 0, 0), 1).status_code == 201
    assert edit(ada, project, mask, (0, 0, 0), 0).status_code == 201
    outside = edit(ada, project, mask, (69, 0, 0), 2)  # z 69..71 passes 70
    assert outside.status_code == 422
    empty = ada.post("/api/projects", json={"name": "Empty"}).json()["id"]
    assert edit(ada, empty, mask, (0, 0, 0), 0).status_code == 409


def test_concurrent_edits_of_one_chunk_both_land(
    ada, project, settings, migrated_database_url
):
    from ml4paleo.labels.deltas import split_into_deltas as split

    async def race():
        engine = create_engine(migrated_database_url)
        sessionmaker = create_sessionmaker(engine)

        async def stroke(value, x):
            mask = np.ones((2, 2, 2), dtype=bool)
            async with sessionmaker() as db:
                await labels.apply_edit(
                    db,
                    settings,
                    uuid.UUID(project),
                    client_op_id=uuid.uuid4(),
                    deltas=split(mask, (0, 0, x), value=value),
                )
                await asyncio.sleep(0.2)
                await db.commit()

        try:
            await asyncio.gather(stroke(2, 0), stroke(3, 10))
        finally:
            await engine.dispose()

    asyncio.run(race())
    data = chunk(ada, project, (0, 0, 0))
    assert (data[:2, :2, :2] == 2).all() and (data[:2, :2, 10:12] == 3).all()
    url = f"/api/projects/{project}/labels/zarr/class/c/0/0/0"
    assert ada.get(url).headers["x-chunk-version"] == "2"


def test_collaborators_follow_changes_live(ada, project, new_browser):
    bob = new_browser()
    signup(bob, username="bob")
    members = ada.post(f"/api/projects/{project}/members", json={"username": "bob"})
    bob_id = next(m["user_id"] for m in members.json() if m["username"] == "bob")
    watched = []

    def watch():
        url = f"/api/projects/{project}/labels/events?after=0"
        with bob.client.stream("GET", url) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    watched.append(json.loads(line[6:]))

    watcher = threading.Thread(target=watch)
    watcher.start()
    time.sleep(0.5)
    seq = edit(ada, project, np.ones((2, 2, 2), dtype=bool), (0, 0, 0), 2).json()["seq"]
    deadline = time.monotonic() + 10
    while not watched and time.monotonic() < deadline:
        time.sleep(0.05)
    # Removing bob from the project ends his stream.
    ada.request("DELETE", f"/api/projects/{project}/members/{bob_id}")
    watcher.join(timeout=10)
    assert not watcher.is_alive()
    assert watched and watched[0]["op"]["seq"] == seq
    assert watched[0]["chunks"][0]["key"] == [0, 0, 0]


def test_rois(ada, project, new_browser):
    base = f"/api/projects/{project}/rois"
    cube = ada.post(base, json={"bbox": [0, 0, 0, 32, 32, 32]})
    assert cube.status_code == 201 and cube.json()["status"] == "open"
    assert (
        ada.post(base, json={"bbox": [5, 0, 0, 6, 64, 64], "kind": "slice"}).status_code
        == 201
    )
    assert (
        ada.post(base, json={"bbox": [5, 0, 0, 7, 64, 64], "kind": "slice"}).status_code
        == 422
    )
    assert ada.post(base, json={"bbox": [0, 0, 0, 71, 10, 10]}).status_code == 422
    roi = cube.json()["id"]
    done = ada.patch(
        f"{base}/{roi}", json={"status": "complete", "split": "val"}
    ).json()
    assert (done["status"], done["split"]) == ("complete", "val")
    bob = new_browser()
    signup(bob, username="bob")
    assert bob.get(base).status_code == 404
    assert ada.request("DELETE", f"{base}/{roi}").status_code == 204
    assert len(ada.get(base).json()) == 1
