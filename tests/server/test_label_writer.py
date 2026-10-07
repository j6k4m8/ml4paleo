"""
The label writer and its API: classes, edits, strict edits, undo and redo,
the history and change feed, the labels as zarr, and ROIs.
"""

import asyncio
import base64
import datetime
import json
import random
import threading
import time
import uuid

import numpy as np
import pytest
from helpers import run_db, signup
from ml4paleo_server import artifacts, jobs, labels
from ml4paleo_server.db import (
    Artifact,
    LabelOp,
    TrainedModel,
    TrainingSet,
    create_engine,
    create_sessionmaker,
)
from ml4paleo_server.storage import project_storage
from sqlalchemy import func, select, update

from ml4paleo.labels import LABEL_CHUNK_ZYX, Source
from ml4paleo.labels.codec import decode_chunk
from ml4paleo.labels.deltas import split_into_deltas
from ml4paleo.segmentation.predict import create_prediction, open_prediction

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


def test_ops_commit_in_seq_order(project, settings, migrated_database_url):
    """
    An op on other chunks can't commit ahead of an earlier op still in
    flight, so a reader paging by seq never skips the earlier one.
    """
    pid = uuid.UUID(project)
    mask = np.ones((2, 2, 2), dtype=bool)

    async def scenario():
        engine = create_engine(migrated_database_url)
        sessionmaker = create_sessionmaker(engine)

        async def feed() -> list[int]:
            async with sessionmaker() as db:
                return [op.seq for op, _ in await labels.changes_since(db, pid, 0)]

        async def apply(db, x):
            return await labels.apply_edit(
                db,
                settings,
                pid,
                client_op_id=uuid.uuid4(),
                deltas=split_into_deltas(mask, (0, 0, x), value=2),
            )

        try:
            async with sessionmaker() as a, sessionmaker() as b:
                first = await apply(a, 0)
                second = asyncio.create_task(apply(b, 80))
                await asyncio.sleep(0.3)
                assert not second.done()
                assert await feed() == []
                await a.commit()
                later = await second
                await b.commit()
            assert first.seq < later.seq
            assert await feed() == [first.seq, later.seq]
        finally:
            await engine.dispose()

    asyncio.run(scenario())


def test_concurrent_retries_return_the_first_result(
    project, settings, migrated_database_url
):
    pid = uuid.UUID(project)
    edit_id, undo_id = uuid.uuid4(), uuid.uuid4()
    deltas = split_into_deltas(np.ones((2, 2, 2), dtype=bool), (0, 0, 0), value=2)

    async def race():
        engine = create_engine(migrated_database_url)
        sessionmaker = create_sessionmaker(engine)

        async def twice(call):
            async def attempt():
                async with sessionmaker() as db:
                    result = await call(db)
                    await asyncio.sleep(0.2)
                    await db.commit()
                    return result

            return await asyncio.gather(attempt(), attempt())

        try:
            edits = await twice(
                lambda db: labels.apply_edit(
                    db, settings, pid, client_op_id=edit_id, deltas=deltas
                )
            )
            undos = await twice(
                lambda db: labels.set_live(
                    db,
                    settings,
                    pid,
                    target_seq=edits[0].seq,
                    live=False,
                    client_op_id=undo_id,
                )
            )
            async with sessionmaker() as db:
                count = await db.scalar(
                    select(func.count())
                    .select_from(LabelOp)
                    .where(LabelOp.project_id == pid)
                )
            return edits, undos, count
        finally:
            await engine.dispose()

    edits, undos, count = asyncio.run(race())
    assert edits[0] == edits[1]
    assert undos[0] == undos[1]
    assert count == 2


def add_prediction(
    settings, database_url, project: str, head_slot=None, inputs=None
) -> str:
    """
    A committed prediction: bone in z 0..8, y 0..8, x 0..8, background
    elsewhere. With `head_slot`, it becomes that slot's head.
    """

    async def create(db):
        artifact = await artifacts.create_staging(
            db,
            project_id=uuid.UUID(project),
            kind="prediction",
            inputs=inputs or {"model_id": None},
            head_slot=head_slot,
        )
        group = create_prediction(
            project_storage(settings).child(artifacts.artifact_path(artifact)), SHAPE
        )
        classes = np.ones(SHAPE, dtype=np.uint8)
        classes[:8, :8, :8] = 2
        group["class"][:] = classes  # type: ignore[index]
        artifact.state = "committed"
        artifact.manifest = {"kind": "prediction", "shape_zyx": list(SHAPE)}
        if head_slot is not None:
            await artifacts.set_head(db, artifact)
        return str(artifact.id)

    return run_db(database_url, create)


def test_accepting_a_prediction_is_checked_against_it(
    ada, project, settings, migrated_database_url
):
    prediction = add_prediction(settings, migrated_database_url, project)
    ada.post(
        f"/api/projects/{project}/rois",
        json={"bbox": [0, 0, 0, 10, 10, 10], "kind": "cube"},
    )
    roi = ada.get(f"/api/projects/{project}/rois").json()[0]["id"]
    url = f"/api/projects/{project}/labels/accept"

    def accept(mask, origin, value, **kw):
        return ada.post(
            url,
            json={
                "client_op_id": str(uuid.uuid4()),
                "prediction_artifact_id": prediction,
                "roi_id": roi,
                "deltas": deltas_for(
                    mask, origin, value=value, only_if=kw.pop("only_if", "unlabeled")
                ),
                **kw,
            },
        )

    bone = np.ones((8, 8, 8), dtype=bool)
    # The prediction doesn't say tooth (3) there, or bone outside its cube.
    assert accept(bone, (0, 0, 0), 3).status_code == 422
    assert accept(np.ones((2, 2, 2), dtype=bool), (7, 7, 7), 2).status_code == 422
    # Accepted labels go only into unlabeled voxels, and stay in the ROI.
    assert accept(bone, (0, 0, 0), 2, only_if="any").status_code == 422
    assert accept(np.ones((1, 1, 1), dtype=bool), (10, 0, 0), 1).status_code == 422
    # Nobody can claim the result is anything but the server's call.
    assert accept(bone, (0, 0, 0), 2, tool={"name": "mine"}).status_code == 422

    accepted = accept(bone, (0, 0, 0), 2)
    assert accepted.status_code == 201, accepted.text
    source = chunk(ada, project, (0, 0, 0), array="source")
    assert (source[:8, :8, :8] == Source.MODEL_VERIFIED).all()
    [op] = ada.get(f"/api/projects/{project}/labels/ops?limit=1").json()
    assert op["source"] == Source.MODEL_VERIFIED
    assert op["tool"] == {
        "name": "accept-prediction",
        "prediction": prediction,
        "model": None,
        "roi": roi,
    }

    # Plain edits are always people's own.
    mask = np.ones((2, 2, 2), dtype=bool)
    claimed = ada.post(
        f"/api/projects/{project}/labels/ops",
        json={
            "client_op_id": str(uuid.uuid4()),
            "deltas": deltas_for(mask, (20, 0, 0), value=2),
            "source": "model_verified",
        },
    )
    assert claimed.status_code == 422


def test_predictions_labels_were_accepted_from_are_kept(
    ada, project, settings, migrated_database_url
):
    me = uuid.UUID(ada.get("/api/auth/session").json()["user"]["id"])
    ada.post(
        f"/api/projects/{project}/rois",
        json={"bbox": [0, 0, 0, 10, 10, 10], "kind": "cube"},
    )
    roi = ada.get(f"/api/projects/{project}/rois").json()[0]["id"]
    accepted, unused = [], []
    # A proposal, and a prediction of the whole image.
    for z, slot in enumerate((artifacts.proposal_slot(me), "prediction")):
        source = add_prediction(settings, migrated_database_url, project, slot)
        response = ada.post(
            f"/api/projects/{project}/labels/accept",
            json={
                "client_op_id": str(uuid.uuid4()),
                "prediction_artifact_id": source,
                "roi_id": roi,
                "deltas": deltas_for(
                    np.ones((1, 8, 8), dtype=bool),
                    (z, 0, 0),
                    value=2,
                    only_if="unlabeled",
                ),
            },
        )
        assert response.status_code == 201, response.text
        if slot == "prediction":
            # Undone labels still count: a redo brings them back.
            undone = ada.post(
                f"/api/projects/{project}/labels/ops/{response.json()['seq']}/undo",
                json={"client_op_id": str(uuid.uuid4())},
            )
            assert undone.status_code == 201
        # Newer ones replace it, and the first of those in turn.
        unused.append(add_prediction(settings, migrated_database_url, project, slot))
        add_prediction(settings, migrated_database_url, project, slot)
        accepted.append(source)

    # Past every grace period.
    no_wait = settings.model_copy(
        update={
            "storage": settings.storage.model_copy(update={"keep_superseded_days": 0})
        }
    )

    async def collect(db):
        long_ago = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)
        await db.execute(
            update(Artifact)
            .where(Artifact.state == "superseded")
            .values(state_changed_at=long_ago)
        )
        await db.commit()
        return await artifacts.collect_garbage(create_sessionmaker(db.bind), no_wait)

    assert run_db(migrated_database_url, collect) == len(unused)

    async def rows(db, ids):
        return [await db.get(Artifact, uuid.UUID(i)) for i in ids]

    for artifact in run_db(migrated_database_url, lambda db: rows(db, unused)):
        assert artifact.state == "deleted"
    # The ones labels came from stay, files and all.
    for artifact in run_db(migrated_database_url, lambda db: rows(db, accepted)):
        assert artifact.state == "superseded"
        group = open_prediction(
            project_storage(settings).child(artifacts.artifact_path(artifact))
        )
        assert (np.asarray(group["class"][:8, :8, :8]) == 2).all()  # type: ignore[index]


def test_collection_waits_for_an_accept_in_progress(
    project, settings, migrated_database_url
):
    # A replaced prediction, past every grace period...
    replaced = add_prediction(settings, migrated_database_url, project, "prediction")
    add_prediction(settings, migrated_database_url, project, "prediction")
    no_wait = settings.model_copy(
        update={
            "storage": settings.storage.model_copy(update={"keep_superseded_days": 0})
        }
    )

    async def race():
        engine = create_engine(migrated_database_url)
        sessionmaker = create_sessionmaker(engine)
        try:
            async with sessionmaker() as accepting:
                # ...that an accept has locked, as the API does, and is
                # recording labels from...
                await accepting.scalar(
                    select(Artifact.id)
                    .where(Artifact.id == uuid.UUID(replaced))
                    .with_for_update(read=True)
                )
                accepting.add(
                    LabelOp(
                        project_id=uuid.UUID(project),
                        client_op_id=uuid.uuid4(),
                        kind="edit",
                        source=int(Source.MODEL_VERIFIED),
                        tool={"name": "accept-prediction", "prediction": replaced},
                        bbox=[0, 0, 0, 1, 1, 1],
                    )
                )
                await accepting.flush()
                # ...while collection starts on it.
                collecting = asyncio.create_task(
                    artifacts.collect_garbage(sessionmaker, no_wait)
                )
                await asyncio.sleep(0.5)
                await accepting.commit()
            return await asyncio.wait_for(collecting, 10)
        finally:
            await engine.dispose()

    async def age(db):
        long_ago = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)
        await db.execute(update(Artifact).values(state_changed_at=long_ago))

    run_db(migrated_database_url, age)
    assert asyncio.run(race()) == 0

    async def state(db):
        return (await db.get(Artifact, uuid.UUID(replaced))).state

    assert run_db(migrated_database_url, state) == "superseded"


def add_model(database_url, project: str, name: str) -> str:
    """A trained model's row, for the history to name."""

    async def create(db):
        training_set = TrainingSet(
            id=uuid.uuid4().hex * 2, project_id=uuid.UUID(project), summary={}
        )
        db.add(training_set)
        await db.flush()
        model = TrainedModel(
            project_id=uuid.UUID(project),
            name=name,
            plugin="rf",
            params={},
            training_set_id=training_set.id,
            class_values=[2, 3],
        )
        db.add(model)
        await db.flush()
        return str(model.id)

    return run_db(database_url, create)


def accept_slice(browser, project, prediction, roi, z):
    """Accept the prediction's bone on slice `z` (8 × 8 voxels)."""
    return browser.post(
        f"/api/projects/{project}/labels/accept",
        json={
            "client_op_id": str(uuid.uuid4()),
            "prediction_artifact_id": prediction,
            "roi_id": roi,
            "deltas": deltas_for(
                np.ones((1, 8, 8), dtype=bool),
                (z, 0, 0),
                value=2,
                only_if="unlabeled",
            ),
        },
    )


def add_roi(browser, project) -> str:
    base = f"/api/projects/{project}/rois"
    return browser.post(
        base, json={"bbox": [0, 0, 0, 10, 10, 10], "kind": "cube"}
    ).json()["id"]


def test_the_history_names_people_and_models(
    ada, project, new_browser, settings, migrated_database_url
):
    base = f"/api/projects/{project}/labels"
    bob = new_browser()
    signup(bob, username="bob")
    members = ada.post(f"/api/projects/{project}/members", json={"username": "bob"})
    bob_id = next(m["user_id"] for m in members.json() if m["username"] == "bob")
    mask = np.ones((2, 2, 2), dtype=bool)
    stroke = edit(ada, project, mask, (0, 0, 0), 2).json()["seq"]
    edit(bob, project, mask, (0, 0, 4), 3)
    undone = bob.post(
        f"{base}/ops/{stroke}/undo", json={"client_op_id": str(uuid.uuid4())}
    )
    assert undone.status_code == 201
    # Labels accepted from a model's prediction, and from ada's proposal.
    model = add_model(migrated_database_url, project, "rf one")
    roi = add_roi(ada, project)
    me = uuid.UUID(ada.get("/api/auth/session").json()["user"]["id"])
    for z, slot in [(2, "prediction"), (3, artifacts.proposal_slot(me))]:
        prediction = add_prediction(
            settings, migrated_database_url, project, slot, {"model_id": model}
        )
        assert accept_slice(ada, project, prediction, roi, z).status_code == 201
    # People's own edits can't pass for accepted ones.
    forged = ada.post(
        f"{base}/ops",
        json={
            "client_op_id": str(uuid.uuid4()),
            "deltas": deltas_for(mask, (20, 0, 0), value=2),
            "tool": {"name": "accept-prediction", "model": model, "roi": roi},
        },
    )
    assert forged.status_code == 201

    history = ada.get(f"{base}/ops").json()
    assert [(h["kind"], h["username"]) for h in history] == [
        ("edit", "ada"),
        ("edit", "ada"),
        ("edit", "ada"),
        ("undo", "bob"),
        ("edit", "bob"),
        ("edit", "ada"),
    ]
    mine, proposed, predicted, undo, bobs, first = history
    assert mine["accepted"] is None and mine["source"] == Source.HUMAN
    assert predicted["accepted"] == {
        "kind": "prediction",
        "model_id": model,
        "model_name": "rf one",
        "v1_job_id": None,
        "roi_id": roi,
    }
    assert proposed["accepted"]["kind"] == "proposal"
    assert proposed["accepted"]["model_name"] == "rf one"
    # An undo comes with its edit, as that is now.
    assert undo["target_seq"] == first["seq"] == stroke
    assert undo["target"]["username"] == "ada"
    assert undo["target"]["live"] is False and undo["target"]["target"] is None
    assert first["live"] is False and first["target"] is None
    assert all(h["job_kind"] is None for h in history)

    def seqs(query: str) -> list[int]:
        response = ada.get(f"{base}/ops?{query}")
        assert response.status_code == 200, response.text
        return [h["seq"] for h in response.json()]

    assert seqs(f"user_id={bob_id}") == [undo["seq"], bobs["seq"]]
    assert seqs(f"user_id={bob_id}&before={undo['seq']}") == [bobs["seq"]]
    assert seqs(f"after={undo['seq']}&limit=2") == [mine["seq"], proposed["seq"]]
    assert seqs(f"after={first['seq']}&before={predicted['seq']}") == [
        undo["seq"],
        bobs["seq"],
    ]
    # An undo has its edit's source.
    assert seqs(f"user_id={bob_id}&source=1&limit=1") == [undo["seq"]]
    assert seqs(f"source={int(Source.MODEL_VERIFIED)}") == [
        proposed["seq"],
        predicted["seq"],
    ]
    assert seqs(f"source={int(Source.PROPAGATED)}") == []
    assert ada.get(f"{base}/ops?source=9").status_code == 422
    assert ada.get(f"{base}/ops?user_id=bob").status_code == 422
    # Only members can read it.
    eve = new_browser()
    signup(eve, username="eve")
    assert eve.get(f"{base}/ops?user_id={bob_id}").status_code == 404


def test_the_history_names_the_job_that_imported_labels(
    ada, project, settings, migrated_database_url
):
    pid = uuid.UUID(project)

    async def import_sample(db):
        job = await jobs.enqueue(db, "v1.labels", {}, project_id=pid)
        result = await labels.apply_edit(
            db,
            settings,
            pid,
            client_op_id=uuid.uuid4(),
            deltas=split_into_deltas(
                np.ones((1, 4, 4), dtype=bool), (5, 0, 0), value=2
            ),
            tool={"name": "v1-import", "job": "ab12cd", "sample": "1700000000"},
            job_id=job.id,
        )
        return result.seq

    imported_seq = run_db(migrated_database_url, import_sample)
    # Accepted from the prediction v1 made, which no model here did.
    roi = add_roi(ada, project)
    prediction = add_prediction(
        settings,
        migrated_database_url,
        project,
        "prediction",
        {"model_id": None, "v1_job_id": "ab12cd"},
    )
    assert accept_slice(ada, project, prediction, roi, 0).status_code == 201

    accepted, imported = ada.get(f"/api/projects/{project}/labels/ops").json()
    assert imported["seq"] == imported_seq
    assert (imported["username"], imported["job_kind"]) == (None, "v1.labels")
    assert imported["source"] == Source.HUMAN and imported["accepted"] is None
    assert accepted["accepted"] == {
        "kind": "prediction",
        "model_id": None,
        "model_name": None,
        "v1_job_id": "ab12cd",
        "roi_id": roi,
    }


def test_out_of_range_numbers_are_refused(ada, project):
    base = f"/api/projects/{project}/labels"
    huge = "9" * 20
    assert ada.get(f"{base}/ops?limit=-1").status_code == 422
    assert ada.get(f"{base}/ops?before={huge}").status_code == 422
    assert ada.get(f"{base}/ops?after={huge}").status_code == 422
    assert ada.get(f"{base}/ops?after=-1").status_code == 422
    assert ada.get(f"{base}/changes?after={huge}").status_code == 422
    toggle = {"client_op_id": str(uuid.uuid4())}
    assert ada.post(f"{base}/ops/{huge}/undo", json=toggle).status_code == 422
    assert ada.get(f"{base}/zarr/class/c/{huge}/0/0").status_code == 404


def test_collaborators_follow_changes_live(ada, project, new_browser):
    bob = new_browser()
    signup(bob, username="bob")
    members = ada.post(f"/api/projects/{project}/members", json={"username": "bob"})
    bob_id = next(m["user_id"] for m in members.json() if m["username"] == "bob")
    watched = []
    mask = np.ones((2, 2, 2), dtype=bool)
    seen = edit(ada, project, mask, (0, 0, 0), 3).json()["seq"]

    def watch():
        url = f"/api/projects/{project}/labels/events?after=0"
        # As a reconnecting browser sends it: resume after that op.
        headers = {"Last-Event-ID": str(seen)}
        with bob.client.stream("GET", url, headers=headers) as response:
            for line in response.iter_lines():
                if line.startswith("data: "):
                    watched.append(json.loads(line[6:]))

    watcher = threading.Thread(target=watch)
    watcher.start()
    time.sleep(0.5)
    seq = edit(ada, project, mask, (0, 0, 0), 2).json()["seq"]
    deadline = time.monotonic() + 10
    while not watched and time.monotonic() < deadline:
        time.sleep(0.05)
    # Removing bob from the project ends his stream.
    ada.request("DELETE", f"/api/projects/{project}/members/{bob_id}")
    watcher.join(timeout=10)
    assert not watcher.is_alive()
    assert [change["op"]["seq"] for change in watched] == [seq]
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


def test_exploring_makes_an_roi_where_none_is(ada, project):
    base = f"/api/projects/{project}/rois"
    made = ada.post(f"{base}/explore")
    assert made.status_code == 201, made.text
    roi = made.json()
    assert (roi["kind"], roi["status"], roi["origin"]) == ("cube", "open", "explore")
    # A cube 128 voxels a side, cut to the image and inside it.
    z0, y0, x0, z1, y1, x1 = roi["bbox"]
    assert (z1 - z0, y1 - y0, x1 - x0) == (70, 128, 100)
    assert (z0, x0) == (0, 0) and 0 <= y0 <= 2
    # Every place overlaps it, even smaller ones, until it's gone.
    full = ada.post(f"{base}/explore")
    assert full.status_code == 409
    assert (
        full.json()["detail"] == "There's no room left for an ROI clear of the others."
    )
    assert ada.request("DELETE", f"{base}/{roi['id']}").status_code == 204
    assert ada.post(f"{base}/explore").status_code == 201


def test_explored_places_are_cubes_and_miss_rois():
    from ml4paleo_server.api.rois import EXPLORE_SIDE, explore_box

    rng = random.Random(0)
    assert explore_box((500, 500, 500), [], rng=rng)[3:] != [0, 0, 0]
    # Voxels twice as deep as they are wide: half as many along z.
    box = explore_box((500, 500, 500), [], (2.0, 1.0, 1.0), rng=rng)
    assert [box[a + 3] - box[a] for a in range(3)] == [64, 128, 128]
    taken = [[0, 0, 0, 500, 500, 250]]
    for _ in range(20):
        box = explore_box((500, 500, 500), taken, rng=rng)
        assert box[2] >= 250 and box[5] - box[2] == EXPLORE_SIDE
    # Where ROIs leave no room for that, a smaller cube.
    box = explore_box((100, 200, 200), [[0, 0, 0, 100, 200, 100]], rng=rng)
    assert [box[a + 3] - box[a] for a in range(3)] == [64, 64, 64]
    assert explore_box((10, 10, 10), [[0, 0, 0, 1, 1, 1]], rng=rng) is None
