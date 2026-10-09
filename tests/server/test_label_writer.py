"""
The label writer and its API: classes, edits, strict edits, undo and redo,
the history and change feed, the labels as zarr, and ROIs.
"""

import asyncio
import base64
import datetime
import itertools
import json
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx2
import numpy as np
import pytest
from helpers import NOT_JSON, run_db, signup, strict_json
from ml4paleo_server import artifacts, jobs, labels
from ml4paleo_server.api import labels as api_labels
from ml4paleo_server.db import (
    Artifact,
    LabelOp,
    TrainedModel,
    TrainingSet,
    create_engine,
    create_sessionmaker,
)
from ml4paleo_server.storage import project_storage
from sqlalchemy import func, select, text, update

from ml4paleo.labels import LABEL_CHUNK_ZYX, Source
from ml4paleo.labels.codec import decode_chunk
from ml4paleo.labels.deltas import split_into_deltas
from ml4paleo.segmentation.predict import create_prediction, open_prediction

SHAPE = (70, 130, 100)  # (z, y, x): chunks along each edge are partial


def add_image(database_url, project: str, shape=SHAPE) -> str:
    """A committed image of `shape` as the project's image (replacing any)."""

    async def create(db):
        artifact = await artifacts.create_staging(
            db, project_id=uuid.UUID(project), kind="image", head_slot="image"
        )
        artifact.state = "committed"
        artifact.manifest = {"shape_czyx": [1, *shape]}
        await artifacts.set_head(db, artifact)
        return str(artifact.id)

    return run_db(database_url, create)


def make_project(browser, settings, database_url, name="Skull", shape=SHAPE) -> str:
    project = browser.post("/api/projects", json={"name": name}).json()["id"]
    add_image(database_url, project, shape)
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


def test_counts_say_what_has_been_labeled(ada, project):
    base = f"/api/projects/{project}/labels"
    assert ada.get(f"{base}/counts").json() == {"1": 0, "2": 0, "3": 0}
    mask = np.ones((2, 3, 4), dtype=bool)
    assert edit(ada, project, mask, (0, 0, 0), 2).status_code == 201
    assert edit(ada, project, mask, (5, 5, 5), 1).status_code == 201
    # Across chunks too, and a later edit over earlier labels counts once.
    assert edit(ada, project, mask, (5, 66, 5), 1).status_code == 201
    assert edit(ada, project, mask, (5, 5, 5), 3).status_code == 201
    assert ada.get(f"{base}/counts").json() == {"1": 24, "2": 24, "3": 24}
    # A deleted class isn't listed.
    assert ada.request("DELETE", f"{base}/classes/2").status_code == 204
    assert ada.get(f"{base}/counts").json() == {"1": 24, "3": 24}


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
    settings, database_url, project: str, head_slot=None, inputs=None, shape=SHAPE
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
            project_storage(settings).child(artifacts.artifact_path(artifact)), shape
        )
        classes = np.ones(shape, dtype=np.uint8)
        classes[:8, :8, :8] = 2
        group["class"][:] = classes  # type: ignore[index]
        artifact.state = "committed"
        artifact.manifest = {"kind": "prediction", "shape_zyx": list(shape)}
        if head_slot is not None:
            await artifacts.set_head(db, artifact)
        return str(artifact.id)

    return run_db(database_url, create)


def add_sparse_prediction(settings, database_url, project: str, shape, fills) -> str:
    """
    A committed prediction of `shape` that holds each value in the box (z0, y0,
    x0, z1, y1, x1) given for it and 0 elsewhere, storing only the chunks those
    boxes reach (so it can be as big as an image is).
    """

    async def create(db):
        artifact = await artifacts.create_staging(
            db,
            project_id=uuid.UUID(project),
            kind="prediction",
            inputs={"model_id": None},
            head_slot=None,
        )
        group = create_prediction(
            project_storage(settings).child(artifacts.artifact_path(artifact)), shape
        )
        for box, value in fills:
            region = tuple(slice(box[a], box[a + 3]) for a in range(3))
            group["class"][region] = value  # type: ignore[index]
        artifact.state = "committed"
        artifact.manifest = {"kind": "prediction", "shape_zyx": list(shape)}
        return str(artifact.id)

    return run_db(database_url, create)


class Reads:
    """
    What accepts read of a prediction's classes, noted as they go: the regions
    asked for, and the (64 voxel) chunks of the prediction those reach into.
    With a `latency` each read takes that long, as one from a store a way off
    does, and the reads under way at once (in all, and for each `owner` of a
    region, if a function of the region names one), the threads that read, and
    the time they took are noted too. With `meet`, no read goes on until that
    many are under way at once, so a number of accepts' reads all overlap.
    With `gives`, a function of the region, that is what a read gives, instead of
    what the prediction holds.
    """

    def __init__(
        self, monkeypatch, latency: float = 0, owner=None, meet: int = 0, gives=None
    ):
        self.regions: list[tuple[slice, ...]] = []
        self.latency = latency
        self.owner = owner
        self.meet = meet
        self.gives = gives
        self.met = False
        self.under_way = 0
        self.most_at_once = 0
        self.under_way_of: dict = {}
        self.most_at_once_of: dict = {}
        self.threads: set[int] = set()
        self.began: float | None = None
        self.ended: float | None = None
        self.lock = threading.Condition()
        real = api_labels.open_prediction

        def open_noting(grant):
            group = real(grant)
            return {"class": NotingArray(group["class"], self)}

        monkeypatch.setattr(api_labels, "open_prediction", open_noting)

    @property
    def chunks(self) -> set[tuple[int, ...]]:
        reached = set()
        for region in self.regions:
            reached.update(
                itertools.product(
                    *(range(r.start // 64, (r.stop - 1) // 64 + 1) for r in region)
                )
            )
        return reached

    @property
    def took(self) -> float:
        """From the first read beginning to the last one ending."""
        assert self.began is not None and self.ended is not None
        return self.ended - self.began


class NotingArray:
    """A zarr array that notes the regions read from it, which may be from several threads."""

    def __init__(self, array, reads: Reads):
        self._array = array
        self._reads = reads

    def __getattr__(self, name):
        return getattr(self._array, name)

    def __getitem__(self, region):
        reads = self._reads
        owner = reads.owner(region) if reads.owner else None
        with reads.lock:
            reads.regions.append(region)
            reads.threads.add(threading.get_ident())
            reads.under_way += 1
            reads.most_at_once = max(reads.most_at_once, reads.under_way)
            if reads.owner:
                now = reads.under_way_of[owner] = reads.under_way_of.get(owner, 0) + 1
                reads.most_at_once_of[owner] = max(
                    reads.most_at_once_of.get(owner, 0), now
                )
            if reads.began is None:
                reads.began = time.monotonic()
            if reads.meet:
                reads.met = reads.met or reads.under_way >= reads.meet
                reads.lock.notify_all()
                # Never for long: if they never are, the test says so after.
                reads.lock.wait_for(lambda: reads.met, timeout=10)
        try:
            if reads.latency:
                time.sleep(reads.latency)
            return reads.gives(region) if reads.gives else self._array[region]
        finally:
            with reads.lock:
                reads.under_way -= 1
                if reads.owner:
                    reads.under_way_of[owner] -= 1
                reads.ended = time.monotonic()


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


def replaced_and_aged(settings, database_url, project: str):
    """
    A prediction that another has replaced, past every grace period, and the
    settings to collect it with.
    """
    replaced = add_prediction(settings, database_url, project, "prediction")
    add_prediction(settings, database_url, project, "prediction")
    no_wait = settings.model_copy(
        update={
            "storage": settings.storage.model_copy(update={"keep_superseded_days": 0})
        }
    )

    async def age(db):
        long_ago = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)
        await db.execute(update(Artifact).values(state_changed_at=long_ago))

    run_db(database_url, age)
    return replaced, no_wait


async def prediction_state(db, prediction: str) -> str:
    artifact = await db.get(Artifact, uuid.UUID(prediction))
    return artifact.state


def test_collection_waits_for_an_accept_in_progress(
    project, settings, migrated_database_url
):
    replaced, no_wait = replaced_and_aged(settings, migrated_database_url, project)

    async def race():
        engine = create_engine(migrated_database_url)
        sessionmaker = create_sessionmaker(engine)
        try:
            async with sessionmaker() as accepting:
                # An accept has locked it, as the API does, and is
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

    assert asyncio.run(race()) == 0
    assert run_db(migrated_database_url, lambda db: prediction_state(db, replaced)) == (
        "superseded"
    )


def test_the_accept_endpoint_holds_collection_back_while_it_runs(
    ada, project, settings, migrated_database_url, monkeypatch
):
    # The test above takes the accept's lock itself, so it holds whether or
    # not the endpoint does. Here a real accept is held part way, reading the
    # prediction, while collection starts on that prediction.
    replaced, no_wait = replaced_and_aged(settings, migrated_database_url, project)
    reading, release = threading.Event(), threading.Event()
    real = api_labels.open_prediction

    def hold(grant):
        reading.set()
        assert release.wait(30), "the accept was never let go"
        return real(grant)

    monkeypatch.setattr(api_labels, "open_prediction", hold)
    accepted = {}
    accepting = threading.Thread(
        target=lambda: accepted.update(
            response=accept_box(
                ada,
                project,
                replaced,
                [0, 0, 0, 10, 10, 10],
                np.ones((8, 8, 8), dtype=bool),
                (0, 0, 0),
                2,
            )
        )
    )

    async def collect():
        engine = create_engine(migrated_database_url)
        try:
            return await artifacts.collect_garbage(create_sessionmaker(engine), no_wait)
        finally:
            await engine.dispose()

    collected = {}
    collecting = threading.Thread(
        target=lambda: collected.update(count=asyncio.run(collect()))
    )

    async def waiting_for_a_lock(db) -> int:
        return await db.scalar(
            text(
                "select count(*) from pg_stat_activity "
                "where datname = current_database() and wait_event_type = 'Lock'"
            )
        )

    accepting.start()
    try:
        assert reading.wait(10), "the accept never got to reading the prediction"
        collecting.start()
        # Collection must be left waiting on the accept (not collect, nor finish).
        deadline = time.monotonic() + 10
        while collecting.is_alive() and not run_db(
            migrated_database_url, waiting_for_a_lock
        ):
            assert time.monotonic() < deadline, "collection neither waited nor ended"
            time.sleep(0.05)
        assert collecting.is_alive(), "collection didn't wait for the accept"
    finally:
        release.set()
    accepting.join(30)
    collecting.join(30)
    assert accepted["response"].status_code == 201, accepted["response"].text
    # The accept's op is there for it to find, so it keeps the prediction.
    assert collected["count"] == 0
    assert run_db(migrated_database_url, lambda db: prediction_state(db, replaced)) == (
        "superseded"
    )


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


def accept_box(browser, project, prediction, box, mask, origin, value, **kw):
    """Accept `value` over `mask` at `origin`, into `box` (z0, y0, x0, z1, y1, x1)."""
    return browser.post(
        f"/api/projects/{project}/labels/accept",
        json={
            "client_op_id": str(uuid.uuid4()),
            "prediction_artifact_id": prediction,
            "box": box,
            "deltas": deltas_for(
                mask, origin, value=value, only_if=kw.pop("only_if", "unlabeled")
            ),
            **kw,
        },
    )


def current_image(database_url, project: str) -> str:
    async def find(db):
        image = await artifacts.head(db, uuid.UUID(project), "image")
        assert image is not None
        return str(image.id)

    return run_db(database_url, find)


def toggle(browser, project, seq, action):
    return browser.post(
        f"/api/projects/{project}/labels/ops/{seq}/{action}",
        json={"client_op_id": str(uuid.uuid4())},
    )


def test_accepting_a_prediction_into_a_box_is_checked_against_it(
    ada, project, settings, migrated_database_url
):
    prediction = add_prediction(settings, migrated_database_url, project)
    box = [0, 0, 0, 10, 10, 10]
    bone = np.ones((8, 8, 8), dtype=bool)

    def accept(mask, origin, value, where=box, **kw):
        return accept_box(ada, project, prediction, where, mask, origin, value, **kw)

    def refused(response, why):
        assert response.status_code == 422, response.text
        assert response.json()["detail"] == why

    # The prediction doesn't say tooth (3) there, or bone outside its cube.
    differs = "Those labels don't match the prediction"
    refused(accept(bone, (0, 0, 0), 3), differs)
    refused(accept(np.ones((2, 2, 2), dtype=bool), (7, 7, 7), 2), differs)
    # Accepted labels go only into unlabeled voxels, and stay in the box.
    refused(
        accept(bone, (0, 0, 0), 2, only_if="any"),
        "An accepted prediction writes one predicted value per chunk, "
        "only into unlabeled voxels",
    )
    outside = "Those labels reach outside the box"
    refused(accept(np.ones((1, 1, 1), dtype=bool), (10, 0, 0), 1), outside)
    refused(accept(bone, (0, 0, 0), 2, where=[0, 0, 0, 8, 8, 7]), outside)
    # Nobody can claim the result is anything but the server's call.
    assert accept(bone, (0, 0, 0), 2, tool={"name": "mine"}).status_code == 422
    # A prediction that isn't the project's.
    nowhere = accept_box(ada, project, str(uuid.uuid4()), box, bone, (0, 0, 0), 2)
    assert nowhere.status_code == 404
    assert nowhere.json()["detail"] == "No such prediction."
    assert not chunk(ada, project, (0, 0, 0)).any()

    # A box is whole voxels, has some, and is inside the image (70 × 130 × 100).
    one = np.ones((1, 1, 1), dtype=bool)
    for off in (
        [0, 0, 0, 71, 10, 10],
        [0, 0, 0, 10, 131, 10],
        [0, 0, 0, 10, 10, 101],
        [-1, 0, 0, 10, 10, 10],
        [5, 0, 0, 5, 10, 10],
        [9, 0, 0, 3, 10, 10],
    ):
        refused(
            accept(one, (0, 0, 0), 2, where=off), "The box must be inside the image."
        )
    for malformed in ([0, 0, 0, 10.5, 10, 10], [0, 0, 0, 10, 10], "0,0,0,10,10,10"):
        response = accept(one, (0, 0, 0), 2, where=malformed)
        assert response.status_code == 422
        assert "box" in str(response.json()["detail"])
    assert accept(one, (0, 0, 0), 2, where=[0, 0, 0, 70, 130, 100]).status_code == 201

    # Exactly one of an ROI and a box says where.
    roi = add_roi(ada, project)
    place = {
        "client_op_id": str(uuid.uuid4()),
        "prediction_artifact_id": prediction,
        "deltas": deltas_for(one, (1, 0, 0), value=2, only_if="unlabeled"),
    }
    url = f"/api/projects/{project}/labels/accept"
    for both_or_neither in (
        {},
        {"roi_id": None, "box": None},
        {"roi_id": roi, "box": box},
    ):
        response = ada.post(url, json={**place, **both_or_neither})
        assert response.status_code == 422
        assert "exactly one of roi_id and box" in str(response.json()["detail"])
    # A null for the one that isn't used is the same as leaving it out.
    assert ada.post(url, json={**place, "roi_id": None, "box": box}).status_code == 201

    accepted = accept(bone, (0, 0, 0), 2)
    assert accepted.status_code == 201, accepted.text
    classes = chunk(ada, project, (0, 0, 0))
    source = chunk(ada, project, (0, 0, 0), array="source")
    assert (classes[:8, :8, :8] == 2).all()
    assert (source[:8, :8, :8] == Source.MODEL_VERIFIED).all()
    [op] = ada.get(f"/api/projects/{project}/labels/ops?limit=1").json()
    assert op["source"] == Source.MODEL_VERIFIED
    assert op["bbox"] == [0, 0, 0, 8, 8, 8]
    # The tool record names the prediction and the box, and no ROI.
    assert op["tool"] == {
        "name": "accept-prediction",
        "prediction": prediction,
        "model": None,
        "box": box,
    }


def test_a_box_accept_fills_only_unlabeled_voxels_and_undoes_and_redoes(
    ada, project, settings, migrated_database_url
):
    prediction = add_prediction(settings, migrated_database_url, project)
    base = f"/api/projects/{project}/labels"
    # Someone labeled a corner of the bone by hand, as matrix (3).
    mine = np.zeros((8, 8, 8), dtype=bool)
    mine[:, :4, :4] = True
    hand = edit(ada, project, mine[:, :4, :4], (0, 0, 0), 3)
    assert hand.status_code == 201

    box = [0, 0, 0, 8, 8, 8]
    accepted = accept_box(
        ada, project, prediction, box, np.ones((8, 8, 8), dtype=bool), (0, 0, 0), 2
    )
    assert accepted.status_code == 201, accepted.text

    def region():
        classes = chunk(ada, project, (0, 0, 0))
        source = chunk(ada, project, (0, 0, 0), array="source")
        return classes[:8, :8, :8], source[:8, :8, :8], classes

    def filled():
        classes, source, whole = region()
        # The hand labels stay, the rest of the box is the model's, and
        # nothing outside the box was touched.
        assert (classes[mine] == 3).all() and (source[mine] == Source.HUMAN).all()
        assert (classes[~mine] == 2).all()
        assert (source[~mine] == Source.MODEL_VERIFIED).all()
        assert whole.sum() == classes.sum()

    def left_to_hand():
        classes, source, whole = region()
        assert (classes[mine] == 3).all() and (classes[~mine] == 0).all()
        assert (source[~mine] == Source.NONE).all()
        assert whole.sum() == classes.sum()

    filled()
    seq = accepted.json()["seq"]
    assert toggle(ada, project, seq, "undo").status_code == 201
    left_to_hand()
    assert toggle(ada, project, seq, "redo").status_code == 201
    filled()
    assert ada.get(f"{base}/counts").json() == {"1": 0, "2": 512 - 128, "3": 128}


def test_a_box_accept_across_chunks_takes_a_predicted_value_at_a_time(
    ada, project, settings, migrated_database_url
):
    prediction = add_prediction(settings, migrated_database_url, project)
    # Chunks are 64 wide, so this box crosses their edges along every axis.
    box = [0, 0, 0, 70, 70, 70]
    place = np.ones((70, 70, 70), dtype=bool)
    bone = np.zeros_like(place)
    bone[:8, :8, :8] = True
    # What the annotator sends: an op for each value the prediction has.
    first = accept_box(ada, project, prediction, box, bone, (0, 0, 0), 2)
    second = accept_box(ada, project, prediction, box, place & ~bone, (0, 0, 0), 1)
    assert (first.status_code, second.status_code) == (201, 201), second.text
    counts = ada.get(f"/api/projects/{project}/labels/counts").json()
    assert counts == {"1": 70**3 - 512, "2": 512, "3": 0}
    history = ada.get(f"/api/projects/{project}/labels/ops").json()
    assert [h["tool"]["box"] for h in history] == [box, box]
    # The first touched one chunk, the second all eight.
    assert [len(r.json()["chunks"]) for r in (second, first)] == [8, 1]


def test_a_box_to_accept_at_once_holds_at_most_256_cubed(
    ada, settings, migrated_database_url, monkeypatch
):
    reads = Reads(monkeypatch)
    big = make_project(ada, settings, migrated_database_url, "Big", (300, 300, 300))
    prediction = add_prediction(
        settings, migrated_database_url, big, shape=(300, 300, 300)
    )
    one = np.ones((1, 1, 1), dtype=bool)
    ada.post(
        f"/api/projects/{big}/labels/classes", json={"name": "bone", "color": "#ffffff"}
    )

    def accept(box):
        return accept_box(ada, big, prediction, box, one, (0, 0, 0), 2)

    too_much = "That's too much to accept at once; use a smaller box"
    # One voxel more than 256 cubed, however it's shaped.
    for over in (
        [0, 0, 0, 257, 257, 257],
        [0, 0, 0, 256, 256, 257],
        [0, 0, 0, 300, 300, 300],
    ):
        refused = accept(over)
        assert refused.status_code == 422, over
        assert refused.json()["detail"] == too_much
    assert not reads.regions
    # Only the chunk the one label is in is read, not the box's 64 of them.
    assert accept([0, 0, 0, 256, 256, 256]).status_code == 201
    assert reads.chunks == {(0, 0, 0)}
    assert accept([0, 0, 0, 1, 300, 300]).status_code == 201


@pytest.mark.parametrize("into", ["box", "roi"])
def test_an_accept_reads_only_the_chunks_its_labels_are_in(
    ada, settings, migrated_database_url, monkeypatch, into
):
    # A slab of the most voxels a box may hold, which has 4,096 chunks of the
    # prediction in it, with a label in each of two corner ones.
    shape = (64, 4096, 4096)
    slab = [0, 0, 0, 1, 4096, 4096]
    big = make_project(ada, settings, migrated_database_url, "Slab", shape)
    prediction = add_sparse_prediction(
        settings,
        migrated_database_url,
        big,
        shape,
        [([0, 0, 0, 1, 64, 64], 1), ([0, 4032, 4032, 1, 4096, 4096], 1)],
    )
    one = np.ones((1, 1, 1), dtype=bool)
    deltas = deltas_for(one, (0, 0, 0), value=1, only_if="unlabeled")
    deltas += deltas_for(one, (0, 4095, 4095), value=1, only_if="unlabeled")
    if into == "roi":
        roi = ada.post(
            f"/api/projects/{big}/rois", json={"bbox": slab, "kind": "slice"}
        )
        where = {"roi_id": roi.json()["id"]}
    else:
        where = {"box": slab}
    reads = Reads(monkeypatch)
    response = ada.post(
        f"/api/projects/{big}/labels/accept",
        json={
            "client_op_id": str(uuid.uuid4()),
            "prediction_artifact_id": prediction,
            "deltas": deltas,
            **where,
        },
    )
    assert response.status_code == 201, response.text
    assert reads.chunks == {(0, 0, 0), (0, 63, 63)}
    assert chunk(ada, big, (0, 0, 0))[0, 0, 0] == 1
    assert chunk(ada, big, (0, 63, 63))[0, 63, 63] == 1


def test_an_accept_of_as_many_labels_as_a_request_holds_reads_just_their_chunks(
    ada, settings, migrated_database_url, monkeypatch
):
    # 512 labels, the most one request holds, that start and end part way along
    # a row of the slab's chunks: the box around them has 576 chunks in it.
    shape = (64, 4096, 4096)
    box = [0, 0, 0, 1, 576, 4096]
    big = make_project(ada, settings, migrated_database_url, "Slab", shape)
    prediction = add_sparse_prediction(
        settings, migrated_database_url, big, shape, [(box, 1)]
    )
    labels_in = [
        ((1, 64, 2048), (0, 0, 2048)),
        ((1, 448, 4096), (0, 64, 0)),
        ((1, 64, 2048), (0, 512, 0)),
    ]
    chunks_of = (
        {(0, 0, x) for x in range(32, 64)}
        | {(0, y, x) for y in range(1, 8) for x in range(64)}
        | {(0, 8, x) for x in range(32)}
    )
    assert len(chunks_of) == api_labels.MAX_DELTAS

    def accept(parts):
        deltas = []
        for size, origin in parts:
            mask = np.ones(size, dtype=bool)
            deltas += deltas_for(mask, origin, value=1, only_if="unlabeled")
        return ada.post(
            f"/api/projects/{big}/labels/accept",
            json={
                "client_op_id": str(uuid.uuid4()),
                "prediction_artifact_id": prediction,
                "box": box,
                "deltas": deltas,
            },
        )

    reads = Reads(monkeypatch)
    # One more is more than a request may hold, and nothing is read for it.
    refused = accept([*labels_in, ((1, 64, 64), (0, 512, 2048))])
    assert refused.status_code == 422, refused.text
    assert "at most 512 items" in str(refused.json()["detail"])
    assert not reads.regions
    accepted = accept(labels_in)
    assert accepted.status_code == 201, accepted.text
    assert len(accepted.json()["chunks"]) == api_labels.MAX_DELTAS
    assert reads.chunks == chunks_of


SLAB = (64, 4096, 4096)


def accept_slab(ada, project, prediction, box, deltas):
    return ada.post(
        f"/api/projects/{project}/labels/accept",
        json={
            "client_op_id": str(uuid.uuid4()),
            "prediction_artifact_id": prediction,
            "box": box,
            "deltas": deltas,
        },
    )


def test_an_accept_reads_its_chunks_at_once_not_one_after_another(
    ada, settings, migrated_database_url, monkeypatch
):
    # 512 labels in 512 chunks, the most a request holds, with each read taking as
    # long as one from a store a way off does.
    rows = [0, 0, 0, 1, 512, 4096]
    big = make_project(ada, settings, migrated_database_url, "Slab", SLAB)
    prediction = add_sparse_prediction(
        settings, migrated_database_url, big, SLAB, [(rows, 1)]
    )
    mask = np.ones((1, 512, 4096), dtype=bool)
    deltas = deltas_for(mask, (0, 0, 0), value=1, only_if="unlabeled")
    assert len(deltas) == api_labels.MAX_DELTAS
    latency = 0.03
    readers = api_labels.ACCEPT_READS_AT_ONCE
    # No read goes on until `readers` of them are under way at once, so that
    # they are is certain, not a matter of how quickly threads start.
    reads = Reads(monkeypatch, latency, meet=readers)
    accepted = accept_slab(ada, big, prediction, rows, deltas)
    assert accepted.status_code == 201, accepted.text
    assert len(reads.regions) == len(reads.chunks) == api_labels.MAX_DELTAS
    # All the accept's readers are busy at once, and no more than they...
    assert reads.most_at_once == readers
    # ...so it is not all the reads in turn (512 × 30 ms), as far from it as
    # this is: a bound for a loaded machine, not a measure.
    assert reads.took < api_labels.MAX_DELTAS * latency / 2


def test_the_deltas_of_a_chunk_share_a_read(
    ada, project, settings, migrated_database_url, monkeypatch
):
    prediction = add_prediction(settings, migrated_database_url, project)
    reads = Reads(monkeypatch)
    # Two values in one chunk, each as the prediction has it (bone in the corner,
    # background elsewhere). A request may hold one delta for a chunk, as it is
    # refused when applied, but the prediction is read once for both.
    one = np.ones((2, 2, 2), dtype=bool)
    deltas = deltas_for(one, (0, 0, 0), value=2, only_if="unlabeled")
    deltas += deltas_for(one, (20, 20, 20), value=1, only_if="unlabeled")
    assert deltas[0]["key"] == deltas[1]["key"]
    refused = ada.post(
        f"/api/projects/{project}/labels/accept",
        json={
            "client_op_id": str(uuid.uuid4()),
            "prediction_artifact_id": prediction,
            "box": [0, 0, 0, 30, 30, 30],
            "deltas": deltas,
        },
    )
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"] == "Send one delta per chunk"
    assert len(reads.regions) == 1
    assert reads.chunks == {(0, 0, 0)}


def test_a_label_that_does_not_match_stops_the_other_reads(
    ada, settings, migrated_database_url, monkeypatch
):
    # The prediction says 2 where every one of the 512 labels says 1.
    rows = [0, 0, 0, 1, 512, 4096]
    big = make_project(ada, settings, migrated_database_url, "Slab", SLAB)
    prediction = add_sparse_prediction(
        settings, migrated_database_url, big, SLAB, [(rows, 2)]
    )
    mask = np.ones((1, 512, 4096), dtype=bool)
    deltas = deltas_for(mask, (0, 0, 0), value=1, only_if="unlabeled")
    latency = 0.05
    reads = Reads(monkeypatch, latency)
    # A pool of two, so most of the accept's reads wait their turn in its queue,
    # as they do when other accepts are using the process's threads.
    pool = ThreadPoolExecutor(2)
    monkeypatch.setattr(api_labels, "_readers", pool)
    try:
        refused = accept_slab(ada, big, prediction, rows, deltas)
        assert refused.status_code == 422, refused.text
        assert refused.json()["detail"] == "Those labels don't match the prediction"
        # The reads that had begun finished, those still waiting never began, and
        # when the request has its answer none is left running.
        began = len(reads.regions)
        assert 0 < began < api_labels.ACCEPT_READS_AT_ONCE
        assert reads.under_way == 0
        time.sleep(3 * latency)
        assert len(reads.regions) == began
    finally:
        pool.shutdown()


def test_accepts_at_once_share_the_processs_reader_threads(
    ada, settings, migrated_database_url, monkeypatch
):
    # 20 accepts of 40 chunks each, at once, each read taking as long as one from
    # a store a way off does: 800 reads, from one row of 40 chunks of the slab
    # for each. One of them (7) is wrong, saying 1 where the prediction says 2.
    accepts, chunks_each, wrong = 20, 40, 7
    columns = chunks_each * 64
    big = make_project(ada, settings, migrated_database_url, "Slab", SLAB)
    prediction = add_sparse_prediction(settings, migrated_database_url, big, SLAB, [])

    def says(region):
        """What the prediction holds: 1, but 2 in the wrong one's row."""
        size = tuple(r.stop - r.start for r in region)
        return np.full(size, 2 if region[1].start // 64 == wrong else 1, np.uint8)

    # It is the reads that are measured, so they are made up (the prediction is
    # empty) and the label writer is left out (its tests are above), and nothing
    # else the process does is in the way of the timing.
    async def writes_nothing(*args, **kwargs):
        return labels.OpResult(seq=1, chunks=[])

    monkeypatch.setattr(labels, "apply_edit", writes_nothing)

    def accept(row):
        mask = np.ones((1, 64, columns), dtype=bool)
        deltas = deltas_for(mask, (0, row * 64, 0), value=1, only_if="unlabeled")
        assert len(deltas) == chunks_each
        box = [0, row * 64, 0, 1, (row + 1) * 64, columns]
        return accept_slab(ada, big, prediction, box, deltas)

    latency = 0.05
    # A region's row of chunks tells which accept it is for.
    reads = Reads(
        monkeypatch,
        latency,
        owner=lambda region: region[1].start // 64,
        meet=api_labels.PREDICTION_READERS,
        gives=says,
    )
    answers = {}
    threads = [
        threading.Thread(target=lambda row=row: answers.update({row: accept(row)}))
        for row in range(accepts)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(120)
    assert sorted(answers) == list(range(accepts))

    # The wrong one is refused as it would be on its own, and no other is.
    refused = answers[wrong]
    assert refused.status_code == 422, refused.text
    assert refused.json()["detail"] == "Those labels don't match the prediction"
    for row in set(range(accepts)) - {wrong}:
        assert answers[row].status_code == 201, (row, answers[row].text)

    # In all, no more reads were under way than the process has threads for, and
    # it used all of them, and no more threads than those (not a pool for each
    # accept, which would be 20 times 16): the total is bounded in the process.
    assert reads.most_at_once == api_labels.PREDICTION_READERS
    assert len(reads.threads) <= api_labels.PREDICTION_READERS
    assert reads.under_way == 0
    # No accept had more than its share of them under way.
    assert max(reads.most_at_once_of.values()) <= api_labels.ACCEPT_READS_AT_ONCE
    # Each accept's chunks, every one and no others, were read once.
    columns_read: dict[int, list[int]] = {row: [] for row in range(accepts)}
    for region in reads.regions:
        columns_read[region[1].start // 64].append(region[2].start // 64)
    for row in set(range(accepts)) - {wrong}:
        assert sorted(columns_read[row]) == list(range(chunks_each)), row
    # The wrong one stopped its own reads that had not begun (and only its own).
    assert 0 < len(columns_read[wrong]) <= api_labels.ACCEPT_READS_AT_ONCE
    # Not all the reads in turn (800 × 50 ms), as far from it as this is: a bound
    # for a loaded machine, not a measure.
    assert reads.took < accepts * chunks_each * latency / 2


def test_accepting_refuses_a_prediction_of_an_image_that_was_replaced(
    ada, project, settings, migrated_database_url
):
    named = {
        "model_id": None,
        "image_artifact_id": current_image(migrated_database_url, project),
    }
    prediction = add_prediction(settings, migrated_database_url, project, inputs=named)
    roi = add_roi(ada, project)
    one = np.ones((1, 1, 1), dtype=bool)
    box = [0, 0, 0, 10, 10, 10]

    def accept(z, *, into_roi=False):
        where = {"roi_id": roi} if into_roi else {"box": box}
        return ada.post(
            f"/api/projects/{project}/labels/accept",
            json={
                "client_op_id": str(uuid.uuid4()),
                "prediction_artifact_id": prediction,
                "deltas": deltas_for(one, (z, 0, 0), value=2, only_if="unlabeled"),
                **where,
            },
        )

    assert accept(0).status_code == 201
    assert accept(1, into_roi=True).status_code == 201
    # A new image takes the place of the one the prediction was made from.
    add_image(migrated_database_url, project)
    for refused in (accept(2), accept(3, into_roi=True)):
        assert refused.status_code == 409, refused.text
        assert (
            refused.json()["detail"]
            == "That prediction is of an image that was replaced."
        )
    assert ada.get(f"/api/projects/{project}/labels/counts").json()["2"] == 2


def test_a_prediction_that_does_not_cover_the_labels_is_refused(
    ada, project, settings, migrated_database_url, monkeypatch
):
    reads = Reads(monkeypatch)
    # Made for a smaller image, and not saying which.
    small = add_prediction(settings, migrated_database_url, project, shape=(10, 10, 10))
    big = {"bbox": [0, 0, 0, 20, 20, 20], "kind": "cube"}
    roi = ada.post(f"/api/projects/{project}/rois", json=big).json()["id"]
    mask = np.ones((1, 1, 15), dtype=bool)
    into_box = accept_box(
        ada, project, small, [0, 0, 0, 20, 20, 20], mask, (0, 0, 0), 1
    )
    into_roi = ada.post(
        f"/api/projects/{project}/labels/accept",
        json={
            "client_op_id": str(uuid.uuid4()),
            "prediction_artifact_id": small,
            "roi_id": roi,
            "deltas": deltas_for(mask, (0, 0, 0), value=1, only_if="unlabeled"),
        },
    )
    for refused in (into_box, into_roi):
        assert refused.status_code == 422, refused.text
        assert refused.json()["detail"] == "That prediction doesn't cover those labels"
    # It's seen from its shape, without reading any of it.
    assert not reads.regions


def test_predictions_labels_were_accepted_into_a_box_from_are_kept(
    ada, project, settings, migrated_database_url
):
    me = uuid.UUID(ada.get("/api/auth/session").json()["user"]["id"])
    accepted, unused = [], []
    # A proposal, and a prediction of the whole image.
    for z, slot in enumerate((artifacts.proposal_slot(me), "prediction")):
        source = add_prediction(settings, migrated_database_url, project, slot)
        response = accept_box(
            ada,
            project,
            source,
            [0, 0, 0, 10, 10, 10],
            np.ones((1, 8, 8), dtype=bool),
            (z, 0, 0),
            2,
        )
        assert response.status_code == 201, response.text
        if slot == "prediction":
            # Undone labels still count: a redo brings them back.
            assert (
                toggle(ada, project, response.json()["seq"], "undo").status_code == 201
            )
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


def test_the_history_lists_a_box_accept_without_an_roi(
    ada, project, settings, migrated_database_url
):
    model = add_model(migrated_database_url, project, "rf one")
    prediction = add_prediction(
        settings, migrated_database_url, project, "prediction", {"model_id": model}
    )
    box = [0, 0, 0, 8, 8, 8]
    accepted = accept_box(
        ada, project, prediction, box, np.ones((1, 8, 8), dtype=bool), (3, 0, 0), 2
    )
    assert accepted.status_code == 201, accepted.text
    [entry] = ada.get(f"/api/projects/{project}/labels/ops").json()
    assert entry["source"] == Source.MODEL_VERIFIED
    assert entry["bbox"] == [3, 0, 0, 4, 8, 8]
    assert entry["tool"]["box"] == box
    assert entry["accepted"] == {
        "kind": "prediction",
        "model_id": model,
        "model_name": "rf one",
        "v1_job_id": None,
        "roi_id": None,
    }


def with_number(body: dict, literal: str) -> str:
    """
    `body` as JSON text, with each "@number@" in it written as `literal`: NaN, an
    infinity, or a number too big for a float, which Python's JSON parser takes.
    """
    return json.dumps(body).replace('"@number@"', literal)


@pytest.mark.parametrize("literal", NOT_JSON)
def test_numbers_json_cannot_hold_are_a_422_in_the_labels_api(
    ada, project, settings, migrated_database_url, literal
):
    prediction = add_prediction(settings, migrated_database_url, project)
    base = f"/api/projects/{project}/labels"
    delta = deltas_for(np.ones((1, 1, 1), dtype=bool), (0, 0, 0), value=2)[0]
    accept = {
        "client_op_id": str(uuid.uuid4()),
        "prediction_artifact_id": prediction,
        "box": [0, 0, 0, 10, 10, 10],
        "deltas": [delta],
    }
    ops = {"client_op_id": str(uuid.uuid4()), "deltas": [delta]}
    shown = NOT_JSON[literal]
    # Each: where it is, the request, the error's place, and the input it shows.
    cases = {
        "the accept's box": (
            f"{base}/accept",
            {**accept, "box": [0, 0, 0, "@number@", 10, 10]},
            ["body", "box", 3],
            shown,
        ),
        "an edit's base version": (
            f"{base}/ops",
            {**ops, "deltas": [{**delta, "base_version": "@number@"}]},
            ["body", "deltas", 0, "base_version"],
            shown,
        ),
        "a delta's box": (
            f"{base}/ops",
            {**ops, "deltas": [{**delta, "box": [0, 0, 0, 1, 1, "@number@"]}]},
            ["body", "deltas", 0, "box", 5],
            shown,
        ),
        "a delta's value": (
            f"{base}/accept",
            {**accept, "deltas": [{**delta, "value": "@number@"}]},
            ["body", "deltas", 0, "value"],
            shown,
        ),
        "a class's color": (
            f"{base}/classes",
            {"name": "tooth", "color": "@number@"},
            ["body", "color"],
            shown,
        ),
        "an edit's strict flag": (
            f"{base}/ops",
            {**ops, "strict": "@number@"},
            ["body", "strict"],
            shown,
        ),
        # Free-form, so it passes as a model field and reaches the database unless
        # it is checked: here the input is the whole tool, with the number as text.
        "an edit's tool": (
            f"{base}/ops",
            {**ops, "tool": {"x": "@number@"}},
            ["body", "tool"],
            {"x": shown},
        ),
    }
    before = ada.get(f"{base}/counts").json()
    for name, (url, body, place, seen) in cases.items():
        response = ada.post(
            url,
            content=with_number(body, literal),
            headers={"content-type": "application/json"},
        )
        assert response.status_code == 422, (name, response.status_code, response.text)
        errors = strict_json(response.text)["detail"]
        [error] = [e for e in errors if e["loc"] == place]
        assert error["input"] == seen, name
        assert error["msg"], name
    # Nothing was written or made.
    assert ada.get(f"{base}/counts").json() == before
    assert len(ada.get(f"{base}/classes").json()) == 2
    assert ada.get(f"{base}/ops").json() == []


@pytest.mark.parametrize("literal", NOT_JSON)
def test_numbers_json_cannot_hold_get_their_422_over_a_real_connection(
    ada, project, settings, migrated_database_url, live_server, literal
):
    prediction = add_prediction(settings, migrated_database_url, project)
    base = f"/api/projects/{project}/labels"
    delta = deltas_for(np.ones((1, 1, 1), dtype=bool), (0, 0, 0), value=2)[0]
    ops = {
        "client_op_id": str(uuid.uuid4()),
        "deltas": [{**delta, "base_version": "@n@"}],
    }
    accept = {
        "client_op_id": str(uuid.uuid4()),
        "prediction_artifact_id": prediction,
        "box": [0, 0, 0, "@n@", 10, 10],
        "deltas": [delta],
    }
    # The same session, on a real connection, where a request the server can't answer
    # drops the connection instead of getting a 500.
    cookies = {cookie.name: cookie.value for cookie in ada.client.cookies.jar}
    headers = {"content-type": "application/json", "x-csrf-token": ada.csrf_token}
    with httpx2.Client(base_url=live_server, cookies=cookies, timeout=10) as raw:
        for url, body, place in (
            (f"{base}/ops", ops, ["body", "deltas", 0, "base_version"]),
            (f"{base}/accept", accept, ["body", "box", 3]),
        ):
            text = json.dumps(body).replace('"@n@"', literal)
            response = raw.post(url, content=text, headers=headers)
            assert response.status_code == 422, (
                url,
                response.status_code,
                response.text,
            )
            [error] = strict_json(response.text)["detail"]
            assert error["loc"] == place and error["input"] == NOT_JSON[literal]
        assert raw.get(f"{base}/classes").status_code == 200


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
