"""
Importing v1 jobs, end to end: claiming one by its id makes a project, and a
worker with the v1 volume brings over the image (in z, y, x), the placed
annotation samples as labels with complete slice ROIs, and the finished
segmentation as the prediction. The first claim wins; admins can release.
"""

import asyncio
import base64
import datetime
import math
import shutil
import threading
import time
import uuid

import numpy as np
import pytest
import v1_volume
import zarr
from helpers import SECRET_KEY, add_worker, bearer, make_admin, run_db, signup
from ml4paleo_server import artifacts, jobs, pipelines
from ml4paleo_server.db import (
    AuditEvent,
    Job,
    User,
    create_engine,
    create_sessionmaker,
)
from ml4paleo_server.db import Worker as WorkerRow
from ml4paleo_server.settings import Settings
from ml4paleo_server.storage import project_storage
from ml4paleo_worker.client import ServerClient
from ml4paleo_worker.context import JobContext, PermanentError
from ml4paleo_worker.handlers import HANDLERS, V1_HANDLERS, v1import
from ml4paleo_worker.main import Worker
from sqlalchemy import select, update

from ml4paleo.labels import BACKGROUND, LABEL_CHUNK_ZYX
from ml4paleo.labels.codec import decode_chunk
from ml4paleo.labels.deltas import split_into_deltas
from ml4paleo.ome import OmeImage
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


FINISHED = ("succeeded", "failed", "cancelled")


def finished(pipelines) -> bool:
    return all(p["status"] in FINISHED for p in pipelines)


def image_in(pipelines) -> bool:
    return all(p["status"] in FINISHED for p in pipelines if p["kind"] == "import")


def run_import(
    database_url,
    live_server,
    browser,
    project,
    volume,
    *,
    v1_handlers=V1_HANDLERS,
    until=finished,
) -> list[dict]:
    """
    Run an import as the v1 override does, a worker with the v1 volume for
    the import's own jobs (`v1_handlers`) and a plain worker for the rest,
    until `until` holds for the project's pipelines, and return them.
    """
    run = uuid.uuid4().hex[:8]
    workers = [
        Worker(
            ServerClient(
                add_worker(database_url, name=f"{name}-{run}"), base_url=live_server
            ),
            WorkerCaps(version="test", kinds=sorted(handlers), labels=labels),
            handlers=handlers,
            claim_wait_seconds=0.5,
            heartbeat_seconds=0.2,
            v1_volume=v1_volume,
        )
        for name, handlers, labels, v1_volume in (
            ("worker-v1", v1_handlers, ["v1-volume"], volume),
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
            pipelines = browser.get(f"/api/projects/{project}/pipelines").json()
            if until(pipelines):
                return pipelines
            time.sleep(0.3)
        raise AssertionError("The import never finished")
    finally:
        for worker in workers:
            worker.stop()
        for thread in threads:
            thread.join(timeout=30)
        for worker in workers:
            worker.client.close()


def ran_on(database_url, project) -> dict[str, set[str]]:
    """The workers ("worker-v1" or "worker-cpu") that ran each kind of job."""

    async def look(db):
        rows = await db.execute(
            select(Job.kind, WorkerRow.name)
            .join(WorkerRow, WorkerRow.id == Job.lease_worker_id)
            .where(Job.project_id == uuid.UUID(project))
        )
        found: dict[str, set[str]] = {}
        for kind, name in rows:
            found.setdefault(kind, set()).add(name.rsplit("-", 1)[0])
        return found

    return run_db(database_url, look)


def refusing_after(count: int):
    """A labels job whose server refuses every sample after the first `count`."""

    def labels(ctx: JobContext):
        send = ctx.apply_label_op
        sent = []

        def apply(op):
            if len(sent) >= count:
                raise ValueError("That sample is damaged.")
            sent.append(op)
            return send(op)

        ctx.apply_label_op = apply
        return v1import.labels(ctx)

    return labels


def statuses(pipelines) -> dict[str, list[tuple[str, str | None]]]:
    """Each kind of pipeline's statuses and errors, oldest first."""
    found: dict[str, list[tuple[str, str | None]]] = {}
    for pipeline in reversed(pipelines):
        found.setdefault(pipeline["kind"], []).append(
            (pipeline["status"], pipeline["error"])
        )
    return found


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

    pipelines = run_import(migrated_database_url, live_server, ada, project, volume)
    assert statuses(pipelines) == {
        "import": [("succeeded", None)],
        "import labels": [("succeeded", None)],
        "import prediction": [("succeeded", None)],
    }
    # The worker with the v1 volume ran the import's own jobs and no others.
    for kind, names in ran_on(migrated_database_url, project).items():
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


def test_a_failed_import_starts_again_when_claimed_again(
    new_browser, settings, migrated_database_url, live_server, volume
):
    ada = new_browser()
    signup(ada)

    def room(gb):
        async def set_quota(db):
            await db.execute(
                update(User)
                .where(User.username == "ada")
                .values(quota_override={"storage_gb": gb})
            )

        run_db(migrated_database_url, set_quota)

    # The scan doesn't fit, so nothing of it is copied.
    room(1e-5)
    claimed = ada.post("/api/v1-jobs/ABC123/claim").json()
    project = claimed["project_id"]
    [failed] = run_import(migrated_database_url, live_server, ada, project, volume)
    assert failed["status"] == "failed"
    assert "of storage left" in failed["error"]
    assert ada.get(f"/api/projects/{project}/labels/classes").json() == []

    # With room, claiming it again starts the import again.
    room(1)
    again = ada.post("/api/v1-jobs/ABC123/claim")
    assert again.status_code == 200
    restarted = again.json()["pipeline_id"]
    assert again.json()["project_id"] == project
    assert restarted not in (None, claimed["pipeline_id"])
    # Stop it once the probe has added the class, then start it once more.
    probing = Worker(
        ServerClient(add_worker(migrated_database_url), base_url=live_server),
        WorkerCaps(version="test", kinds=["v1.probe"], labels=["v1-volume"]),
        handlers={"v1.probe": v1import.probe},
        claim_wait_seconds=0.5,
        v1_volume=volume,
    )
    probing.run(max_jobs=1)
    probing.client.close()
    cancel = f"/api/projects/{project}/pipelines/{restarted}/cancel"
    assert ada.post(cancel).status_code == 204
    last = ada.post("/api/v1-jobs/ABC123/claim").json()["pipeline_id"]
    assert last not in (None, restarted)
    done = run_import(migrated_database_url, live_server, ada, project, volume)
    assert statuses(done) == {
        "import": [
            ("failed", failed["error"]),
            ("cancelled", None),
            ("succeeded", None),
        ],
        "import labels": [("succeeded", None)],
        "import prediction": [("succeeded", None)],
    }
    classes = ada.get(f"/api/projects/{project}/labels/classes").json()
    assert [(c["value"], c["name"]) for c in classes] == [(2, "Foreground")]
    # Now that it has its image, claiming it again just opens it.
    assert ada.post("/api/v1-jobs/ABC123/claim").json() == {
        "project_id": project,
        "pipeline_id": None,
    }


def test_the_labels_and_the_prediction_come_over_on_their_own(
    new_browser, settings, migrated_database_url, live_server, volume
):
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/v1-jobs/ABC123/claim").json()["project_id"]
    run_import(migrated_database_url, live_server, ada, project, volume, until=image_in)
    # Both wait for the image, each in a pipeline of its own...
    waiting = statuses(ada.get(f"/api/projects/{project}/pipelines").json())
    assert waiting == {
        "import": [("succeeded", None)],
        "import labels": [("waiting", None)],
        "import prediction": [("waiting", None)],
    }
    # ...so a sample the server refuses doesn't stop the prediction.
    handlers = {**V1_HANDLERS, "v1.labels": refusing_after(1)}
    pipelines = run_import(
        migrated_database_url, live_server, ada, project, volume, v1_handlers=handlers
    )
    found = statuses(pipelines)
    [(status, error)] = found["import labels"]
    assert status == "failed"
    assert error is not None and "refused sample 1745400100-z07" in error
    assert found["import prediction"] == [("succeeded", None)]
    assert ada.get(f"/api/projects/{project}/prediction").json()["class_values"] == [2]


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

    # Releasing deletes the claimer's project, so the owner can claim it, but
    # the claimer can't take it back from an old link.
    admin, _ = make_admin(new_browser, migrated_database_url)
    assert bob.post("/api/v1-jobs/ABC123/release").status_code == 403
    assert admin.post("/api/v1-jobs/ABC123/release").status_code == 204
    assert admin.post("/api/v1-jobs/ABC123/release").status_code == 404
    assert [p["name"] for p in ada.get("/api/projects").json()] == []
    again = ada.post("/api/v1-jobs/ABC123/claim")
    assert again.status_code == 409
    assert again.json()["detail"] == (
        "An admin released this job from your account. If it's yours, ask them."
    )
    assert bob.post("/api/v1-jobs/ABC123/claim").status_code == 201

    # So does deleting your own project.
    project = ada.post("/api/v1-jobs/FEED01/claim").json()["project_id"]
    assert ada.get(f"/api/projects/{project}").json()["name"] == "v1 job FEED01"
    assert ada.request("DELETE", f"/api/projects/{project}").status_code == 204
    assert bob.post("/api/v1-jobs/FEED01/claim").status_code == 201


def test_releasing_a_job_stops_its_import(new_browser, settings, migrated_database_url):
    ada = new_browser()
    signup(ada)
    claimed = ada.post("/api/v1-jobs/ABC123/claim").json()
    # A worker is probing the job.
    token = add_worker(migrated_database_url)
    worker = new_browser()
    caps = {"version": "test", "kinds": ["v1.probe"], "labels": ["v1-volume"]}
    worker.post("/api/worker/v1/hello", json={"caps": caps}, headers=bearer(token))
    lease = worker.post(
        "/api/worker/v1/claim",
        json={"caps": caps, "wait_seconds": 0},
        headers=bearer(token),
    ).json()["job"]
    assert lease["job_id"] == claimed["pipeline_id"]

    admin, _ = make_admin(new_browser, migrated_database_url)
    assert admin.post("/api/v1-jobs/ABC123/release").status_code == 204
    beat = worker.post(
        f"/api/worker/v1/jobs/{lease['job_id']}/heartbeat",
        json={"lease_token": lease["lease_token"]},
        headers=bearer(token),
    )
    assert beat.json()["cancel"] is True


def test_releasing_doesnt_wait_for_rows_that_refer_to_the_project(
    new_browser, settings, migrated_database_url
):
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/v1-jobs/ABC123/claim").json()["project_id"]
    admin, _ = make_admin(new_browser, migrated_database_url)

    async def scenario():
        engine = create_engine(migrated_database_url)
        try:
            async with create_sessionmaker(engine)() as finishing:
                # As a finishing job adds its artifact's head: a row that refers
                # to the project, until that job commits (which may wait for
                # the pipeline jobs the release locks).
                await artifacts.create_staging(
                    finishing, project_id=uuid.UUID(project), kind="image"
                )
                releasing = asyncio.get_running_loop().run_in_executor(
                    None, lambda: admin.post("/api/v1-jobs/ABC123/release")
                )
                released = await asyncio.wait_for(releasing, 10)
        finally:
            await engine.dispose()
        return released.status_code

    assert asyncio.run(scenario()) == 204
    # The next claim lets go of the deleted project's job.
    bob = new_browser()
    signup(bob, username="bob")
    assert bob.post("/api/v1-jobs/ABC123/claim").status_code == 201


def test_claims_are_limited_and_need_a_v1_volume(
    new_browser, settings, migrated_database_url
):
    limits = {"claims_per_hour": 2, "failed_claims_per_hour": 3}
    limited = settings.model_copy(update={"v1": settings.v1.model_copy(update=limits)})
    # Per account...
    ada = new_browser(limited, address="192.0.2.1")
    signup(ada)
    assert ada.post("/api/v1-jobs/000000/claim").status_code == 404
    assert ada.post("/api/v1-jobs/000001/claim").status_code == 404
    assert ada.post("/api/v1-jobs/ABC123/claim").status_code == 429
    # ...and per address, whatever account is signed in there.
    bob = new_browser(limited, address="192.0.2.1")
    signup(bob, username="bob")
    assert bob.post("/api/v1-jobs/ABC123/claim").status_code == 429
    # Too many misses, from anyone, stop every claim for the hour.
    carol = new_browser(limited, address="192.0.2.2")
    signup(carol, username="carol")
    assert carol.post("/api/v1-jobs/000002/claim").status_code == 404
    dan = new_browser(limited, address="192.0.2.3")
    signup(dan, username="dan")
    stopped = dan.post("/api/v1-jobs/ABC123/claim")
    assert stopped.status_code == 429
    assert int(stopped.headers["Retry-After"]) > 3000

    async def misses(db):
        rows = await db.scalars(
            select(AuditEvent).where(AuditEvent.action == "v1.claim.miss")
        )
        return sorted((e.target_type, e.target_id, e.ip) for e in rows)

    assert run_db(migrated_database_url, misses) == [
        ("v1_job", "000000", "192.0.2.1"),
        ("v1_job", "000001", "192.0.2.1"),
        ("v1_job", "000002", "192.0.2.2"),
    ]

    unset = settings.model_copy(
        update={"v1": settings.v1.model_copy(update={"volume_path": None})}
    )
    erin = new_browser(unset, address="192.0.2.4")
    signup(erin, username="erin")
    missing = erin.post("/api/v1-jobs/ABC123/claim")
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


def test_a_repeated_label_op_gets_its_first_result(
    new_browser, settings, migrated_database_url
):
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    classes = f"/api/projects/{project}/labels/classes"
    value = ada.post(classes, json={"name": "Foreground", "color": "#f2c14e"}).json()[
        "value"
    ]

    async def add(db):
        image = await artifacts.create_staging(
            db, project_id=uuid.UUID(project), kind="image", head_slot="image"
        )
        image.state = "committed"
        image.manifest = {"shape_czyx": [1, 4, 8, 8]}
        await artifacts.set_head(db, image)
        await jobs.enqueue(
            db,
            "v1.labels",
            {},
            project_id=uuid.UUID(project),
            required_labels=["v1-volume"],
        )

    run_db(migrated_database_url, add)
    token = add_worker(migrated_database_url)
    worker = new_browser()
    caps = {"version": "test", "kinds": ["v1.labels"], "labels": ["v1-volume"]}
    worker.post("/api/worker/v1/hello", json={"caps": caps}, headers=bearer(token))
    lease = worker.post(
        "/api/worker/v1/claim",
        json={"caps": caps, "wait_seconds": 0},
        headers=bearer(token),
    ).json()["job"]
    values = np.full((1, 4, 4), value, dtype=np.uint8)
    deltas = split_into_deltas(
        np.ones(values.shape, dtype=bool), (0, 0, 0), values=values
    )
    op = {
        "lease_token": lease["lease_token"],
        "client_op_id": str(uuid.uuid4()),
        "deltas": [
            {
                "key": list(delta.key),
                "box": list(delta.box),
                "mask": base64.b64encode(delta.mask).decode(),
                "values": base64.b64encode(delta.values or b"").decode(),
            }
            for delta in deltas
        ],
    }
    url = f"/api/worker/v1/jobs/{lease['job_id']}/label-ops"
    first = worker.post(url, json=op, headers=bearer(token))
    assert first.status_code == 201
    # The class is retired before the job sends the same op again.
    assert ada.request("DELETE", f"{classes}/{value}").status_code == 204
    again = worker.post(url, json=op, headers=bearer(token))
    assert (again.status_code, again.json()) == (201, first.json())


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


def test_the_import_class_and_new_classes_take_turns(
    new_browser, settings, migrated_database_url
):
    ada = new_browser()
    signup(ada)
    project = ada.post("/api/projects", json={"name": "Skull"}).json()["id"]
    url = f"/api/projects/{project}/labels/classes"

    async def scenario():
        engine = create_engine(migrated_database_url)
        try:
            async with create_sessionmaker(engine)() as importing:
                value = await pipelines.v1import._foreground(
                    importing, uuid.UUID(project)
                )
                # A class added meanwhile waits for the import's to commit...
                adding = asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: ada.post(url, json={"name": "Bone", "color": "#ffffff"}),
                )
                await asyncio.sleep(0.5)
                waited = not adding.done()
                await importing.commit()
                added = await adding
        finally:
            await engine.dispose()
        return value, waited, added.json()["value"]

    # ...and then takes the next value.
    assert asyncio.run(scenario()) == (2, True, 3)


def test_segmentations_must_be_named_as_v1_named_them(volume, tmp_path):
    probed = {
        "kind": "v1",
        "shape_zyx": [20, 30, 40],
        "dtype": "<u2",
        "levels": 1,
        "slabs": [[0, 20]],
        "annotations": 0,
        "skipped_annotations": 0,
    }
    check = pipelines.v1import.check_probe_result
    check({**probed, "segmentation": "1745400150.zarr"})
    wrong = ["latest.zarr", "..", "1745400150.zarr/..", "../FEED01/1.zarr"]
    # Other scripts' digits too (Arabic-Indic 17), which a regex's \d takes.
    wrong.append("١٧.zarr")
    for name in wrong:
        with pytest.raises(ValueError, match="segmentation"):
            check({**probed, "segmentation": name})
        payload = {
            "job_id": "ABC123",
            "segmentation": name,
            "shape_zyx": [20, 30, 40],
            "foreground": 2,
        }
        prediction = StorageGrant(url=(tmp_path / "prediction").as_uri(), access="rw")
        ctx = context(volume, "v1.prediction", payload, [prediction])
        with pytest.raises(PermanentError, match="v1 segmentation"):
            v1import.prediction(ctx)


class Counted:
    """A zarr array that notes how many of its chunks each read touches."""

    def __init__(self, array, touched: list[int]):
        self.array = array
        self.touched = touched

    def __getattr__(self, name):
        return getattr(self.array, name)

    def __getitem__(self, key):
        self.touched.append(
            math.prod(
                (s.stop - 1) // size - s.start // size + 1
                for s, size in zip(key, self.array.chunks, strict=True)
            )
        )
        return self.array[key]


def test_small_workers_read_a_chunk_at_a_time(volume, tmp_path, monkeypatch):
    touched: list[int] = []
    opened = zarr.open_array
    monkeypatch.setattr(
        zarr, "open_array", lambda *args, **kw: Counted(opened(*args, **kw), touched)
    )
    # Room for about one of the fixture's chunks at a time.
    small = 64 * 1024
    image = StorageGrant(url=(tmp_path / "image").as_uri(), access="rw")
    probed = v1import.probe(context(volume, "v1.probe", {"job_id": "ABC123"}, [image]))
    for z_range in probed["slabs"]:
        payload = {"job_id": "ABC123", "z_range": z_range}
        v1import.slab(context(volume, "v1.slab", payload, [image], memory=small))
    stored = np.asarray(OmeImage.open(image).array(0)[0])
    np.testing.assert_array_equal(stored, v1_volume.image().transpose(2, 1, 0))
    assert len(touched) > 1 and max(touched) == 1
    touched.clear()

    grant = StorageGrant(url=(tmp_path / "prediction").as_uri(), access="rw")
    payload = {
        "job_id": "ABC123",
        "segmentation": "1745400150.zarr",
        "shape_zyx": probed["shape_zyx"],
        "foreground": 2,
    }
    v1import.prediction(
        context(volume, "v1.prediction", payload, [grant], memory=small)
    )
    predicted = zarr.open_group(store=zarr_store(grant), mode="r")["class"]
    segmented = v1_volume.segmentation().transpose(2, 1, 0)
    np.testing.assert_array_equal(
        np.asarray(predicted[:]),  # type: ignore[index]
        np.where(segmented > 0, 2, BACKGROUND),
    )
    assert len(touched) > 1 and max(touched) == 1


def test_import_jobs_take_only_v1_job_ids(volume, tmp_path):
    image = StorageGrant(url=(tmp_path / "image").as_uri(), access="rw")
    others = {
        "v1.probe": {},
        "v1.slab": {"z_range": [0, 20]},
        "v1.labels": {"shape_zyx": [20, 30, 40], "foreground": 2},
        "v1.prediction": {
            "segmentation": "1745400150.zarr",
            "shape_zyx": [20, 30, 40],
            "foreground": 2,
        },
    }
    for job_id in ("../ABC1", "abc123", "ABC123/..", 123):
        for kind, payload in others.items():
            ctx = context(volume, kind, {"job_id": job_id, **payload}, [image])
            with pytest.raises(PermanentError, match="isn't a v1 job id"):
                V1_HANDLERS[kind](ctx)


def test_a_worker_without_the_volume_leaves_the_import_to_another(tmp_path):
    image = StorageGrant(url=(tmp_path / "image").as_uri(), access="rw")
    ctx = context(None, "v1.probe", {"job_id": "ABC123"}, [image])
    # Not a PermanentError: the job can still run on a worker with the volume.
    with pytest.raises(RuntimeError, match="no v1 volume"):
        v1import.probe(ctx)
