"""
Artifacts and worker storage: the storage proxy, the credential broker,
committing artifacts when their job succeeds, head slots, and garbage
collection.
"""

import datetime
import threading

import httpx2
import numpy as np
import obstore
import pytest
from helpers import CAPS, add_worker, bearer, run_db
from ml4paleo_server import artifacts, broker, jobs
from ml4paleo_server.db import (
    Artifact,
    ArtifactHead,
    Job,
    Project,
    ProjectMember,
    User,
    UserUsage,
    Worker,
    create_sessionmaker,
)
from ml4paleo_server.storage import project_storage
from ml4paleo_worker.client import ServerClient
from ml4paleo_worker.main import Worker as WorkerLoop
from sqlalchemy import select, update

from ml4paleo.ome import OmeImage, build_pyramid, write_from_provider
from ml4paleo.storage import (
    StorageGrant,
    get_bytes,
    object_store,
    put_bytes,
    write_manifest,
)
from ml4paleo.volume_providers import NumpyVolumeProvider

PAST = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)
SMALL_CHUNKS = {"chunk_zyx": (2, 2, 2), "shard_zyx": (2, 4, 4)}


@pytest.fixture(params=["file", "s3"])
def settings(request, migrated_database_url, tmp_path):
    """
    Every test here runs with project storage on local disk and on S3 (an
    in-process S3 server), since uploads, listings, and aborted uploads
    behave differently on each.
    """
    from helpers import SECRET_KEY
    from ml4paleo_server.settings import Settings

    storage = {"url": f"file://{tmp_path}/data"}
    if request.param == "s3":
        storage = {
            "url": f"s3://{request.getfixturevalue('s3_bucket')}/{tmp_path.name}",
            "endpoint": request.getfixturevalue("s3_endpoint"),
            "access_key_id": "test",
            "secret_access_key": "test",
            "region": "us-east-1",
        }
    return Settings(
        database_url=migrated_database_url, secret_key=SECRET_KEY, storage=storage
    )


def make_project(database_url, quota_override=None):
    async def add(db):
        user = User(username="ada", quota_override=quota_override)
        db.add(user)
        await db.flush()
        project = Project(name="Skull", owner_id=user.id)
        db.add(project)
        await db.flush()
        db.add(ProjectMember(project_id=project.id, user_id=user.id))
        return project.id

    return run_db(database_url, add)


def stage(database_url, project_id, *, head_slot="image", extra_grants=()):
    """
    Create a staging artifact and the job that produces it; return their ids.
    """

    async def add(db):
        artifact = await artifacts.create_staging(
            db, project_id=project_id, kind="image", head_slot=head_slot
        )
        job = await jobs.enqueue(
            db,
            "noop",
            {},
            project_id=project_id,
            grants=[artifacts.grant_for(artifact), *extra_grants],
        )
        artifact.produced_by_job = job.id
        return artifact.id, job.id

    return run_db(database_url, add)


def artifact_row(database_url, artifact_id) -> Artifact:
    async def get(db):
        return await db.get(Artifact, artifact_id)

    return run_db(database_url, get)


def storage_used(database_url) -> int:
    async def get(db):
        return await db.scalar(select(UserUsage.storage_bytes)) or 0

    return run_db(database_url, get)


def files_of(settings, artifact) -> StorageGrant:
    return project_storage(settings).child(artifacts.artifact_path(artifact))


def finish(settings, database_url, job_id, *, write=True):
    """
    Claim a job as a local worker, optionally write an artifact's files and
    manifest straight to storage, and report success with the commit check.
    """

    async def run(db):
        worker = await db.scalar(select(Worker).where(Worker.name == "w"))
        if worker is None:
            worker = Worker(name="w", pool="local", token_hash="0" * 64, caps={})
            db.add(worker)
            await db.flush()
        await db.execute(update(Job).where(Job.id == job_id).values(not_before=PAST))
        claimed = await jobs.claim(db, worker, CAPS)
        assert claimed is not None and claimed.job.id == job_id
        produced = (
            await db.scalars(select(Artifact).where(Artifact.produced_by_job == job_id))
        ).all()
        for artifact in produced if write else []:
            grant = files_of(settings, artifact)
            put_bytes(grant, "data/0", b"x" * 1000)
            write_manifest(grant, {"kind": "test"})

        async def check(job):
            await artifacts.commit_outputs(db, settings, job)

        try:
            await jobs.complete(db, job_id, worker, claimed.lease_token, {}, check)
        except jobs.Rejected as exc:
            return str(exc)
        return None

    return run_db(database_url, run)


def test_a_worker_writes_an_image_through_the_proxy(
    settings, migrated_database_url, live_server
):
    project_id = make_project(migrated_database_url)
    artifact_id, job_id = stage(migrated_database_url, project_id)
    rng = np.random.default_rng(0)
    data = rng.integers(0, 60000, size=(10, 6, 5), dtype=np.uint16)  # (x, y, z)

    def ingest(ctx):
        grant = ctx.grants[0]
        assert grant.scheme == "http" and grant.access == "rw"
        provider = NumpyVolumeProvider(data)
        image = OmeImage.create(
            grant, shape_czyx=(1, 5, 6, 10), dtype=np.uint16, **SMALL_CHUNKS
        )
        write_from_provider(provider, image)
        build_pyramid(image, "mean")
        write_manifest(grant, {"levels": image.num_levels})
        return {"levels": image.num_levels}

    client = ServerClient(add_worker(migrated_database_url), base_url=live_server)
    loop = WorkerLoop(client, CAPS, handlers={"noop": ingest}, claim_wait_seconds=1)
    thread = threading.Thread(target=loop.run, kwargs={"max_jobs": 1})
    thread.start()
    thread.join(timeout=60)
    client.close()

    artifact = artifact_row(migrated_database_url, artifact_id)
    assert artifact.state == "committed"
    assert artifact.manifest["levels"] >= 2
    on_disk = sum(
        meta["size"]
        for batch in obstore.list(object_store(files_of(settings, artifact)))
        for meta in batch
    )
    assert artifact.bytes == on_disk == storage_used(migrated_database_url)
    stored = OmeImage.open(
        files_of(settings, artifact).model_copy(update={"access": "r"})
    )
    np.testing.assert_array_equal(
        np.asarray(stored.array(0)[0]), data.transpose(2, 1, 0)
    )

    async def current(db):
        return (await artifacts.head(db, project_id, "image")).id

    assert run_db(migrated_database_url, current) == artifact_id


def test_the_proxy_serves_only_the_jobs_grants(
    settings, migrated_database_url, live_server
):
    project_id = make_project(migrated_database_url)
    upload = f"projects/{project_id}/uploads/u1"
    put_bytes(project_storage(settings).child(upload), "scan.tif", b"tiff")
    artifact_id, job_id = stage(
        migrated_database_url,
        project_id,
        extra_grants=[{"path": upload, "access": "r"}],
    )
    client = ServerClient(add_worker(migrated_database_url), base_url=live_server)
    lease = client.claim(CAPS, wait_seconds=0)
    assert lease is not None and lease.job_id == job_id
    output, source = lease.grants
    assert output.url.startswith(f"{live_server}/api/worker/v1/jobs/{job_id}/")
    assert source.access == "r" and get_bytes(source, "scan.tif") == b"tiff"

    big = bytes(range(256)) * 40_000  # 10 MB, one request
    for key, value in [("a/b.txt", b"hello"), ("a/c/d.txt", b"deep"), ("big", big)]:
        put_bytes(output, key, value)
    store = object_store(output)
    listed = sorted(meta["path"] for batch in obstore.list(store) for meta in batch)
    assert listed == ["a/b.txt", "a/c/d.txt", "big"]
    tree = obstore.list_with_delimiter(store, prefix="a")
    assert tree["common_prefixes"] == ["a/c"]
    assert [meta["path"] for meta in tree["objects"]] == ["a/b.txt"]
    assert bytes(obstore.get_range(store, "big", start=5, end=9)) == big[5:9]
    tail = obstore.get(store, "big", options={"range": {"suffix": 7}})
    assert bytes(tail.bytes()) == big[-7:]
    assert obstore.head(store, "big")["size"] == len(big)
    obstore.delete(store, "a/b.txt")
    assert get_bytes(output, "a/b.txt") is None

    raw = httpx2.Client(base_url=live_server)
    base = f"/api/worker/v1/jobs/{job_id}/storage"
    token = bearer(lease.lease_token)
    # A read-only grant refuses writes, and keys can't climb out of a grant.
    assert raw.put(f"{base}/1/scan.tif", content=b"x", headers=token).status_code == 403
    assert (
        raw.put(f"{base}/0/%2E%2E%2Fescape", content=b"x", headers=token).status_code
        == 400
    )
    assert raw.get(f"{base}/2/anything", headers=token).status_code == 404
    assert raw.get(f"{base}/0/big", headers=bearer("wrong")).status_code == 401

    # Once the lease is gone, so is access.
    async def expire(db):
        await db.execute(update(Job).values(lease_expires_at=PAST))

    run_db(migrated_database_url, expire)
    assert raw.get(f"{base}/0/big", headers=token).status_code == 401
    raw.close()
    client.close()


def test_a_completion_without_a_manifest_is_retried(settings, migrated_database_url):
    project_id = make_project(migrated_database_url)
    artifact_id, job_id = stage(migrated_database_url, project_id)
    error = finish(settings, migrated_database_url, job_id, write=False)
    assert "_MANIFEST.json" in error

    async def status(db):
        return await db.scalar(select(Job.status).where(Job.id == job_id))

    assert run_db(migrated_database_url, status) == "queued"
    assert artifact_row(migrated_database_url, artifact_id).state == "staging"
    assert finish(settings, migrated_database_url, job_id) is None
    assert artifact_row(migrated_database_url, artifact_id).state == "committed"


def test_results_over_quota_fail_the_job(settings, migrated_database_url):
    project_id = make_project(migrated_database_url, {"storage_gb": 1e-9})
    artifact_id, job_id = stage(migrated_database_url, project_id)
    error = finish(settings, migrated_database_url, job_id)
    assert "storage_quota_exceeded" in error

    async def status(db):
        return await db.scalar(select(Job.status).where(Job.id == job_id))

    assert run_db(migrated_database_url, status) == "failed"
    assert storage_used(migrated_database_url) == 0

    # The abandoned files are cleaned up once failed artifacts expire.
    no_wait = settings.model_copy(
        update={"storage": settings.storage.model_copy(update={"keep_failed_hours": 0})}
    )

    async def collect(db):
        return await artifacts.collect_garbage(create_sessionmaker(db.bind), no_wait)

    assert run_db(migrated_database_url, collect) == 1
    artifact = artifact_row(migrated_database_url, artifact_id)
    assert artifact.state == "deleted"
    assert get_bytes(files_of(settings, artifact), "data/0") is None


def test_a_rejected_commit_reserves_nothing(settings, migrated_database_url):
    # Room for one artifact's files but not two.
    project_id = make_project(migrated_database_url, {"storage_gb": 1500 / 1024**3})
    first, job_id = stage(migrated_database_url, project_id)

    async def second_output(db):
        artifact = await artifacts.create_staging(
            db, project_id=project_id, kind="mesh"
        )
        artifact.produced_by_job = job_id
        return artifact.id

    second = run_db(migrated_database_url, second_output)
    assert "storage_quota_exceeded" in finish(settings, migrated_database_url, job_id)
    assert storage_used(migrated_database_url) == 0
    for artifact_id in (first, second):
        assert artifact_row(migrated_database_url, artifact_id).state == "staging"


def test_new_heads_supersede_old_ones_and_gc_frees_them(
    settings, migrated_database_url
):
    project_id = make_project(migrated_database_url)
    first, first_job = stage(migrated_database_url, project_id)
    finish(settings, migrated_database_url, first_job)
    second, second_job = stage(migrated_database_url, project_id)
    finish(settings, migrated_database_url, second_job)
    assert artifact_row(migrated_database_url, first).state == "superseded"
    first_bytes = artifact_row(migrated_database_url, first).bytes
    assert storage_used(migrated_database_url) == 2 * first_bytes

    # Kept for a week by default...
    async def collect(db, current_settings):
        return await artifacts.collect_garbage(
            create_sessionmaker(db.bind), current_settings
        )

    assert run_db(migrated_database_url, lambda db: collect(db, settings)) == 0
    # ...but a job that still reads the old one keeps it, whatever the setting.
    no_wait = settings.model_copy(
        update={
            "storage": settings.storage.model_copy(update={"keep_superseded_days": 0})
        }
    )
    old = artifact_row(migrated_database_url, first)

    async def reader(db):
        job = await jobs.enqueue(db, "noop", {}, grants=[artifacts.grant_for(old, "r")])
        return job.id

    reader_id = run_db(migrated_database_url, reader)
    assert run_db(migrated_database_url, lambda db: collect(db, no_wait)) == 0

    async def finish_reader(db):
        await db.execute(
            update(Job).where(Job.id == reader_id).values(status="succeeded")
        )

    run_db(migrated_database_url, finish_reader)
    assert run_db(migrated_database_url, lambda db: collect(db, no_wait)) == 1
    assert artifact_row(migrated_database_url, first).state == "deleted"
    assert storage_used(migrated_database_url) == first_bytes
    assert artifact_row(migrated_database_url, second).state == "committed"


def test_deleting_a_project_deletes_its_artifacts(settings, migrated_database_url):
    project_id = make_project(migrated_database_url)
    artifact_id, job_id = stage(migrated_database_url, project_id)
    finish(settings, migrated_database_url, job_id)
    assert storage_used(migrated_database_url) > 0

    async def delete_project_and_collect(db):
        await db.execute(
            update(Project).values(deleted_at=datetime.datetime.now(datetime.UTC))
        )
        await db.commit()
        collected = await artifacts.collect_garbage(
            create_sessionmaker(db.bind), settings
        )
        heads = (await db.scalars(select(ArtifactHead))).all()
        return collected, heads

    assert run_db(migrated_database_url, delete_project_and_collect) == (1, [])
    assert artifact_row(migrated_database_url, artifact_id).state == "deleted"
    assert storage_used(migrated_database_url) == 0


def test_expired_caches_are_collected(settings, migrated_database_url):
    project_id = make_project(migrated_database_url)
    artifact_id, job_id = stage(migrated_database_url, project_id, head_slot=None)
    finish(settings, migrated_database_url, job_id)

    async def expire_and_collect(db):
        await db.execute(update(Artifact).values(expires_at=PAST))
        await db.commit()
        return await artifacts.collect_garbage(create_sessionmaker(db.bind), settings)

    assert run_db(migrated_database_url, expire_and_collect) == 1
    assert storage_used(migrated_database_url) == 0


def test_direct_access_is_only_for_local_workers(settings, migrated_database_url):
    project_id = make_project(migrated_database_url)
    artifact_id, job_id = stage(migrated_database_url, project_id)
    direct = settings.model_copy(
        update={
            "storage": settings.storage.model_copy(update={"worker_access": "direct"})
        }
    )

    async def grants(db):
        job = await db.get(Job, job_id)
        local = Worker(name="l", pool="local", token_hash="1" * 64)
        remote = Worker(name="r", pool="remote", token_hash="2" * 64)
        return (
            broker.grants_for(direct, local, job, "token", "https://x.org")[0],
            broker.grants_for(direct, remote, job, "token", "https://x.org")[0],
            broker.grants_for(settings, local, job, "token", "https://x.org")[0],
        )

    local, remote, default = run_db(migrated_database_url, grants)
    # Direct grants are the server's own location and credentials.
    expected = project_storage(settings).child(
        f"projects/{project_id}/artifacts/{artifact_id}"
    )
    assert (local.url, local.access) == (expected.url, "rw")
    assert local.credentials == expected.credentials
    assert remote.url == f"https://x.org/api/worker/v1/jobs/{job_id}/storage/0"
    assert default.scheme == "https" and default.secret("token") == "token"


@pytest.mark.parametrize(
    "grant",
    [
        {"path": "projects/x/../../etc", "access": "r"},
        {"path": "other/place", "access": "r"},
        {"path": "projects/x", "access": "admin"},
        {"path": "projects/x", "access": "r", "extra": "1"},
    ],
)
def test_job_grants_are_checked(migrated_database_url, grant):
    async def add(db):
        with pytest.raises(ValueError):
            await jobs.enqueue(db, "noop", {}, grants=[grant])

    run_db(migrated_database_url, add)


def test_an_upload_cannot_land_after_its_job_finished(
    settings, migrated_database_url, live_server
):
    project_id = make_project(migrated_database_url)
    artifact_id, job_id = stage(migrated_database_url, project_id)

    async def no_commit(db):
        # Nothing to commit, so the job can finish while the upload runs.
        await db.execute(update(Artifact).values(produced_by_job=None))

    run_db(migrated_database_url, no_commit)
    client = ServerClient(add_worker(migrated_database_url), base_url=live_server)
    lease = client.claim(CAPS, wait_seconds=0)
    halfway = threading.Event()
    go_on = threading.Event()

    def slow_body():
        yield b"x" * 1024
        halfway.set()
        go_on.wait(10)
        yield b"y" * 1024

    outcome = {}

    def upload():
        with httpx2.Client(base_url=live_server, timeout=30) as raw:
            outcome["status"] = raw.put(
                f"/api/worker/v1/jobs/{job_id}/storage/0/late",
                content=slow_body(),
                headers=bearer(lease.lease_token),
            ).status_code

    put_bytes(lease.grants[0], "kept", b"committed data")
    thread = threading.Thread(target=upload)
    thread.start()
    assert halfway.wait(10)
    client.complete(job_id, lease.lease_token, {})
    go_on.set()
    thread.join(timeout=20)
    # The finished job's files can't be deleted any more, either.
    with httpx2.Client(base_url=live_server) as raw:
        deleting = raw.delete(
            f"/api/worker/v1/jobs/{job_id}/storage/0/kept",
            headers=bearer(lease.lease_token),
        )
    client.close()
    assert (outcome["status"], deleting.status_code) == (401, 401)
    # The aborted upload never became visible, on disk or in S3.
    files = files_of(settings, artifact_row(migrated_database_url, artifact_id))
    assert get_bytes(files, "late") is None
    assert get_bytes(files, "kept") == b"committed data"
    listed = [
        meta["path"] for batch in obstore.list(object_store(files)) for meta in batch
    ]
    assert listed == ["kept"]


def test_the_grant_root_is_not_an_object(settings, migrated_database_url, live_server):
    project_id = make_project(migrated_database_url)
    _, job_id = stage(migrated_database_url, project_id)
    client = ServerClient(add_worker(migrated_database_url), base_url=live_server)
    lease = client.claim(CAPS, wait_seconds=0)
    with httpx2.Client(base_url=live_server) as raw:
        url = f"/api/worker/v1/jobs/{job_id}/storage/0/"
        headers = bearer(lease.lease_token)
        assert raw.put(url, content=b"x", headers=headers).status_code == 400
        assert raw.get(url, headers=headers).status_code == 400
        assert raw.delete(url, headers=headers).status_code == 400
    client.close()


def test_collection_waits_for_a_job_being_given_the_artifact(
    settings, migrated_database_url
):
    import asyncio

    from ml4paleo_server.db import create_engine

    project_id = make_project(migrated_database_url)
    first, first_job = stage(migrated_database_url, project_id)
    finish(settings, migrated_database_url, first_job)
    _, second_job = stage(migrated_database_url, project_id)
    finish(settings, migrated_database_url, second_job)
    no_wait = settings.model_copy(
        update={
            "storage": settings.storage.model_copy(update={"keep_superseded_days": 0})
        }
    )
    old = artifact_row(migrated_database_url, first)

    async def race():
        engine = create_engine(migrated_database_url)
        sessionmaker = create_sessionmaker(engine)
        try:
            async with sessionmaker() as enqueuing:
                # A new job is being given the old artifact...
                await jobs.enqueue(
                    enqueuing, "noop", {}, grants=[artifacts.grant_for(old, "r")]
                )
                # ...while collection starts on it.
                collecting = asyncio.create_task(
                    artifacts.collect_garbage(sessionmaker, no_wait)
                )
                await asyncio.sleep(0.5)
                await enqueuing.commit()
            return await asyncio.wait_for(collecting, 10)
        finally:
            await engine.dispose()

    assert asyncio.run(race()) == 0
    artifact = artifact_row(migrated_database_url, first)
    assert artifact.state == "superseded"
    assert get_bytes(files_of(settings, artifact), "data/0") is not None


def test_deleted_artifacts_cannot_be_granted(settings, migrated_database_url):
    project_id = make_project(migrated_database_url)
    artifact_id, job_id = stage(migrated_database_url, project_id, head_slot=None)
    finish(settings, migrated_database_url, job_id)

    async def expire_and_collect(db):
        await db.execute(update(Artifact).values(expires_at=PAST))
        await db.commit()
        await artifacts.collect_garbage(create_sessionmaker(db.bind), settings)

    run_db(migrated_database_url, expire_and_collect)
    artifact = artifact_row(migrated_database_url, artifact_id)

    async def grant(db):
        with pytest.raises(ValueError, match="deleted"):
            await jobs.enqueue(
                db, "noop", {}, grants=[artifacts.grant_for(artifact, "r")]
            )

    run_db(migrated_database_url, grant)


def test_one_stuck_artifact_does_not_stop_collection(
    settings, migrated_database_url, monkeypatch
):
    project_id = make_project(migrated_database_url)
    stuck, stuck_job = stage(migrated_database_url, project_id, head_slot=None)
    other, other_job = stage(migrated_database_url, project_id, head_slot=None)
    for job_id in (stuck_job, other_job):
        finish(settings, migrated_database_url, job_id)
    real_delete = artifacts._delete_files

    async def flaky_delete(settings, artifact):
        if artifact.id == stuck:
            raise OSError("storage is down")
        await real_delete(settings, artifact)

    monkeypatch.setattr(artifacts, "_delete_files", flaky_delete)

    async def expire_and_collect(db):
        await db.execute(update(Artifact).values(expires_at=PAST))
        await db.commit()
        return await artifacts.collect_garbage(create_sessionmaker(db.bind), settings)

    assert run_db(migrated_database_url, expire_and_collect) == 1
    assert artifact_row(migrated_database_url, other).state == "deleted"
    assert artifact_row(migrated_database_url, stuck).state == "deleting"
    # Its quota was released when deletion started.
    assert storage_used(migrated_database_url) == 0
    # The next pass finishes the job.
    monkeypatch.setattr(artifacts, "_delete_files", real_delete)

    async def collect(db):
        return await artifacts.collect_garbage(create_sessionmaker(db.bind), settings)

    assert run_db(migrated_database_url, collect) == 1
    assert artifact_row(migrated_database_url, stuck).state == "deleted"


def test_heads_are_never_collected(settings, migrated_database_url):
    project_id = make_project(migrated_database_url)
    artifact_id, job_id = stage(migrated_database_url, project_id)
    finish(settings, migrated_database_url, job_id)

    async def expire_and_collect(db):
        await db.execute(update(Artifact).values(expires_at=PAST))
        await db.commit()
        return await artifacts.collect_garbage(create_sessionmaker(db.bind), settings)

    assert run_db(migrated_database_url, expire_and_collect) == 0
    assert artifact_row(migrated_database_url, artifact_id).state == "committed"

    async def both(db):
        with pytest.raises(ValueError):
            await artifacts.create_staging(
                db, project_id=project_id, kind="x", head_slot="image", expires_at=PAST
            )

    run_db(migrated_database_url, both)
