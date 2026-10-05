"""
Uploads straight from browsers to object storage: presigned part URLs,
resuming, completing, quota, and cleanup. Storage is an in-process S3 server.
"""

import datetime
import urllib.error
import urllib.request
from urllib.parse import parse_qs, urlsplit

import pytest
from helpers import SECRET_KEY, run_db, signup
from ml4paleo_server import jobs, uploads
from ml4paleo_server.db import Job, Upload, UserUsage, create_sessionmaker
from ml4paleo_server.settings import Settings
from ml4paleo_server.storage import project_storage
from sqlalchemy import select, update

from ml4paleo.storage import get_bytes

MiB = 1024**2
PAST = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)


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


@pytest.fixture
def small_parts(monkeypatch):
    # S3's smallest allowed part, so tests can use several parts.
    monkeypatch.setattr(uploads, "DEFAULT_PART_SIZE", 5 * MiB)


@pytest.fixture
def ada(new_browser):
    browser = new_browser()
    signup(browser)
    return browser


def make_project(browser) -> str:
    return browser.post("/api/projects", json={"name": "Skull"}).json()["id"]


def put_part(url: str, data: bytes) -> int:
    request = urllib.request.Request(
        url,
        data=data,
        method="PUT",
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        with urllib.request.urlopen(request) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


def storage_used(database_url) -> int:
    async def get(db):
        return await db.scalar(select(UserUsage.storage_bytes)) or 0

    return run_db(database_url, get)


def file_bytes(settings, project_id, upload_id) -> bytes | None:
    grant = project_storage(settings).child(
        f"projects/{project_id}/uploads/{upload_id}"
    )
    return get_bytes(grant, "data")


def test_an_upload_resumes_and_completes(
    ada, settings, migrated_database_url, small_parts
):
    project = make_project(ada)
    data = bytes(range(256)) * (11 * MiB // 256)  # 11 MiB: parts of 5, 5, and 1
    base = f"/api/projects/{project}/uploads"
    created = ada.post(base, json={"filename": "skull.zip", "size": len(data)})
    assert created.status_code == 201
    upload = created.json()
    assert (upload["part_count"], upload["part_size"]) == (3, 5 * MiB)
    assert storage_used(migrated_database_url) == len(data)

    def send(numbers):
        urls = ada.post(
            f"{base}/{upload['id']}/part-urls", json={"parts": numbers}
        ).json()["urls"]
        for number in numbers:
            start = (number - 1) * 5 * MiB
            assert put_part(urls[str(number)], data[start : start + 5 * MiB]) == 200

    # The connection drops after parts 1 and 3...
    send([1, 3])
    status = ada.get(f"{base}/{upload['id']}").json()
    assert status["stored_parts"] == [1, 3]
    unfinished = ada.post(f"{base}/{upload['id']}/complete")
    assert unfinished.status_code == 409
    assert unfinished.json()["detail"]["parts"] == [2]
    # ...and the browser sends only what is missing.
    send([2])
    done = ada.post(f"{base}/{upload['id']}/complete")
    assert done.status_code == 200
    assert done.json()["state"] == "complete"
    assert file_bytes(settings, project, upload["id"]) == data
    # Finishing again is harmless.
    assert ada.post(f"{base}/{upload['id']}/complete").json()["state"] == "complete"
    assert [u["id"] for u in ada.get(base).json()] == [upload["id"]]


def test_a_part_of_the_wrong_size_must_be_sent_again(ada, small_parts):
    project = make_project(ada)
    base = f"/api/projects/{project}/uploads"
    upload = ada.post(base, json={"filename": "scan.tif", "size": 6 * MiB}).json()
    urls = ada.post(f"{base}/{upload['id']}/part-urls", json={"parts": [1, 2]}).json()
    # (Real storage refuses this, since the URL signs the length.)
    put_part(urls["urls"]["1"], b"short")
    put_part(urls["urls"]["2"], b"x" * MiB)
    assert ada.get(f"{base}/{upload['id']}").json()["stored_parts"] == [2]
    response = ada.post(f"{base}/{upload['id']}/complete")
    assert response.json()["detail"]["parts"] == [1]


def test_part_urls_sign_the_part_length(ada, settings, small_parts):
    project = make_project(ada)
    base = f"/api/projects/{project}/uploads"
    upload = ada.post(base, json={"filename": "a.tif", "size": 7 * MiB}).json()
    urls = ada.post(f"{base}/{upload['id']}/part-urls", json={"parts": [2]}).json()
    url = urlsplit(urls["urls"]["2"])
    query = parse_qs(url.query)
    assert f"{url.scheme}://{url.netloc}" == settings.storage.public_endpoint
    assert url.path.endswith(f"/projects/{project}/uploads/{upload['id']}/data")
    assert query["partNumber"] == ["2"]
    assert query["X-Amz-SignedHeaders"] == ["content-length;host"]
    bad = ada.post(f"{base}/{upload['id']}/part-urls", json={"parts": [3]})
    assert bad.status_code == 422


def test_uploads_use_quota_until_cancelled(new_browser, migrated_database_url):
    browser = new_browser()
    signup(browser)

    async def limit(db):
        from ml4paleo_server.db import User

        await db.execute(update(User).values(quota_override={"storage_gb": 0.01}))

    run_db(migrated_database_url, limit)
    project = make_project(browser)
    base = f"/api/projects/{project}/uploads"
    too_big = browser.post(base, json={"filename": "big.zip", "size": 20 * MiB})
    assert too_big.status_code == 403
    upload = browser.post(base, json={"filename": "ok.zip", "size": 8 * MiB}).json()
    assert storage_used(migrated_database_url) == 8 * MiB
    assert browser.request("DELETE", f"{base}/{upload['id']}").status_code == 204
    assert storage_used(migrated_database_url) == 0
    assert browser.get(f"{base}/{upload['id']}").status_code == 404


def _finish(browser, project, size=MiB) -> dict:
    base = f"/api/projects/{project}/uploads"
    upload = browser.post(base, json={"filename": "f.zip", "size": size}).json()
    urls = browser.post(f"{base}/{upload['id']}/part-urls", json={"parts": [1]})
    assert put_part(urls.json()["urls"]["1"], b"z" * size) == 200
    return browser.post(f"{base}/{upload['id']}/complete").json()


def test_finished_uploads_expire_unless_a_job_uses_them(
    ada, settings, migrated_database_url
):
    project = make_project(ada)
    upload = _finish(ada, project)
    grant = {"path": f"projects/{project}/uploads/{upload['id']}", "access": "r"}

    async def reader(db):
        return (await jobs.enqueue(db, "noop", {}, grants=[grant])).id

    job_id = run_db(migrated_database_url, reader)

    async def expire_and_collect(db):
        await db.execute(update(Upload).values(expires_at=PAST))
        await db.commit()
        return await uploads.collect_garbage(create_sessionmaker(db.bind), settings)

    assert run_db(migrated_database_url, expire_and_collect) == 0
    assert file_bytes(settings, project, upload["id"]) is not None

    async def finish_reader(db):
        await db.execute(update(Job).where(Job.id == job_id).values(status="succeeded"))

    run_db(migrated_database_url, finish_reader)
    assert run_db(migrated_database_url, expire_and_collect) == 1
    assert file_bytes(settings, project, upload["id"]) is None
    assert storage_used(migrated_database_url) == 0

    # A job can't be given an upload that is gone.
    async def late_reader(db):
        with pytest.raises(ValueError):
            await jobs.enqueue(db, "noop", {}, grants=[grant])

    run_db(migrated_database_url, late_reader)


def test_unfinished_uploads_are_aborted_when_they_expire(
    ada, settings, migrated_database_url
):
    project = make_project(ada)
    base = f"/api/projects/{project}/uploads"
    upload = ada.post(base, json={"filename": "f.zip", "size": MiB}).json()
    grant = {"path": f"projects/{project}/uploads/{upload['id']}", "access": "r"}

    async def unfinished_reader(db):
        with pytest.raises(ValueError):
            await jobs.enqueue(db, "noop", {}, grants=[grant])

    run_db(migrated_database_url, unfinished_reader)

    async def expire_and_collect(db):
        await db.execute(update(Upload).values(expires_at=PAST))
        await db.commit()
        return await uploads.collect_garbage(create_sessionmaker(db.bind), settings)

    assert run_db(migrated_database_url, expire_and_collect) == 1

    async def state(db):
        return await db.scalar(select(Upload.state))

    assert run_db(migrated_database_url, state) == "aborted"
    assert storage_used(migrated_database_url) == 0


def test_uploads_need_s3_storage(new_browser, migrated_database_url, tmp_path):
    local = Settings(
        database_url=migrated_database_url,
        secret_key=SECRET_KEY,
        storage={"url": f"file://{tmp_path}/data"},
    )
    browser = new_browser(local)
    signup(browser)
    project = make_project(browser)
    response = browser.post(
        f"/api/projects/{project}/uploads", json={"filename": "a.zip", "size": 10}
    )
    assert response.status_code == 501


@pytest.mark.parametrize("filename", ["../evil.zip", "a/b.zip", "a\\b", "x\ny", ""])
def test_filenames_are_plain_names(ada, filename):
    project = make_project(ada)
    response = ada.post(
        f"/api/projects/{project}/uploads", json={"filename": filename, "size": 10}
    )
    assert response.status_code == 422


def test_other_peoples_uploads_look_missing(ada, new_browser):
    project = make_project(ada)
    upload = ada.post(
        f"/api/projects/{project}/uploads", json={"filename": "a.zip", "size": 10}
    ).json()
    bob = new_browser()
    signup(bob, username="bob")
    for method, path in [
        ("GET", f"/api/projects/{project}/uploads"),
        ("GET", f"/api/projects/{project}/uploads/{upload['id']}"),
        ("POST", f"/api/projects/{project}/uploads/{upload['id']}/complete"),
        ("DELETE", f"/api/projects/{project}/uploads/{upload['id']}"),
    ]:
        assert bob.request(method, path, json={}).status_code == 404
