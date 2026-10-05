"""
Workers end to end: the worker protocol over HTTP, the worker loop, worker
administration, and what happens when workers crash, stop, or are too slow.

The worker loop runs in a thread against the app's test client, so these
tests exercise the real client, server, and queue together.
"""

import datetime
import threading
import time

import pytest
from helpers import CAPS, add_worker, bearer, make_admin, run_db, signup
from ml4paleo_server import housekeeper, jobs
from ml4paleo_server.db import Job, JobAttempt, create_sessionmaker
from ml4paleo_server.jobs.workers import ensure_local_worker, new_worker_token
from ml4paleo_worker import caps as worker_caps
from ml4paleo_worker import cli as worker_cli
from ml4paleo_worker.client import LeaseLost, ServerClient
from ml4paleo_worker.context import PermanentError
from ml4paleo_worker.handlers import HANDLERS
from ml4paleo_worker.main import Worker
from sqlalchemy import select, update

from ml4paleo.protocol import WorkerCaps

PAST = datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)


def wait_for(check, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if value := check():
            return value
        time.sleep(0.05)
    raise AssertionError("timed out")


def enqueue(database_url, payload=None, **options):
    async def add(db):
        job = await jobs.enqueue(db, "noop", payload or {"seconds": 0}, **options)
        return job.id

    return run_db(database_url, add)


def job_row(database_url, job_id) -> Job:
    async def get(db):
        return await db.get(Job, job_id)

    return run_db(database_url, get)


def outcomes(database_url, job_id) -> list[str | None]:
    async def get(db):
        return (
            await db.scalars(
                select(JobAttempt.outcome)
                .where(JobAttempt.job_id == job_id)
                .order_by(JobAttempt.id)
            )
        ).all()

    return run_db(database_url, get)


def make_worker(new_browser, token, handlers=HANDLERS) -> Worker:
    client = ServerClient(token, http=new_browser().client)
    return Worker(
        client,
        CAPS,
        handlers=handlers,
        claim_wait_seconds=0.5,
        heartbeat_seconds=0.1,
    )


def start(worker: Worker, max_jobs: int | None = 1) -> threading.Thread:
    thread = threading.Thread(target=worker.run, kwargs={"max_jobs": max_jobs})
    thread.start()
    return thread


@pytest.fixture
def token(migrated_database_url):
    return add_worker(migrated_database_url)


def test_worker_routes_need_a_worker_token(new_browser, token, migrated_database_url):
    browser = new_browser()
    hello = {"caps": CAPS.model_dump()}
    assert browser.post("/api/worker/v1/hello", json=hello).status_code == 401
    for header in [bearer("m4pw_not-a-real-token"), bearer(token[5:]), {}]:
        response = browser.post("/api/worker/v1/hello", json=hello, headers=header)
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
    # A signed-in person is not a worker.
    signup(browser)
    assert browser.post("/api/worker/v1/hello", json=hello).status_code == 401
    good = browser.post("/api/worker/v1/hello", json=hello, headers=bearer(token))
    assert good.status_code == 200
    assert good.json()["lease_seconds"] == jobs.LEASE.total_seconds()


def test_a_waiting_claim_wakes_when_work_arrives(
    new_browser, token, migrated_database_url
):
    client = ServerClient(token, http=new_browser().client)
    result = {}

    def claim():
        started = time.monotonic()
        result["lease"] = client.claim(CAPS, wait_seconds=20)
        result["seconds"] = time.monotonic() - started

    thread = threading.Thread(target=claim)
    thread.start()
    time.sleep(0.5)
    job_id = enqueue(migrated_database_url)
    thread.join()
    assert result["lease"].job_id == job_id
    # Woken by the notification, well before the five-second poll.
    assert result["seconds"] < 3
    assert client.claim(CAPS, wait_seconds=0.2) is None


def test_a_worker_runs_jobs_and_reports_progress(
    new_browser, token, migrated_database_url
):
    seen = []

    def watched(ctx):
        result = HANDLERS["noop"](ctx)
        seen.append(job_row(migrated_database_url, ctx.job_id).progress)
        return result

    job_id = enqueue(migrated_database_url, {"seconds": 0.5})
    worker = make_worker(new_browser, token, {"noop": watched})
    start(worker).join(timeout=20)
    job = job_row(migrated_database_url, job_id)
    assert job.status == "succeeded"
    assert job.result["seconds"] >= 0.5
    assert seen[0] > 0  # heartbeats carried progress before the job finished
    assert outcomes(migrated_database_url, job_id) == ["succeeded"]


def test_admins_manage_workers_and_jobs(new_browser, migrated_database_url):
    admin, _ = make_admin(new_browser, migrated_database_url)
    created = admin.post("/api/admin/workers", json={"name": "lab-gpu"})
    assert created.status_code == 201
    token = created.json()["token"]
    assert token.startswith("m4pw_")
    assert admin.post("/api/admin/workers", json={"name": "lab-gpu"}).status_code == 409

    job = admin.post("/api/admin/jobs/noop", json={"seconds": 0.2}).json()
    assert job["status"] == "queued"
    worker = make_worker(new_browser, token)
    start(worker).join(timeout=20)
    done = admin.get(f"/api/admin/jobs/{job['id']}").json()
    assert (done["status"], done["worker"]) == ("succeeded", "lab-gpu")
    assert [j["id"] for j in admin.get("/api/admin/jobs?status=succeeded").json()] == [
        job["id"]
    ]
    [listed] = admin.get("/api/admin/workers").json()
    assert listed["online"] and listed["caps"]["kinds"] == ["noop"]

    revoked = admin.request("DELETE", f"/api/admin/workers/{listed['id']}")
    assert revoked.status_code == 204
    hello = {"caps": CAPS.model_dump()}
    response = new_browser().post(
        "/api/worker/v1/hello", json=hello, headers=bearer(token)
    )
    assert response.status_code == 401


def test_cancelling_stops_a_running_job(new_browser, token, migrated_database_url):
    admin, _ = make_admin(new_browser, migrated_database_url)
    job_id = enqueue(migrated_database_url, {"seconds": 60})
    worker = make_worker(new_browser, token)
    thread = start(worker)
    wait_for(lambda: job_row(migrated_database_url, job_id).status == "leased")
    assert admin.post(f"/api/admin/jobs/{job_id}/cancel").status_code == 204
    thread.join(timeout=10)
    assert not thread.is_alive()
    assert job_row(migrated_database_url, job_id).status == "cancelled"
    assert outcomes(migrated_database_url, job_id) == ["cancelled"]


def test_a_dead_workers_job_runs_elsewhere(new_browser, token, migrated_database_url):
    job_id = enqueue(migrated_database_url)
    dead = ServerClient(
        add_worker(migrated_database_url, "dead"), http=new_browser().client
    )
    lease = dead.claim(CAPS, wait_seconds=0)
    assert lease is not None and lease.job_id == job_id

    # The dead worker never sends a heartbeat, so its lease runs out.
    async def expire_and_reap(db):
        await db.execute(update(Job).values(lease_expires_at=PAST))
        await db.commit()
        await housekeeper.reap_jobs(create_sessionmaker(db.bind))
        await db.execute(update(Job).values(not_before=PAST))

    run_db(migrated_database_url, expire_and_reap)
    start(make_worker(new_browser, token)).join(timeout=20)
    assert job_row(migrated_database_url, job_id).status == "succeeded"
    # A late report from the dead worker is refused, not recorded.
    with pytest.raises(LeaseLost):
        dead.complete(job_id, lease.lease_token, {"stale": True})
    assert "stale" not in job_row(migrated_database_url, job_id).result
    assert outcomes(migrated_database_url, job_id) == ["expired", "succeeded"]


def test_crashed_handlers_are_retried(
    new_browser, token, migrated_database_url, monkeypatch
):
    monkeypatch.setattr(jobs.queue, "FIRST_RETRY", datetime.timedelta(0))
    calls = []

    def flaky(ctx):
        calls.append(ctx.lease.attempt)
        if len(calls) == 1:
            raise OSError("disk hiccup")
        return {"ok": True}

    job_id = enqueue(migrated_database_url)
    start(make_worker(new_browser, token, {"noop": flaky}), max_jobs=2).join(timeout=20)
    job = job_row(migrated_database_url, job_id)
    assert (job.status, job.attempts, calls) == ("succeeded", 2, [1, 2])
    assert outcomes(migrated_database_url, job_id) == ["failed", "succeeded"]


def test_permanent_errors_are_not_retried(new_browser, token, migrated_database_url):
    def broken(ctx):
        raise PermanentError("this file is not an image")

    job_id = enqueue(migrated_database_url)
    start(make_worker(new_browser, token, {"noop": broken})).join(timeout=20)
    job = job_row(migrated_database_url, job_id)
    assert (job.status, job.attempts) == ("failed", 1)
    assert "not an image" in job.error


def test_stopping_a_worker_gives_its_job_back(
    new_browser, token, migrated_database_url
):
    job_id = enqueue(migrated_database_url, {"seconds": 60})
    worker = make_worker(new_browser, token)
    thread = start(worker, max_jobs=None)
    wait_for(lambda: job_row(migrated_database_url, job_id).status == "leased")
    worker.stop()
    thread.join(timeout=10)
    assert not thread.is_alive()
    job = job_row(migrated_database_url, job_id)
    assert (job.status, job.attempts) == ("queued", 0)
    assert outcomes(migrated_database_url, job_id) == ["released"]


def test_check_workers_command(new_browser, token, migrated_database_url, monkeypatch):
    from ml4paleo_server import cli

    monkeypatch.setenv("M4P_DATABASE_URL", migrated_database_url)
    assert cli.main(["check-workers", "--timeout", "1"]) == 1
    worker = make_worker(new_browser, token)
    thread = start(worker, max_jobs=None)
    try:
        assert cli.main(["check-workers", "--timeout", "20"]) == 0
    finally:
        worker.stop()
        thread.join(timeout=10)


def test_the_local_worker_token_can_be_rotated(new_browser, migrated_database_url):
    old, new = new_worker_token(), new_worker_token()
    hello = {"caps": CAPS.model_dump()}

    async def register(db, token):
        await ensure_local_worker(db, token)

    run_db(migrated_database_url, lambda db: register(db, old))
    browser = new_browser()
    assert (
        browser.post("/api/worker/v1/hello", json=hello, headers=bearer(old)).json()[
            "name"
        ]
        == "local"
    )
    run_db(migrated_database_url, lambda db: register(db, new))
    assert (
        browser.post(
            "/api/worker/v1/hello", json=hello, headers=bearer(old)
        ).status_code
        == 401
    )
    assert (
        browser.post(
            "/api/worker/v1/hello", json=hello, headers=bearer(new)
        ).status_code
        == 200
    )


def test_the_worker_cli_refuses_to_send_its_token_in_the_clear(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text(new_worker_token())
    for argv in [
        ["--server", "http://ml4paleo.example.org", "--token-file", str(token_file)],
        ["--server", "https://ml4paleo.example.org", "--token-file", "/dev/null"],
    ]:
        with pytest.raises(SystemExit) as exit_info:
            worker_cli.main(argv)
        assert exit_info.value.code == 2


def test_gpus_are_detected_from_nvidia_smi(monkeypatch):
    class Done:
        stdout = "24564\n16384\n"

    monkeypatch.setattr(worker_caps.subprocess, "run", lambda *a, **k: Done())
    caps = worker_caps.detect(["v1-volume"])
    assert caps.labels == ["gpu", "v1-volume"]
    assert caps.vram_gb == 24.0

    def missing(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(worker_caps.subprocess, "run", missing)
    assert worker_caps.detect().labels == []
    assert isinstance(worker_caps.detect(), WorkerCaps)
