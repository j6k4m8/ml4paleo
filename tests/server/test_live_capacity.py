"""Live lanes reserve only while authorized, visible browser leases live."""

import datetime
import uuid

import pytest
from helpers import add_worker, run_db, signup
from ml4paleo_server import jobs
from ml4paleo_server.db import Artifact, ArtifactHead, LiveSession, ProjectMember
from ml4paleo_worker.client import ServerClient
from sqlalchemy import delete, update

from ml4paleo.protocol import Tier, WorkerCaps


def setup(new_browser, url):
    browser = new_browser()
    signup(browser)
    project = browser.post("/api/projects", json={"name": "Live test"}).json()
    pid = uuid.UUID(project["id"])

    async def seed(db):
        image = Artifact(
            project_id=pid,
            kind="image",
            state="committed",
            manifest={"shape_czyx": [1, 10, 10, 10]},
        )
        db.add(image)
        await db.flush()
        db.add(ArtifactHead(project_id=pid, slot="image", artifact_id=image.id))
        normal = await jobs.enqueue(
            db, "noop", {}, project_id=pid, tier=Tier.INTERACTIVE
        )
        preview = await jobs.enqueue(
            db, "predict.live", {}, project_id=pid, tier=Tier.INTERACTIVE
        )
        return image.id, normal.id, preview.id

    image, normal, preview = run_db(url, seed)
    token = add_worker(url)
    client = ServerClient(token, http=new_browser().client)
    caps = WorkerCaps(version="test", kinds=["noop", "predict.live"], slots=2)
    sid = uuid.uuid4()
    path = f"/api/projects/{pid}/live-sessions/{sid}"
    assert browser.put(path, json={"image_artifact_id": str(image)}).status_code == 204
    return browser, client, caps, path, pid, sid, image, normal, preview


@pytest.mark.parametrize(
    "release", ["stop", "expire", "remove_member", "replace_image"]
)
def test_reserved_lane_returns_to_normal_work(
    new_browser, migrated_database_url, release
):
    browser, client, caps, path, pid, sid, image, normal, preview = setup(
        new_browser, migrated_database_url
    )
    leased = client.claim(caps, 0, reserve_for_live=True)
    assert leased is not None and leased.job_id == preview
    assert client.claim(caps, 0, reserve_for_live=True) is None
    if release == "stop":
        assert browser.request("DELETE", path).status_code == 204
    else:

        async def end(db):
            if release == "expire":
                await db.execute(
                    update(LiveSession)
                    .where(LiveSession.id == sid)
                    .values(
                        expires_at=datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)
                    )
                )
            elif release == "remove_member":
                await db.execute(
                    delete(ProjectMember).where(ProjectMember.project_id == pid)
                )
            else:
                await db.execute(
                    delete(ArtifactHead).where(ArtifactHead.project_id == pid)
                )

        run_db(migrated_database_url, end)
    leased = client.claim(caps, 0, reserve_for_live=True)
    assert leased is not None and leased.job_id == normal


def test_ordinary_and_single_slots_still_train(new_browser, migrated_database_url):
    browser, client, caps, path, pid, sid, image, normal, preview = setup(
        new_browser, migrated_database_url
    )
    leased = client.claim(caps, 0)
    assert leased is not None and leased.job_id == normal
    # A one-slot worker must never deadlock on a reservation.
    caps = caps.model_copy(update={"slots": 1})
    leased = client.claim(caps, 0, reserve_for_live=True)
    assert leased is not None and leased.job_id == preview


def test_sessions_are_independent_and_cannot_be_released_by_other_users(
    new_browser, migrated_database_url
):
    browser, client, caps, path, pid, sid, image, normal, preview = setup(
        new_browser, migrated_database_url
    )
    other = new_browser()
    user = signup(other, "bob").json()["user"]["id"]

    async def member(db):
        db.add(ProjectMember(project_id=pid, user_id=uuid.UUID(user)))

    run_db(migrated_database_url, member)
    assert other.request("DELETE", path).status_code == 204
    assert other.put(path, json={"image_artifact_id": str(image)}).status_code == 404
    another = f"/api/projects/{pid}/live-sessions/{uuid.uuid4()}"
    assert (
        browser.put(another, json={"image_artifact_id": str(image)}).status_code == 204
    )
    assert browser.request("DELETE", path).status_code == 204
    assert client.claim(caps, 0, reserve_for_live=True).job_id == preview
    assert browser.request("DELETE", another).status_code == 204
    assert client.claim(caps, 0, reserve_for_live=True).job_id == normal
