"""Activity hides live previews without losing older project actions."""

import datetime
import uuid

from helpers import run_db
from ml4paleo_server import jobs
from test_projects import create_project, make_user


def test_activity_filters_live_previews_before_limiting(
    new_browser, migrated_database_url
):
    ada = make_user(new_browser, "ada")
    project = create_project(ada)["id"]
    other = create_project(ada, "Other scan")["id"]
    kinds = [
        "ingest.probe",
        "model.train",
        "predict.prepare",
        "predict.region",
        "export.files",
    ]
    states = ["queued", "leased", "succeeded", "failed", "cancelled"]

    async def seed(db):
        base = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)
        visible = []
        for index, kind in enumerate(kinds):
            job = await jobs.enqueue(db, kind, {}, project_id=uuid.UUID(project))
            job.created_at = base + datetime.timedelta(seconds=index)
            job.status = states[index]
            visible.append(str(job.id))
        previews = []
        for index in range(55):
            job = await jobs.enqueue(
                db, "predict.live", {}, project_id=uuid.UUID(project)
            )
            job.created_at = base + datetime.timedelta(minutes=1, seconds=index)
            job.status = states[index % len(states)]
            previews.append(str(job.id))
        await jobs.enqueue(db, "model.train", {}, project_id=uuid.UUID(other))
        return visible[::-1], previews

    visible, previews = run_db(migrated_database_url, seed)
    url = f"/api/projects/{project}/pipelines"
    # The default remains unchanged for other consumers and diagnostics.
    unfiltered = ada.get(url).json()
    assert len(unfiltered) == 50
    assert {p["kind"] for p in unfiltered} == {"live preview"}
    assert ada.get(url, params={"include_live_previews": "true"}).json() == unfiltered

    activity = ada.get(url, params={"include_live_previews": "false"})
    assert activity.status_code == 200
    assert [p["id"] for p in activity.json()] == visible
    assert {p["kind"] for p in activity.json()} == {
        "ingest",
        "training",
        "prediction",
        "proposal",
        "export",
    }
    # Nothing is deleted, and the annotator can still follow a live job.
    assert ada.get(f"{url}/{previews[-1]}").json()["kind"] == "live preview"


def test_activity_with_only_live_previews_is_empty(new_browser, migrated_database_url):
    ada = make_user(new_browser, "ada")
    project = create_project(ada)["id"]

    async def seed(db):
        await jobs.enqueue(db, "predict.live", {}, project_id=uuid.UUID(project))

    run_db(migrated_database_url, seed)
    url = f"/api/projects/{project}/pipelines?include_live_previews=false"
    assert ada.get(url).json() == []
    bob = make_user(new_browser, "bob")
    assert bob.get(url).status_code == 404
