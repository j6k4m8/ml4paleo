"""
Projects, collaborators, the audit log, and quotas.
"""

import uuid

import pytest
from fastapi import HTTPException
from helpers import make_admin, run_db, signup
from ml4paleo_server import quotas
from ml4paleo_server.db import AuditEvent, User, UserUsage
from sqlalchemy import select, update


def make_user(new_browser, username, **extra):
    browser = new_browser()
    assert signup(browser, username=username, **extra).status_code == 201
    return browser


def create_project(browser, name="Allosaurus skull"):
    response = browser.post("/api/projects", json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()


def test_create_list_get_and_rename(new_browser):
    ada = make_user(new_browser, "ada")
    project = create_project(ada)
    assert project["owner"] == "ada"
    assert [m["username"] for m in project["members"]] == ["ada"]
    assert [p["id"] for p in ada.get("/api/projects").json()] == [project["id"]]

    renamed = ada.patch(f"/api/projects/{project['id']}", json={"name": "  Skull 2  "})
    assert renamed.json()["name"] == "Skull 2"
    assert ada.post("/api/projects", json={"name": "   "}).status_code == 422


def test_other_peoples_projects_look_missing(new_browser):
    ada = make_user(new_browser, "ada")
    bob = make_user(new_browser, "bob")
    project_id = create_project(ada)["id"]
    missing = str(uuid.uuid4())
    for path in [project_id, missing]:
        url = f"/api/projects/{path}"
        assert bob.get(url).status_code == 404
        assert bob.patch(url, json={"name": "mine now"}).status_code == 404
        assert bob.request("DELETE", url).status_code == 404
        assert bob.get(f"{url}/members").status_code == 404
        assert bob.post(f"{url}/members", json={"user": "bob"}).status_code == 404
    assert bob.get("/api/projects").json() == []


def test_collaborators_get_full_access_but_cannot_delete(new_browser):
    ada = make_user(new_browser, "ada")
    bob = make_user(new_browser, "bob", email="bob@example.org")
    project_id = create_project(ada)["id"]
    members = ada.post(
        f"/api/projects/{project_id}/members", json={"user": "Bob@Example.org"}
    ).json()
    assert sorted(m["username"] for m in members) == ["ada", "bob"]

    assert (
        bob.patch(f"/api/projects/{project_id}", json={"name": "Ours"}).status_code
        == 200
    )
    assert bob.request("DELETE", f"/api/projects/{project_id}").status_code == 403
    assert ada.request("DELETE", f"/api/projects/{project_id}").status_code == 204
    assert ada.get(f"/api/projects/{project_id}").status_code == 404
    assert bob.get("/api/projects").json() == []


def test_membership_changes(new_browser):
    ada = make_user(new_browser, "ada")
    bob = make_user(new_browser, "bob")
    project = create_project(ada)
    project_id = project["id"]
    ada_id = project["members"][0]["user_id"]
    url = f"/api/projects/{project_id}/members"
    assert ada.post(url, json={"user": "nobody"}).status_code == 404
    bob_id = next(
        m["user_id"]
        for m in ada.post(url, json={"user": "bob"}).json()
        if m["username"] == "bob"
    )
    # Adding twice is harmless.
    assert len(ada.post(url, json={"user": "bob"}).json()) == 2
    # Nobody can remove the owner; a collaborator can leave.
    assert bob.request("DELETE", f"{url}/{ada_id}").status_code == 403
    assert bob.request("DELETE", f"{url}/{bob_id}").status_code == 204
    assert bob.get(f"/api/projects/{project_id}").status_code == 404
    assert ada.request("DELETE", f"{url}/{bob_id}").status_code == 404


def test_disabled_accounts_cannot_be_added(new_browser, migrated_database_url):
    ada = make_user(new_browser, "ada")
    make_user(new_browser, "bob")

    async def disable(db):
        await db.execute(
            update(User).where(User.username == "bob").values(status="disabled")
        )

    run_db(migrated_database_url, disable)
    project_id = create_project(ada)["id"]
    response = ada.post(f"/api/projects/{project_id}/members", json={"user": "bob"})
    assert response.status_code == 404


def test_project_actions_are_audited(new_browser, migrated_database_url):
    ada = make_user(new_browser, "ada")
    make_user(new_browser, "bob")
    project_id = create_project(ada)["id"]
    ada.patch(f"/api/projects/{project_id}", json={"name": "Renamed"})
    ada.post(f"/api/projects/{project_id}/members", json={"user": "bob"})

    async def actions(db):
        return (
            await db.scalars(
                select(AuditEvent.action)
                .where(AuditEvent.target_id == project_id)
                .order_by(AuditEvent.id)
            )
        ).all()

    assert run_db(migrated_database_url, actions) == [
        "project.create",
        "project.rename",
        "project.member.add",
    ]


def test_quota_requests_reach_admins_and_raise_limits(
    new_browser, migrated_database_url
):
    admin, _ = make_admin(new_browser, migrated_database_url)
    ada = make_user(new_browser, "ada")
    quota = ada.get("/api/me/quota").json()
    assert quota["storage_bytes_limit"] == 10 * 1024**3
    assert quota["trained_models_limit"] == 20
    assert quota["storage_bytes_used"] == 0

    message = {"message": "I have a 200 GB synchrotron scan."}
    assert ada.post("/api/me/quota-requests", json=message).status_code == 202
    assert ada.get("/api/me/quota").json()["open_request"] is True
    [pending] = admin.get("/api/admin/quota-requests").json()
    assert pending["username"] == "ada"

    granted = admin.post(
        f"/api/admin/quota-requests/{pending['id']}",
        json={"decision": "grant", "quota_override": {"storage_gb": 250}},
    )
    assert granted.status_code == 200
    quota = ada.get("/api/me/quota").json()
    assert quota["storage_bytes_limit"] == 250 * 1024**3
    assert quota["trained_models_limit"] == 20  # unchanged default
    assert quota["open_request"] is False
    assert admin.get("/api/admin/quota-requests").json() == []

    # Admins can also set limits directly; null means unlimited.
    ada_id = ada.get("/api/auth/session").json()["user"]["id"]
    admin.put(f"/api/admin/users/{ada_id}/quota", json={"trained_models": None})
    assert ada.get("/api/me/quota").json()["trained_models_limit"] is None


def test_quota_requests_are_rate_limited(new_browser):
    ada = make_user(new_browser, "ada")
    statuses = [
        ada.post("/api/me/quota-requests", json={"message": "more please"}).status_code
        for _ in range(4)
    ]
    assert statuses == [202, 202, 202, 429]


def test_quota_checks(settings, migrated_database_url):
    async def scenario(db):
        user = User(username="ada", quota_override={"trained_models": 2})
        db.add(user)
        await db.flush()
        db.add(UserUsage(user_id=user.id, storage_bytes=9 * 1024**3, trained_models=2))
        await db.flush()
        await quotas.check_storage(db, settings, user, 1024**3)  # exactly at 10 GB
        with pytest.raises(HTTPException) as too_big:
            await quotas.check_storage(db, settings, user, 1024**3 + 1)
        with pytest.raises(HTTPException) as too_many:
            await quotas.check_trained_models(db, settings, user)
        user.quota_override = {"storage_gb": None, "trained_models": 3}
        await quotas.check_storage(db, settings, user, 10**15)
        await quotas.check_trained_models(db, settings, user)
        return too_big.value.detail, too_many.value.detail

    assert run_db(migrated_database_url, scenario) == (
        "storage_quota_exceeded",
        "trained_model_quota_exceeded",
    )
