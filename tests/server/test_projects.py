"""
Projects, collaborators, the audit log, and quotas.
"""

import asyncio
import uuid

import pytest
from fastapi import HTTPException
from helpers import make_admin, run_db, signup
from ml4paleo_server import quotas
from ml4paleo_server.db import (
    AuditEvent,
    User,
    UserUsage,
    create_engine,
    create_sessionmaker,
)
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
        assert bob.post(f"{url}/members", json={"username": "bob"}).status_code == 404
    assert bob.get("/api/projects").json() == []


def test_collaborators_get_full_access_but_cannot_delete(new_browser):
    ada = make_user(new_browser, "ada")
    bob = make_user(new_browser, "bob", email="bob@example.org")
    project_id = create_project(ada)["id"]
    members = ada.post(
        f"/api/projects/{project_id}/members", json={"username": "Bob"}
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
    assert ada.post(url, json={"username": "nobody"}).status_code == 404
    bob_id = next(
        m["user_id"]
        for m in ada.post(url, json={"username": "bob"}).json()
        if m["username"] == "bob"
    )
    # Adding someone twice says so, and changes nothing.
    again = ada.post(url, json={"username": "bob"})
    assert again.status_code == 409
    assert again.json()["detail"] == "bob is already a member."
    assert ada.post(url, json={"username": "ada"}).status_code == 409
    assert len(ada.get(url).json()) == 2
    # Nobody can remove the owner; a collaborator can leave.
    assert bob.request("DELETE", f"{url}/{ada_id}").status_code == 403
    assert bob.request("DELETE", f"{url}/{bob_id}").status_code == 204
    assert bob.get(f"/api/projects/{project_id}").status_code == 404
    assert ada.request("DELETE", f"{url}/{bob_id}").status_code == 404


def test_collaborators_cannot_be_found_by_email(new_browser):
    ada = make_user(new_browser, "ada")
    make_user(new_browser, "bob", email="bob@example.org")
    url = f"/api/projects/{create_project(ada)['id']}/members"
    registered = ada.post(url, json={"username": "bob@example.org"})
    unregistered = ada.post(url, json={"username": "carol@example.org"})
    assert registered.status_code == unregistered.status_code == 404
    assert registered.json() == unregistered.json()


def test_disabled_accounts_cannot_be_added(new_browser, migrated_database_url):
    ada = make_user(new_browser, "ada")
    make_user(new_browser, "bob")

    async def disable(db):
        await db.execute(
            update(User).where(User.username == "bob").values(status="disabled")
        )

    run_db(migrated_database_url, disable)
    project_id = create_project(ada)["id"]
    response = ada.post(f"/api/projects/{project_id}/members", json={"username": "bob"})
    assert response.status_code == 404


def test_project_actions_are_audited(new_browser, migrated_database_url):
    ada = make_user(new_browser, "ada")
    make_user(new_browser, "bob")
    project_id = create_project(ada)["id"]
    ada.patch(f"/api/projects/{project_id}", json={"name": "Renamed"})
    ada.post(f"/api/projects/{project_id}/members", json={"username": "bob"})

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


def test_quota_reservations(settings, migrated_database_url):
    async def scenario(db):
        user = User(username="ada", quota_override={"trained_models": 2})
        db.add(user)
        await db.flush()
        await quotas.reserve_storage(db, settings, user, 9 * 1024**3)
        await quotas.reserve_storage(db, settings, user, 1024**3)  # exactly 10 GB
        with pytest.raises(HTTPException) as too_big:
            await quotas.reserve_storage(db, settings, user, 1)
        await quotas.release_storage(db, user.id, 1024**3)
        await quotas.reserve_storage(db, settings, user, 1024**3)
        for _ in range(2):
            await quotas.reserve_trained_model(db, settings, user)
        with pytest.raises(HTTPException) as too_many:
            await quotas.reserve_trained_model(db, settings, user)
        user.quota_override = {"storage_gb": None, "trained_models": 3}
        await quotas.reserve_storage(db, settings, user, 10**15)
        await quotas.reserve_trained_model(db, settings, user)
        usage = await quotas.usage_for(db, user.id)
        return (
            too_big.value.detail,
            too_many.value.detail,
            usage.storage_bytes,
            usage.trained_models,
        )

    assert run_db(migrated_database_url, scenario) == (
        "storage_quota_exceeded",
        "trained_model_quota_exceeded",
        10 * 1024**3 + 10**15,
        3,
    )


@pytest.mark.parametrize("has_usage_row", [True, False])
def test_concurrent_reservations_cannot_overshoot_the_limit(
    settings, migrated_database_url, has_usage_row
):
    async def race():
        engine = create_engine(migrated_database_url)
        sessionmaker = create_sessionmaker(engine)
        try:
            async with sessionmaker() as db:
                user = User(username="ada")
                db.add(user)
                await db.flush()
                if has_usage_row:
                    db.add(UserUsage(user_id=user.id))
                await db.commit()
                user_id = user.id

            # Each attempt connects and loads the user first, then all reserve
            # at once.
            barrier = asyncio.Barrier(3)

            async def attempt():
                async with sessionmaker() as db:
                    owner = await db.get(User, user_id)
                    assert owner is not None
                    await barrier.wait()
                    try:
                        await quotas.reserve_storage(db, settings, owner, 6 * 1024**3)
                    except HTTPException:
                        return False
                    # Hold the transaction open so the attempts overlap.
                    await asyncio.sleep(0.3)
                    await db.commit()
                    return True

            results = await asyncio.gather(attempt(), attempt(), attempt())
            async with sessionmaker() as db:
                usage = await quotas.usage_for(db, user_id)
                return sorted(results), usage.storage_bytes
        finally:
            await engine.dispose()

    assert asyncio.run(race()) == ([False, False, True], 6 * 1024**3)
