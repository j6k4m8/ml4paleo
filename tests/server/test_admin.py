"""
Admins see every account, with what it uses, and can disable one (it is
signed out at once) or enable it again.
"""

import uuid

import pytest
from helpers import PASSWORD, make_admin, outbox, run_db, signup
from ml4paleo_server import jobs
from ml4paleo_server.app import content_security_policy
from ml4paleo_server.cli import main
from ml4paleo_server.db import Job, User
from sqlalchemy import select


def test_admins_list_accounts(new_browser, migrated_database_url):
    admin, _ = make_admin(new_browser, migrated_database_url)
    for name in ("ada", "adam", "bob"):
        signup(new_browser(), username=name, email=f"{name}@example.org")
    users = admin.get("/api/admin/users").json()
    assert [u["username"] for u in users][:3] == ["bob", "adam", "ada"]
    assert users[0]["storage_bytes_used"] == 0
    matched = [u["username"] for u in admin.get("/api/admin/users?q=AD").json()]
    assert matched == ["adam", "ada", "admin"]
    assert [u["username"] for u in admin.get("/api/admin/users?q=bob@").json()] == [
        "bob"
    ]
    # Wildcards are plain characters.
    assert admin.get("/api/admin/users?q=%25").json() == []


def test_only_admins_see_accounts(new_browser):
    ada = new_browser()
    signup(ada)
    assert ada.get("/api/admin/users").status_code == 403


def test_disabling_signs_people_out(new_browser, migrated_database_url):
    admin, _ = make_admin(new_browser, migrated_database_url)
    ada = new_browser()
    signup(ada)
    assert ada.get("/api/projects").status_code == 200
    ada_id = ada.get("/api/auth/session").json()["user"]["id"]

    assert (
        admin.put(
            f"/api/admin/users/{ada_id}/status", json={"status": "disabled"}
        ).status_code
        == 204
    )
    assert ada.get("/api/projects").status_code == 401
    again = new_browser()
    login = {"username": "ada", "password": PASSWORD}
    assert again.post("/api/auth/login", json=login).status_code == 401

    assert (
        admin.put(
            f"/api/admin/users/{ada_id}/status", json={"status": "active"}
        ).status_code
        == 204
    )
    assert again.post("/api/auth/login", json=login).status_code == 200

    admin_id = admin.get("/api/auth/session").json()["user"]["id"]
    selfie = admin.put(
        f"/api/admin/users/{admin_id}/status", json={"status": "disabled"}
    )
    assert selfie.status_code == 409


@pytest.fixture
def smtp_settings(settings):
    return settings.model_copy(
        update={"smtp": settings.smtp.model_copy(update={"host": "smtp.test"})}
    )


def user_id(browser) -> str:
    return browser.get("/api/auth/session").json()["user"]["id"]


def test_invites_with_an_address_are_mailed(
    new_browser, smtp_settings, migrated_database_url
):
    admin, _ = make_admin(lambda: new_browser(smtp_settings), migrated_database_url)
    invite = admin.post("/api/admin/invites", json={"email": "eve@example.org"})
    assert invite.status_code == 201
    [mail] = [m for m in outbox(migrated_database_url) if "invited" in m.subject]
    assert mail.to_address == "eve@example.org"
    assert invite.json()["url"] in mail.body


def test_granting_adds_to_the_limits_someone_has(new_browser, migrated_database_url):
    admin, _ = make_admin(new_browser, migrated_database_url)
    ada = new_browser()
    signup(ada)
    admin.put(
        f"/api/admin/users/{user_id(ada)}/quota",
        json={"storage_gb": 100, "trained_models": None},
    )
    ada.post("/api/me/quota-requests", json={"message": "More models, please."})
    [pending] = admin.get("/api/admin/quota-requests").json()
    admin.post(
        f"/api/admin/quota-requests/{pending['id']}",
        json={"decision": "grant", "quota_override": {"trained_models": 25}},
    )
    quota = ada.get("/api/me/quota").json()
    assert quota["storage_bytes_limit"] == 100 * 1024**3
    assert quota["trained_models_limit"] == 25
    [listed] = admin.get("/api/admin/users?q=ada").json()
    assert listed["storage_bytes_limit"] == 100 * 1024**3


def test_enabling_keeps_an_unconfirmed_email_unconfirmed(
    new_browser, smtp_settings, migrated_database_url
):
    admin, _ = make_admin(lambda: new_browser(smtp_settings), migrated_database_url)
    eve = new_browser(smtp_settings)
    signup(eve, username="eve", email="eve@example.org")
    eve_id = user_id(eve)
    for status in ("disabled", "active"):
        admin.put(f"/api/admin/users/{eve_id}/status", json={"status": status})
    [listed] = admin.get("/api/admin/users?q=eve").json()
    assert (listed["status"], listed["email_confirmed"]) == ("active", False)
    assert (
        admin.put(
            f"/api/admin/users/{uuid.uuid4()}/status", json={"status": "active"}
        ).status_code
        == 404
    )


def test_disabling_stops_what_someone_started(new_browser, migrated_database_url):
    admin, _ = make_admin(new_browser, migrated_database_url)
    ada = new_browser()
    signup(ada)
    ada_id = uuid.UUID(user_id(ada))

    async def start(db):
        job = await jobs.enqueue(db, "noop", {"seconds": 0}, created_by=ada_id)
        return job.id

    job_id = run_db(migrated_database_url, start)
    admin.put(f"/api/admin/users/{ada_id}/status", json={"status": "disabled"})

    async def status(db):
        return (await db.get(Job, job_id)).status

    assert run_db(migrated_database_url, status) == "cancelled"


def test_admins_can_disable_each_other_and_the_server_can_undo_it(
    new_browser, migrated_database_url, monkeypatch, settings
):
    admin, _ = make_admin(new_browser, migrated_database_url)
    other = new_browser()
    signup(other, username="otto")

    async def promote(db):
        user = await db.scalar(select(User).where(User.username == "otto"))
        user.is_admin = True

    run_db(migrated_database_url, promote)
    otto_id = user_id(other)
    response = admin.put(
        f"/api/admin/users/{otto_id}/status", json={"status": "disabled"}
    )
    assert response.status_code == 204

    monkeypatch.setenv("M4P_DATABASE_URL", migrated_database_url)
    monkeypatch.setenv("M4P_SECRET_KEY", settings.secret_key.get_secret_value())
    assert main(["enable-user", "otto"]) == 0
    assert main(["set-email", "otto", "otto@example.org"]) == 0

    async def otto(db):
        return await db.scalar(select(User).where(User.username == "otto"))

    user = run_db(migrated_database_url, otto)
    assert (user.status, user.email) == ("active", "otto@example.org")
    assert user.email_verified_at is not None


def test_uploads_may_go_to_a_bucket_on_another_site():
    policy = content_security_policy(None, "https://s3.us-east-1.amazonaws.com")
    assert "connect-src 'self' https://s3.us-east-1.amazonaws.com" in policy
    assert "connect-src 'self';" in content_security_policy(None)
