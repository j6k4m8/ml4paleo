"""
Admins see every account, with what it uses, and can disable one (it is
signed out at once) or enable it again.
"""

import uuid

import pytest
from helpers import NOT_JSON, PASSWORD, make_admin, outbox, run_db, signup, strict_json
from ml4paleo_server import jobs
from ml4paleo_server.app import content_security_policy
from ml4paleo_server.cli import main
from ml4paleo_server.db import Job, User
from ml4paleo_server.settings import QuotaSettings
from pydantic import ValidationError
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


@pytest.mark.parametrize("literal", NOT_JSON)
def test_a_limit_of_nan_or_infinity_is_a_422_not_a_500(
    new_browser, migrated_database_url, literal
):
    # An override is stored as JSON, which can't hold them (and infinity isn't
    # how to say unlimited: null is).
    admin, _ = make_admin(new_browser, migrated_database_url)
    ada = new_browser()
    signup(ada)
    ada.post("/api/me/quota-requests", json={"message": "More, please."})
    [pending] = admin.get("/api/admin/quota-requests").json()
    before = ada.get("/api/me/quota").json()
    quota = f"/api/admin/users/{user_id(ada)}/quota"
    json_type = {"content-type": "application/json"}
    for field in ("storage_gb", "cpu_hours_per_day", "gpu_hours_per_day"):
        refused = admin.put(
            quota, content=f'{{"{field}": {literal}}}', headers=json_type
        )
        assert refused.status_code == 422, (field, refused.text)
        [error] = strict_json(refused.text)["detail"]
        assert error["loc"] == ["body", field] and error["input"] == NOT_JSON[literal]
    grant = f'{{"decision": "grant", "quota_override": {{"storage_gb": {literal}}}}}'
    refused = admin.post(
        f"/api/admin/quota-requests/{pending['id']}", content=grant, headers=json_type
    )
    assert refused.status_code == 422, refused.text
    [error] = strict_json(refused.text)["detail"]
    assert error["loc"] == ["body", "quota_override", "storage_gb"]
    # Nothing changed, and the request is still open.
    assert ada.get("/api/me/quota").json() == before
    assert [r["id"] for r in admin.get("/api/admin/quota-requests").json()] == [
        pending["id"]
    ]


def test_a_deploys_limits_cannot_be_infinity_either():
    for field in ("storage_gb", "cpu_hours_per_day", "gpu_hours_per_day"):
        with pytest.raises(ValidationError):
            QuotaSettings(**{field: float("inf")})
    assert QuotaSettings(storage_gb=None).storage_gb is None


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
