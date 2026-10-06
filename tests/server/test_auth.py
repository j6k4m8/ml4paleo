"""
Accounts and sign-in: signup, login, sessions, CSRF, password changes and
resets, email addresses and their confirmation (with starter limits until
then), admin bootstrap with two-factor, and invites.
"""

import datetime
import re

import pyotp
import pytest
from fastapi import HTTPException
from helpers import (
    PASSWORD,
    link_token,
    make_admin,
    next_code,
    outbox,
    run_db,
    signup,
)
from ml4paleo_server import email as email_module
from ml4paleo_server import quotas
from ml4paleo_server.app import create_app
from ml4paleo_server.auth import ensure_admin
from ml4paleo_server.db import (
    EmailOutbox,
    User,
    create_sessionmaker,
)
from pydantic import SecretStr
from sqlalchemy import update


def test_signup_signs_in_and_normalizes_the_username(new_browser):
    browser = new_browser()
    response = signup(browser, username="Ada.Lovelace")
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["user"]["username"] == "ada.lovelace"
    assert body["required_steps"] == []
    session = browser.get("/api/auth/session").json()
    assert session["user"]["username"] == "ada.lovelace"
    assert session["csrf_token"] == body["csrf_token"]


@pytest.mark.parametrize(
    ("username", "password", "status"),
    [
        ("ada", "short", 422),
        ("ada", "Unbelievable", 422),  # on the common-password list
        ("ada", "aaaaaaaaaaaaaaaa", 422),  # too few distinct characters
        ("ada", "my name is ada lovelace", 422),  # contains the username
        ("a", PASSWORD, 422),
        ("../etc", PASSWORD, 422),
    ],
)
def test_signup_rejects_bad_input(new_browser, username, password, status):
    assert (
        signup(new_browser(), username=username, password=password).status_code
        == status
    )


def test_usernames_are_unique(new_browser):
    assert signup(new_browser()).status_code == 201
    assert signup(new_browser(), username="ADA").status_code == 409


def test_login_logout_and_generic_errors(new_browser, migrated_database_url):
    signup(new_browser())
    browser = new_browser()
    wrong = browser.post(
        "/api/auth/login", json={"username": "ada", "password": "nope"}
    )
    unknown = browser.post(
        "/api/auth/login", json={"username": "bob", "password": "nope"}
    )
    assert wrong.status_code == unknown.status_code == 401
    assert wrong.json() == unknown.json()

    login = browser.post(
        "/api/auth/login", json={"username": "ADA", "password": PASSWORD}
    )
    assert login.status_code == 200
    assert browser.get("/api/auth/session").status_code == 200
    assert browser.post("/api/auth/logout").status_code == 204
    assert browser.get("/api/auth/session").status_code == 401

    # Disabled accounts can't sign in.
    async def disable(db):
        await db.execute(update(User).values(status="disabled"))

    run_db(migrated_database_url, disable)
    blocked = browser.post(
        "/api/auth/login", json={"username": "ada", "password": PASSWORD}
    )
    assert blocked.status_code == 401


def test_repeated_failed_logins_are_rate_limited(new_browser):
    browser = new_browser()
    statuses = [
        browser.post(
            "/api/auth/login", json={"username": "ada", "password": "wrong"}
        ).status_code
        for _ in range(6)
    ]
    assert statuses == [401] * 5 + [429]


def test_csrf_and_origin_checks(new_browser, settings):
    browser = new_browser()
    signup(browser)
    # A signed-in request without the CSRF token is refused.
    forged = browser.client.post("/api/auth/totp/setup")
    assert forged.status_code == 403
    assert browser.post("/api/auth/totp/setup").status_code == 200
    # Requests from another site are refused, even to log in.
    for headers in [
        {"Origin": "https://evil.example"},
        {"Sec-Fetch-Site": "cross-site"},
    ]:
        response = new_browser().post(
            "/api/auth/login",
            json={"username": "ada", "password": PASSWORD},
            headers=headers,
        )
        assert response.status_code == 403
    same_site = new_browser().post(
        "/api/auth/login",
        json={"username": "ada", "password": PASSWORD},
        headers={"Origin": settings.public_url},
    )
    assert same_site.status_code == 200


def test_changing_a_password_signs_out_other_sessions(new_browser):
    first = new_browser()
    signup(first)
    second = new_browser()
    second.post("/api/auth/login", json={"username": "ada", "password": PASSWORD})
    wrong = first.post(
        "/api/auth/password",
        json={"current_password": "nope", "new_password": "a brand new passphrase"},
    )
    assert wrong.status_code == 403
    changed = first.post(
        "/api/auth/password",
        json={"current_password": PASSWORD, "new_password": "a brand new passphrase"},
    )
    assert changed.status_code == 204
    assert first.get("/api/auth/session").status_code == 200
    assert second.get("/api/auth/session").status_code == 401


def test_bootstrap_admin_must_change_password_and_set_up_two_factor(
    new_browser, migrated_database_url
):
    password = run_db(migrated_database_url, ensure_admin)
    assert password is not None
    assert run_db(migrated_database_url, ensure_admin) is None  # only once

    admin = new_browser()
    login = admin.post(
        "/api/auth/login", json={"username": "admin", "password": password}
    )
    assert login.json()["required_steps"] == ["change_password", "set_up_two_factor"]
    assert (
        admin.get("/api/admin/settings").json()["detail"] == "password_change_required"
    )

    admin.post(
        "/api/auth/password",
        json={"current_password": password, "new_password": "fossil dig site 1923"},
    )
    assert admin.get("/api/admin/settings").json()["detail"] == "two_factor_required"

    secret = admin.post("/api/auth/totp/setup").json()["secret"]
    bad = admin.post("/api/auth/totp/confirm", json={"code": "000000"})
    assert bad.status_code == 422
    good = admin.post("/api/auth/totp/confirm", json={"code": pyotp.TOTP(secret).now()})
    assert good.status_code == 204
    assert admin.get("/api/admin/settings").status_code == 200

    # Signing in now needs the second factor.
    fresh = new_browser()
    credentials = {"username": "admin", "password": "fossil dig site 1923"}
    needs_code = fresh.post("/api/auth/login", json=credentials)
    assert needs_code.json()["detail"] == "totp_required"
    # The code used to confirm setup can't be used again, so use the next one.
    reused = fresh.post(
        "/api/auth/login", json={**credentials, "totp_code": pyotp.TOTP(secret).now()}
    )
    assert reused.json()["detail"] == "Wrong two-factor code."
    with_code = fresh.post(
        "/api/auth/login", json={**credentials, "totp_code": next_code(secret)}
    )
    assert with_code.status_code == 200
    assert with_code.json()["required_steps"] == []


def test_invite_only_signup(new_browser, migrated_database_url):
    admin, _ = make_admin(new_browser, migrated_database_url)
    assert (
        admin.put("/api/admin/settings", json={"signup_mode": "invite"}).status_code
        == 200
    )
    assert signup(new_browser()).status_code == 403

    invite_url = admin.post("/api/admin/invites", json={}).json()["url"]
    invite = invite_url.split("invite=")[1]
    assert signup(new_browser(), invite=invite).status_code == 201
    reused = signup(new_browser(), username="bob", invite=invite)
    assert reused.status_code == 400


def test_non_admins_cannot_use_admin_routes(new_browser):
    browser = new_browser()
    signup(browser)
    assert browser.get("/api/admin/settings").status_code == 403
    assert browser.post("/api/admin/invites", json={}).status_code == 403


@pytest.fixture
def smtp_settings(settings):
    return settings.model_copy(
        update={"smtp": settings.smtp.model_copy(update={"host": "smtp.test"})}
    )


def test_email_verification_flow(new_browser, smtp_settings, migrated_database_url):
    browser = new_browser(smtp_settings)
    created = signup(browser, email="Ada@Example.org")
    assert created.status_code == 201
    assert created.json()["user"]["email"] == "ada@example.org"
    assert created.json()["user"]["email_verified"] is False

    [message] = outbox(migrated_database_url)
    assert message.to_address == "ada@example.org"
    token = link_token(message.body)
    assert (
        browser.post("/api/auth/verify-email", json={"token": token}).status_code == 204
    )
    session = browser.get("/api/auth/session").json()
    assert session["user"]["email_verified"]
    again = browser.post("/api/auth/verify-email", json={"token": token})
    assert again.status_code == 400


GB = 1024**3


def requiring_email(settings):
    return settings.model_copy(
        update={"auth": settings.auth.model_copy(update={"require_email": True})}
    )


def test_unconfirmed_accounts_get_starter_limits(
    new_browser, smtp_settings, migrated_database_url
):
    browser = new_browser(requiring_email(smtp_settings))
    assert new_browser().get("/api/auth/config").json()["require_email"] is False
    assert browser.get("/api/auth/config").json()["require_email"] is True
    missing = signup(browser)
    assert (missing.status_code, missing.json()["detail"]) == (
        422,
        "Enter an email address.",
    )
    created = signup(browser, email="ada@example.org")
    assert created.status_code == 201
    assert created.json()["starter_limits"] is True
    assert created.json()["required_steps"] == []
    quota = browser.get("/api/me/quota").json()
    assert (quota["storage_bytes_limit"], quota["trained_models_limit"]) == (GB, 1)
    # Meanwhile they can use the site, and ask for the link again.
    assert browser.post("/api/projects", json={"name": "Trial"}).status_code == 201
    assert browser.post("/api/auth/verify-email/resend").status_code == 202
    _, again = outbox(migrated_database_url)
    assert again.to_address == "ada@example.org"

    token = link_token(again.body)
    assert (
        browser.post("/api/auth/verify-email", json={"token": token}).status_code == 204
    )
    assert browser.get("/api/auth/session").json()["starter_limits"] is False
    quota = browser.get("/api/me/quota").json()
    assert (quota["storage_bytes_limit"], quota["trained_models_limit"]) == (
        10 * GB,
        20,
    )
    # Nothing more to confirm.
    assert browser.post("/api/auth/verify-email/resend").status_code == 202
    assert len(outbox(migrated_database_url)) == 2


def test_starter_limits_hold_reservations(settings, migrated_database_url):
    required = requiring_email(settings)

    async def reserve(db):
        user = User(username="ada", email="ada@example.org")
        db.add(user)
        await db.flush()
        with pytest.raises(HTTPException):
            await quotas.reserve_storage(db, required, user, 2 * GB)
        user.email_verified_at = datetime.datetime.now(datetime.UTC)
        await quotas.reserve_storage(db, required, user, 2 * GB)
        return (await quotas.usage_for(db, user.id)).storage_bytes

    assert run_db(migrated_database_url, reserve) == 2 * GB


def test_admins_set_the_requirement_and_confirm_addresses(
    new_browser, settings, migrated_database_url
):
    required = requiring_email(settings)
    admin, _ = make_admin(lambda: new_browser(required), migrated_database_url)
    # Admins never have starter limits.
    assert admin.get("/api/auth/session").json()["starter_limits"] is False
    shown = admin.get("/api/admin/settings").json()
    assert shown["require_email"] is True
    assert shown["unconfirmed_quota"]["storage_gb"] == 1
    ada = new_browser(required)
    signup(ada, email="ada@example.org")
    ada_id = ada.get("/api/auth/session").json()["user"]["id"]
    [listed] = admin.get("/api/admin/users?q=ada").json()
    assert (listed["email_confirmed"], listed["starter_limits"]) == (False, True)
    assert listed["storage_bytes_limit"] == GB

    # A limit an admin sets still applies.
    admin.put(f"/api/admin/users/{ada_id}/quota", json={"trained_models": 5})
    quota = ada.get("/api/me/quota").json()
    assert (quota["storage_bytes_limit"], quota["trained_models_limit"]) == (GB, 5)
    confirmed = admin.post(f"/api/admin/users/{ada_id}/confirm-email")
    assert confirmed.status_code == 204
    quota = ada.get("/api/me/quota").json()
    assert (quota["storage_bytes_limit"], quota["trained_models_limit"]) == (
        10 * GB,
        5,
    )

    changed = admin.put("/api/admin/settings", json={"require_email": False}).json()
    assert (changed["signup_mode"], changed["require_email"]) == ("open", False)
    bob = new_browser(required)
    assert signup(bob, username="bob").status_code == 201
    assert bob.get("/api/auth/session").json()["starter_limits"] is False
    assert bob.get("/api/me/quota").json()["storage_bytes_limit"] == 10 * GB
    bob_id = bob.get("/api/auth/session").json()["user"]["id"]
    no_address = admin.post(f"/api/admin/users/{bob_id}/confirm-email")
    assert no_address.status_code == 409
    assert ada.post(f"/api/admin/users/{bob_id}/confirm-email").status_code == 403


def test_password_reset_flow(new_browser, smtp_settings, migrated_database_url):
    browser = new_browser(smtp_settings)
    signup(browser, email="ada@example.org")
    browser.post(
        "/api/auth/verify-email",
        json={"token": link_token(outbox(migrated_database_url)[0].body)},
    )
    anonymous = new_browser(smtp_settings)
    for address in ["ada@example.org", "nobody@example.org"]:
        response = anonymous.post(
            "/api/auth/password-reset/request", json={"email": address}
        )
        assert response.status_code == 202
    messages = outbox(migrated_database_url)
    assert [m.to_address for m in messages] == ["ada@example.org"] * 2
    token = link_token(messages[-1].body)

    weak = anonymous.post(
        "/api/auth/password-reset/confirm",
        json={"token": token, "new_password": "short"},
    )
    assert weak.status_code == 422
    reset = anonymous.post(
        "/api/auth/password-reset/confirm",
        json={"token": token, "new_password": "a brand new passphrase"},
    )
    # The weak attempt did not use up the token.
    assert reset.status_code == 204
    assert browser.get("/api/auth/session").status_code == 401  # signed out
    login = anonymous.post(
        "/api/auth/login",
        json={"username": "ada", "password": "a brand new passphrase"},
    )
    assert login.status_code == 200


def test_queued_email_is_sent_and_retried(smtp_settings, migrated_database_url):
    async def queue(db):
        email_module.queue_email(db, smtp_settings, "a@example.org", "Hello", "Body")

    run_db(migrated_database_url, queue)
    sent_messages, failures = [], []

    def flaky_send(smtp, message):
        if not failures:
            failures.append(message)
            raise OSError("mail server is down")
        sent_messages.append(message)

    async def send(db):
        return await email_module.send_pending(
            create_sessionmaker(db.bind), smtp_settings, send=flaky_send
        )

    assert run_db(migrated_database_url, send) == 0
    [message] = outbox(migrated_database_url)
    assert (message.status, message.attempts) == ("queued", 1)
    # The retry waits (backoff) instead of hammering a failing mail server...
    assert run_db(migrated_database_url, send) == 0

    async def make_due(db):
        await db.execute(
            update(EmailOutbox).values(
                next_attempt_at=datetime.datetime(2000, 1, 1, tzinfo=datetime.UTC)
            )
        )

    run_db(migrated_database_url, make_due)
    # ...and goes out once it is due.
    assert run_db(migrated_database_url, send) == 1
    # Sent mail is deleted, so its single-use links don't linger.
    assert outbox(migrated_database_url) == []
    assert sent_messages[0]["Subject"] == "Hello"


PUBLIC_ROUTES = {
    "/api/health",
    "/api/auth/config",
    "/api/auth/signup",
    "/api/auth/login",
    "/api/auth/logout",
    "/api/auth/verify-email",
    "/api/auth/password-reset/request",
    "/api/auth/password-reset/confirm",
}


def test_every_other_api_route_requires_sign_in(app, new_browser):
    browser = new_browser()
    checked = 0
    for path, operations in app.openapi()["paths"].items():
        if path in PUBLIC_ROUTES:
            continue
        url = re.sub(r"{[^}]+}", "00000000-0000-0000-0000-000000000000", path)
        for method in operations:
            response = browser.request(method.upper(), url, json={})
            assert response.status_code == 401, (method, path)
            checked += 1
    assert checked >= 6


def test_short_secret_keys_are_refused(settings):
    with pytest.raises(RuntimeError):
        create_app(settings.model_copy(update={"secret_key": SecretStr("too-short")}))


def test_signing_out_clears_the_browser_cache(new_browser):
    browser = new_browser()
    signup(browser)
    response = browser.post("/api/auth/logout")
    assert response.headers["clear-site-data"] == '"cache"'
