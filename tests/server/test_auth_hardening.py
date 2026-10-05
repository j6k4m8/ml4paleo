"""
Hardening found in review: rate limits per account, two-factor replay and
key rotation, enforced email verification, single-use reset links, admin
bootstrap edge cases, and outbox cleanup.
"""

import datetime

import pytest
from helpers import (
    ADMIN_PASSWORD,
    PASSWORD,
    link_token,
    make_admin,
    next_code,
    outbox,
    run_db,
    signup,
)
from ml4paleo_server import email, housekeeper
from ml4paleo_server.auth import ensure_admin
from ml4paleo_server.db import AuthToken, EmailOutbox, RateLimit, User, UserSession
from pydantic import SecretStr
from sqlalchemy import func, select, text, update


def test_username_and_email_share_one_login_limit(new_browser):
    signup(new_browser(), email="ada@example.org")
    browser = new_browser()
    names = ["ada", "ada@example.org"] * 3
    statuses = [
        browser.post(
            "/api/auth/login", json={"username": name, "password": "wrong"}
        ).status_code
        for name in names
    ]
    assert statuses == [401] * 5 + [429]


def test_reserved_usernames_are_refused(new_browser):
    for name in ["admin", "Root", "ml4paleo"]:
        assert signup(new_browser(), username=name).status_code == 422


def test_without_smtp_emails_stay_unverified(new_browser):
    response = signup(new_browser(), email="ada@example.org")
    assert response.status_code == 201
    assert response.json()["user"]["email_verified"] is False
    assert response.json()["user"]["status"] == "active"


@pytest.fixture
def smtp_settings(settings):
    return settings.model_copy(
        update={"smtp": settings.smtp.model_copy(update={"host": "smtp.test"})}
    )


def test_unverified_accounts_can_only_finish_setup(
    new_browser, smtp_settings, migrated_database_url
):
    browser = new_browser(smtp_settings)
    signup(browser, email="ada@example.org")
    # Signed-in routes refuse the account until its email is verified...
    refused = browser.post("/api/admin/invites", json={})
    assert refused.json()["detail"] == "email_verification_required"
    # ...but it can ask for another verification email.
    assert browser.post("/api/auth/verify-email/resend").status_code == 202
    assert len(outbox(migrated_database_url)) == 2


def test_changing_a_password_voids_outstanding_reset_links(
    new_browser, smtp_settings, migrated_database_url
):
    browser = new_browser(smtp_settings)
    signup(browser, email="ada@example.org")
    browser.post(
        "/api/auth/verify-email",
        json={"token": link_token(outbox(migrated_database_url)[0].body)},
    )
    new_browser(smtp_settings).post(
        "/api/auth/password-reset/request", json={"email": "ada@example.org"}
    )
    reset_token = link_token(outbox(migrated_database_url)[-1].body)
    browser.post(
        "/api/auth/password",
        json={"current_password": PASSWORD, "new_password": "a brand new passphrase"},
    )
    stale = new_browser(smtp_settings).post(
        "/api/auth/password-reset/confirm",
        json={"token": reset_token, "new_password": "attacker chosen phrase"},
    )
    assert stale.status_code == 400


def test_two_factor_codes_work_once_and_failures_are_limited(
    new_browser, migrated_database_url
):
    _, secret = make_admin(new_browser, migrated_database_url)
    credentials = {"username": "admin", "password": ADMIN_PASSWORD}
    code = next_code(secret)
    first = new_browser().post(
        "/api/auth/login", json={**credentials, "totp_code": code}
    )
    assert first.status_code == 200
    replay = new_browser().post(
        "/api/auth/login", json={**credentials, "totp_code": code}
    )
    assert replay.status_code == 401
    browser = new_browser()
    statuses = [
        browser.post(
            "/api/auth/login", json={**credentials, "totp_code": "000000"}
        ).status_code
        for _ in range(5)
    ]
    assert statuses[-1] == 429


def test_a_new_secret_key_gives_a_clear_error_and_can_be_reset(
    new_browser, settings, migrated_database_url, monkeypatch
):
    _, secret = make_admin(new_browser, migrated_database_url)
    rotated = settings.model_copy(
        update={"secret_key": SecretStr("a-completely-different-secret-key-0123456789")}
    )
    response = new_browser(rotated).post(
        "/api/auth/login",
        json={
            "username": "admin",
            "password": ADMIN_PASSWORD,
            "totp_code": next_code(secret),
        },
    )
    assert response.status_code == 401
    assert "administrator" in response.json()["detail"]

    from ml4paleo_server import cli

    monkeypatch.setenv("M4P_DATABASE_URL", migrated_database_url)
    assert cli.main(["reset-two-factor", "admin"]) == 0
    again = new_browser(rotated).post(
        "/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD}
    )
    assert again.json()["required_steps"] == ["set_up_two_factor"]


def test_bootstrap_takes_over_a_squatted_admin_name(new_browser, migrated_database_url):
    squatter = new_browser()

    async def squat(db):
        db.add(User(username="admin", password_hash=None))

    run_db(migrated_database_url, squat)

    async def plant_session(db):
        user = await db.scalar(select(User).where(User.username == "admin"))
        db.add(
            UserSession(
                token_hash="0" * 64,
                user_id=user.id,
                expires_at=datetime.datetime.now(datetime.UTC)
                + datetime.timedelta(days=1),
            )
        )
        await db.execute(
            update(User).where(User.id == user.id).values(totp_secret_enc="garbage")
        )

    run_db(migrated_database_url, plant_session)

    async def bootstrap(db):
        return await ensure_admin(db, "operator chosen passphrase")

    assert run_db(migrated_database_url, bootstrap) is None

    async def inspect(db):
        user = await db.scalar(select(User).where(User.username == "admin"))
        sessions = await db.scalar(
            select(func.count())
            .select_from(UserSession)
            .where(UserSession.user_id == user.id)
        )
        return user.is_admin, user.totp_secret_enc, sessions

    assert run_db(migrated_database_url, inspect) == (True, None, 0)
    login = squatter.post(
        "/api/auth/login",
        json={"username": "admin", "password": "operator chosen passphrase"},
    )
    assert login.json()["required_steps"] == ["change_password", "set_up_two_factor"]


def test_non_ascii_csrf_headers_are_refused_not_crashing(new_browser):
    browser = new_browser()
    signup(browser)
    response = browser.client.post(
        "/api/auth/totp/setup", headers={"X-CSRF-Token": "tökén".encode("latin-1")}
    )
    assert response.status_code == 403


def test_housekeeper_prunes_expired_rows(new_browser, migrated_database_url):
    signup(new_browser())
    past = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=10)

    async def age_everything(db):
        await db.execute(update(UserSession).values(expires_at=past))
        db.add(AuthToken(token_hash="1" * 64, kind="reset", expires_at=past))
        await db.execute(update(RateLimit).values(window_start=past))

    run_db(migrated_database_url, age_everything)

    async def prune_and_count(db):
        from ml4paleo_server.db import create_sessionmaker

        await housekeeper.prune(create_sessionmaker(db.bind))
        counts = []
        for table in (UserSession, AuthToken, RateLimit):
            counts.append(await db.scalar(select(func.count()).select_from(table)))
        return counts

    assert run_db(migrated_database_url, prune_and_count) == [0, 0, 0]


def test_concurrent_sign_ins_cannot_share_a_code(new_browser, migrated_database_url):
    import threading

    _, secret = make_admin(new_browser, migrated_database_url)
    code = next_code(secret)
    browsers = [new_browser(address=f"10.0.0.{i}") for i in range(4)]
    barrier = threading.Barrier(len(browsers))
    statuses = []

    def sign_in(browser):
        barrier.wait()
        response = browser.post(
            "/api/auth/login",
            json={"username": "admin", "password": ADMIN_PASSWORD, "totp_code": code},
        )
        statuses.append(response.status_code)

    threads = [threading.Thread(target=sign_in, args=(b,)) for b in browsers]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(statuses) == [200, 401, 401, 401]


def test_takeover_also_removes_the_squatters_email_and_links(migrated_database_url):
    async def squat(db):
        user = User(
            username="admin",
            email="squatter@example.org",
            email_verified_at=datetime.datetime.now(datetime.UTC),
        )
        db.add(user)
        await db.flush()
        db.add(
            AuthToken(
                token_hash="2" * 64,
                kind="reset",
                user_id=user.id,
                email=user.email,
                expires_at=datetime.datetime.now(datetime.UTC)
                + datetime.timedelta(hours=1),
            )
        )

    run_db(migrated_database_url, squat)

    async def bootstrap_and_inspect(db):
        await ensure_admin(db, "operator chosen passphrase")
        user = await db.scalar(select(User).where(User.username == "admin"))
        tokens = await db.scalar(select(func.count()).select_from(AuthToken))
        return user.email, user.email_verified_at, tokens

    assert run_db(migrated_database_url, bootstrap_and_inspect) == (None, None, 0)


def test_an_attacker_elsewhere_cannot_lock_the_owner_out(new_browser):
    signup(new_browser())
    attacker = new_browser(address="203.0.113.7")
    for _ in range(6):
        attacker.post("/api/auth/login", json={"username": "ada", "password": "wrong"})
    owner = new_browser(address="198.51.100.4")
    login = owner.post(
        "/api/auth/login", json={"username": "ada", "password": PASSWORD}
    )
    assert login.status_code == 200


def test_ipv4_mapped_addresses_are_counted_as_ipv4():
    from types import SimpleNamespace

    from ml4paleo_server.auth.ratelimit import client_key

    def request(host):
        return SimpleNamespace(client=SimpleNamespace(host=host))

    assert client_key(request("::ffff:203.0.113.7")) == "203.0.113.7"
    assert client_key(request("::ffff:198.51.100.4")) == "198.51.100.4"
    assert client_key(request("2001:db8::1")) == client_key(request("2001:db8::2"))


def test_confirming_two_factor_after_a_key_change_asks_to_start_over(
    new_browser, settings
):
    browser = new_browser()
    signup(browser)
    browser.post("/api/auth/totp/setup")
    rotated = settings.model_copy(
        update={"secret_key": SecretStr("a-completely-different-secret-key-0123456789")}
    )
    other = new_browser(rotated)
    other.post("/api/auth/login", json={"username": "ada", "password": PASSWORD})
    response = other.post("/api/auth/totp/confirm", json={"code": "123456"})
    assert response.status_code == 409


def test_the_reset_password_command_voids_reset_links(
    new_browser, smtp_settings, migrated_database_url, monkeypatch
):
    browser = new_browser(smtp_settings)
    signup(browser, email="ada@example.org")
    browser.post(
        "/api/auth/verify-email",
        json={"token": link_token(outbox(migrated_database_url)[0].body)},
    )
    new_browser(smtp_settings).post(
        "/api/auth/password-reset/request", json={"email": "ada@example.org"}
    )
    token = link_token(outbox(migrated_database_url)[-1].body)

    from ml4paleo_server import cli

    monkeypatch.setenv("M4P_DATABASE_URL", migrated_database_url)
    assert cli.main(["reset-password", "ada"]) == 0
    stale = new_browser(smtp_settings).post(
        "/api/auth/password-reset/confirm",
        json={"token": token, "new_password": "attacker chosen phrase"},
    )
    assert stale.status_code == 400


@pytest.mark.parametrize(
    ("name", "status"),
    [("admin1", 422), ("root_", 422), ("ml4paleo-2", 422), ("rootbeer", 201)],
)
def test_reserved_name_look_alikes_are_refused(new_browser, name, status):
    assert signup(new_browser(), username=name).status_code == status


def test_queued_links_are_not_readable_in_the_database(
    new_browser, smtp_settings, migrated_database_url
):
    signup(new_browser(smtp_settings), email="ada@example.org")
    [message] = outbox(migrated_database_url)
    token = link_token(message.body)
    assert token not in message.body_sealed
    assert "token=" not in message.body_sealed

    async def raw_rows(db):
        result = await db.execute(text("SELECT * FROM email_outbox"))
        return [str(row) for row in result]

    assert all(token not in row for row in run_db(migrated_database_url, raw_rows))


def test_mail_sealed_with_an_old_key_fails_without_sending(
    new_browser, smtp_settings, migrated_database_url
):
    signup(new_browser(smtp_settings), email="ada@example.org")
    rotated = smtp_settings.model_copy(
        update={"secret_key": SecretStr("a-completely-different-secret-key-0123456789")}
    )
    sent = []

    async def send(db):
        from ml4paleo_server.db import create_sessionmaker

        return await email.send_pending(
            create_sessionmaker(db.bind),
            rotated,
            send=lambda smtp, message: sent.append(message),
        )

    assert run_db(migrated_database_url, send) == 0
    assert sent == []

    async def statuses(db):
        return (await db.scalars(select(EmailOutbox.status))).all()

    assert run_db(migrated_database_url, statuses) == ["failed"]
