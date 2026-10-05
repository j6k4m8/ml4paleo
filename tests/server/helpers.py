"""
Helpers shared by the server tests.
"""

import asyncio
import re
import time
from types import SimpleNamespace

import pyotp
from ml4paleo_server import email, sealing
from ml4paleo_server.auth import ensure_admin
from ml4paleo_server.db import EmailOutbox, create_engine, create_sessionmaker
from sqlalchemy import select

PASSWORD = "correct horse battery staple"
ADMIN_PASSWORD = "fossil dig site 1923"
SECRET_KEY = "test-secret-key-that-is-long-enough-0123456789"


def run_db(database_url, fn):
    """
    Run `await fn(db)` against the test database and return the result.
    """

    async def runner():
        engine = create_engine(database_url)
        try:
            async with create_sessionmaker(engine)() as db:
                result = await fn(db)
                await db.commit()
                return result
        finally:
            await engine.dispose()

    return asyncio.run(runner())


def signup(browser, username="ada", password=PASSWORD, **extra):
    return browser.post(
        "/api/auth/signup", json={"username": username, "password": password, **extra}
    )


def outbox(database_url):
    """
    Return the queued emails, with their sealed bodies opened as `body`.
    """

    async def fetch(db):
        return (
            await db.scalars(select(EmailOutbox).order_by(EmailOutbox.created_at))
        ).all()

    return [
        SimpleNamespace(
            to_address=row.to_address,
            subject=row.subject,
            status=row.status,
            attempts=row.attempts,
            body_sealed=row.body_sealed,
            body=sealing.unseal(
                SECRET_KEY, email.SEAL_PURPOSE, row.body_sealed, row.to_address
            ),
        )
        for row in run_db(database_url, fetch)
    ]


def link_token(body: str) -> str:
    return re.search(r"token=([A-Za-z0-9_-]+)", body).group(1)


def make_admin(new_browser, database_url):
    """
    Bootstrap the admin account and finish its required setup. Returns the
    signed-in browser and the TOTP secret.
    """
    password = run_db(database_url, ensure_admin)
    admin = new_browser()
    admin.post("/api/auth/login", json={"username": "admin", "password": password})
    admin.post(
        "/api/auth/password",
        json={"current_password": password, "new_password": ADMIN_PASSWORD},
    )
    secret = admin.post("/api/auth/totp/setup").json()["secret"]
    admin.post("/api/auth/totp/confirm", json={"code": pyotp.TOTP(secret).now()})
    return admin, secret


def next_code(secret: str) -> str:
    """
    The TOTP code for the next time step. Each code works only once, so tests
    that sign in right after confirming setup need the next one.
    """
    return pyotp.TOTP(secret).at(int(time.time()) + 30)
