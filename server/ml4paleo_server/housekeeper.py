"""
The housekeeper: one background process for periodic upkeep.

It takes jobs back from workers that stopped responding, deletes the files
of artifacts that are no longer needed (`artifacts.collect_garbage`), sends
queued email, and deletes expired sessions, used or expired tokens, stale
rate-limit counters, and mail that failed for good.
"""

import asyncio
import datetime
import logging
import signal

from sqlalchemy import delete, or_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from .artifacts import collect_garbage
from .db import (
    AuthToken,
    EmailOutbox,
    RateLimit,
    UserSession,
    create_engine,
    create_sessionmaker,
)
from .email import send_pending
from .jobs import reap_one, sweep
from .settings import Settings

log = logging.getLogger(__name__)

INTERVAL_SECONDS = 10


# Keep used and expired tokens and failed mail around briefly for debugging.
KEEP_USED_TOKENS = datetime.timedelta(days=1)
KEEP_FAILED_MAIL = datetime.timedelta(days=7)
KEEP_RATE_LIMITS = datetime.timedelta(days=1)


async def prune(sessionmaker: async_sessionmaker[AsyncSession]) -> None:
    now = datetime.datetime.now(datetime.UTC)
    async with sessionmaker() as db:
        await db.execute(delete(UserSession).where(UserSession.expires_at < now))
        await db.execute(
            delete(AuthToken).where(
                or_(
                    AuthToken.expires_at < now - KEEP_USED_TOKENS,
                    AuthToken.used_at < now - KEEP_USED_TOKENS,
                )
            )
        )
        await db.execute(
            delete(RateLimit).where(RateLimit.window_start < now - KEEP_RATE_LIMITS)
        )
        await db.execute(
            delete(EmailOutbox).where(
                EmailOutbox.status == "failed",
                EmailOutbox.created_at < now - KEEP_FAILED_MAIL,
            )
        )
        await db.commit()


async def reap_jobs(sessionmaker: async_sessionmaker[AsyncSession]) -> int:
    """
    Take back expired leases, one job per transaction, then run the queue's
    safety nets. Returns how many leases expired.
    """
    expired = 0
    while True:
        async with sessionmaker() as db:
            if not await reap_one(db):
                break
            await db.commit()
        expired += 1
    async with sessionmaker() as db:
        await sweep(db)
        await db.commit()
    return expired


async def run_once(
    sessionmaker: async_sessionmaker[AsyncSession], settings: Settings
) -> None:
    """
    One pass of every task. A task that fails is logged and doesn't stop the
    others.
    """

    async def reap() -> None:
        if expired := await reap_jobs(sessionmaker):
            log.info("Took back %d jobs from workers that stopped responding", expired)

    async def collect() -> None:
        if collected := await collect_garbage(sessionmaker, settings):
            log.info("Deleted %d artifacts that are no longer needed", collected)

    async def send() -> None:
        if sent := await send_pending(sessionmaker, settings):
            log.info("Sent %d queued emails", sent)

    async def tidy() -> None:
        await prune(sessionmaker)

    for task in (reap, collect, send, tidy):
        try:
            await task()
        except Exception:
            log.exception("Housekeeping task %s failed", task.__name__)


async def run_forever(settings: Settings | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    settings = settings or Settings()
    engine = create_engine(settings.database_url.get_secret_value())
    sessionmaker = create_sessionmaker(engine)
    # Stop promptly (between passes) when Docker asks the container to stop.
    stopping = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signum, stopping.set)
    try:
        while not stopping.is_set():
            try:
                await run_once(sessionmaker, settings)
            except Exception:
                log.exception("Housekeeping pass failed")
            try:
                await asyncio.wait_for(stopping.wait(), INTERVAL_SECONDS)
            except TimeoutError:
                pass
    finally:
        await engine.dispose()
