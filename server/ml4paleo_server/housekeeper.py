"""
The housekeeper: one background process for periodic upkeep.

For now it sends queued email. Later build steps add job lease reaping and
storage garbage collection here.
"""

import asyncio
import logging

from .db import create_engine, create_sessionmaker
from .email import send_pending
from .settings import Settings

log = logging.getLogger(__name__)

INTERVAL_SECONDS = 10


async def run_once(sessionmaker, settings: Settings) -> None:
    sent = await send_pending(sessionmaker, settings.smtp)
    if sent:
        log.info("Sent %d queued emails", sent)


async def run_forever(settings: Settings | None = None) -> None:
    logging.basicConfig(level=logging.INFO)
    settings = settings or Settings()
    engine = create_engine(settings.database_url.get_secret_value())
    sessionmaker = create_sessionmaker(engine)
    try:
        while True:
            try:
                await run_once(sessionmaker, settings)
            except Exception:
                log.exception("Housekeeping pass failed")
            await asyncio.sleep(INTERVAL_SECONDS)
    finally:
        await engine.dispose()
