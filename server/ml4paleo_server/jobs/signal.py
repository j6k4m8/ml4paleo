"""
Wake waiting claims when jobs become claimable.

Each API process keeps one Postgres connection that LISTENs on the job
channel. The queue sends a notification whenever a job is queued, and every
claim waiting in that process wakes up and tries again. Claims also retry on
a timer, so a missed notification (for example while the listener
reconnects) only delays a job by a few seconds.
"""

import asyncio
import logging

import psycopg
from sqlalchemy.engine import make_url

from .queue import NOTIFY_CHANNEL

log = logging.getLogger(__name__)

RECONNECT_SECONDS = 5


class JobSignal:
    def __init__(self, database_url: str):
        url = make_url(database_url).set(drivername="postgresql")
        self._dsn = url.render_as_string(hide_password=False)
        self._event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self._listen())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def wait(self, timeout: float) -> None:
        """
        Return after the next notification, or after `timeout` seconds.
        """
        event = self._event
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except TimeoutError:
            pass

    def wake(self) -> None:
        # Swap in a fresh event, so each waiter wakes once per notification.
        event, self._event = self._event, asyncio.Event()
        event.set()

    async def _listen(self) -> None:
        while True:
            try:
                async with await psycopg.AsyncConnection.connect(
                    self._dsn, autocommit=True
                ) as connection:
                    await connection.execute(f"LISTEN {NOTIFY_CHANNEL}")
                    # Jobs may have been queued while we were not listening.
                    self.wake()
                    async for _ in connection.notifies():
                        self.wake()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any failure
                log.warning("Job listener lost its connection: %s", exc)
            await asyncio.sleep(RECONNECT_SECONDS)
