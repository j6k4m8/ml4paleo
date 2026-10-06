"""
Limits on server-sent event streams. Each open stream is a coroutine that
asks the database for news every second, so an account may keep only a few
open at once in each API process (a browser uses one or two per open page).
"""

import uuid
import weakref
from collections import Counter
from collections.abc import AsyncIterator

from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTask

PER_USER = 16

_open: Counter[uuid.UUID] = Counter()


class Slot:
    """
    One account's place for one open stream, taken by `reserve`. `release`
    gives it back, once however often it's called.
    """

    def __init__(self, user_id: uuid.UUID):
        self.user_id = user_id
        self._held = True

    def release(self) -> None:
        if not self._held:
            return
        self._held = False
        _open[self.user_id] -= 1
        if _open[self.user_id] <= 0:
            del _open[self.user_id]


def reserve(user_id: uuid.UUID) -> Slot:
    """
    Take a place for a new stream, or refuse it (429) if `user_id` has as
    many open as allowed. Checking and counting happen together, with
    nothing awaited in between, so requests arriving at once can't all slip
    under the limit; call it right before making the response.
    """
    if _open[user_id] >= PER_USER:
        raise HTTPException(
            status_code=429,
            detail="Too many live updates are open; close some tabs and try again.",
        )
    _open[user_id] += 1
    return Slot(user_id)


def response(slot: Slot, events: AsyncIterator[str]) -> StreamingResponse:
    """
    A server-sent event response for `events` that gives `slot` back when
    the stream ends, when the client leaves (even before it started), or if
    the response is dropped unsent.
    """

    async def stream() -> AsyncIterator[str]:
        try:
            async for event in events:
                yield event
        finally:
            slot.release()

    body = stream()
    # A stream that never starts never runs its `finally`.
    weakref.finalize(body, slot.release)
    return StreamingResponse(
        body,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
        background=BackgroundTask(slot.release),
    )
