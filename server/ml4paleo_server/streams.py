"""
Limits on server-sent event streams. Each open stream is a coroutine that
asks the database for news every second, so an account may keep only a few
open at once in each API process (a browser uses one or two per open page).
"""

import uuid
from collections import Counter
from collections.abc import AsyncIterator

from fastapi import HTTPException

PER_USER = 16

_open: Counter[uuid.UUID] = Counter()


def check(user_id: uuid.UUID) -> None:
    """Refuse a new stream (429) if `user_id` has as many open as allowed."""
    if _open[user_id] >= PER_USER:
        raise HTTPException(
            status_code=429,
            detail="Too many live updates are open; close some tabs and try again.",
        )


async def counted(user_id: uuid.UUID, events: AsyncIterator[str]) -> AsyncIterator[str]:
    """
    Pass `events` on, counting the stream against `user_id` while it runs.
    It counts only once it starts, so a client gone before then costs nothing.
    """
    _open[user_id] += 1
    try:
        async for event in events:
            yield event
    finally:
        _open[user_id] -= 1
        if _open[user_id] <= 0:
            del _open[user_id]
