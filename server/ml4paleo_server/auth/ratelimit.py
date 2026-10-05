"""
Fixed-window rate limits, counted in Postgres so they hold across API
processes.
"""

import datetime

from fastapi import HTTPException
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

_INCREMENT = text(
    """
    INSERT INTO rate_limits (key, window_start, count)
    VALUES (:key, now(), 1)
    ON CONFLICT (key) DO UPDATE SET
        count = CASE WHEN rate_limits.window_start <= now() - :window
                     THEN 1 ELSE rate_limits.count + 1 END,
        window_start = CASE WHEN rate_limits.window_start <= now() - :window
                            THEN now() ELSE rate_limits.window_start END
    RETURNING count, window_start
    """
)


async def hit(
    session: AsyncSession, key: str, *, limit: int, window: datetime.timedelta
) -> None:
    """
    Count one request against `key`, and raise 429 once more than `limit`
    requests arrive within `window`. Commits immediately, so the count sticks
    even if the request then fails.
    """
    count, window_start = (
        await session.execute(_INCREMENT, {"key": key, "window": window})
    ).one()
    await session.commit()
    if count > limit:
        retry_after = window_start + window - datetime.datetime.now(datetime.UTC)
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Try again later.",
            headers={"Retry-After": str(max(1, int(retry_after.total_seconds())))},
        )
