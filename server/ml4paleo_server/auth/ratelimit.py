"""
Fixed-window rate limits, counted in Postgres so they hold across API
processes.

Counts are written on their own short transaction, never the request's: a
count must stick even when the request then fails, and committing it must
not commit (or release locks held by) the request's own work.
"""

import datetime
import ipaddress

from fastapi import HTTPException, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

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


def client_key(request: Request) -> str:
    """
    The client address to count against. IPv6 clients usually control a
    whole /64, so they are counted per /64.
    """
    host = request.client.host if request.client else "unknown"
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if isinstance(address, ipaddress.IPv6Address):
        if address.ipv4_mapped is not None:
            return str(address.ipv4_mapped)
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


_CURRENT = text(
    """
    SELECT count, window_start FROM rate_limits
    WHERE key = :key AND window_start > now() - :window
    """
)


async def peek(
    engine: AsyncEngine, key: str, *, limit: int, window: datetime.timedelta
) -> None:
    """
    Raise 429 if `key` has already reached `limit` in its current window,
    without counting this request. Pair with `hit` to count only failures.
    """
    async with engine.connect() as connection:
        row = (
            await connection.execute(_CURRENT, {"key": key, "window": window})
        ).one_or_none()
    if row is not None:
        count, window_start = row
        if count >= limit:
            _too_many(window_start, window)


async def hit(
    engine: AsyncEngine, key: str, *, limit: int, window: datetime.timedelta
) -> None:
    """
    Count one request against `key`, and raise 429 once more than `limit`
    requests arrive within `window`.
    """
    async with engine.begin() as connection:
        count, window_start = (
            await connection.execute(_INCREMENT, {"key": key, "window": window})
        ).one()
    if count > limit:
        _too_many(window_start, window)


_TAKE = text(
    """
    INSERT INTO rate_limits (key, window_start, count)
    VALUES (:key, now(), 1)
    ON CONFLICT (key) DO UPDATE SET
        count = CASE WHEN rate_limits.window_start <= now() - :window
                     THEN 1 ELSE rate_limits.count + 1 END,
        window_start = CASE WHEN rate_limits.window_start <= now() - :window
                            THEN now() ELSE rate_limits.window_start END
    WHERE rate_limits.window_start <= now() - :window
       OR rate_limits.count < :limit
    RETURNING window_start
    """
)
_STARTED = text("SELECT window_start FROM rate_limits WHERE key = :key")
_GIVE_BACK = text(
    """
    UPDATE rate_limits SET count = count - 1
    WHERE key = :key AND window_start = :window_start AND count > 0
    """
)


async def take(
    engine: AsyncEngine, key: str, *, limit: int, window: datetime.timedelta
) -> datetime.datetime:
    """
    Count one request against `key` if fewer than `limit` have been counted
    in its window, in one statement, so requests at once can't get past the
    limit together; otherwise raise 429 without counting it. Returns the
    window's start, for `give_back`.
    """
    async with engine.begin() as connection:
        started = (
            await connection.execute(
                _TAKE, {"key": key, "window": window, "limit": limit}
            )
        ).scalar_one_or_none()
        if started is None:
            started = (await connection.execute(_STARTED, {"key": key})).scalar_one()
            _too_many(started, window)
    return started


async def give_back(
    engine: AsyncEngine, key: str, window_start: datetime.datetime
) -> None:
    """Uncount a request `take` counted, if its window hasn't ended since."""
    async with engine.begin() as connection:
        await connection.execute(_GIVE_BACK, {"key": key, "window_start": window_start})


def _too_many(window_start: datetime.datetime, window: datetime.timedelta) -> None:
    retry_after = window_start + window - datetime.datetime.now(datetime.UTC)
    raise HTTPException(
        status_code=429,
        detail="Too many attempts. Try again later.",
        headers={"Retry-After": str(max(1, int(retry_after.total_seconds())))},
    )
