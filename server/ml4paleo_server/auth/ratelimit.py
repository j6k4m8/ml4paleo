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
    if address.version == 6:
        return str(ipaddress.ip_network(f"{address}/64", strict=False))
    return str(address)


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
        retry_after = window_start + window - datetime.datetime.now(datetime.UTC)
        raise HTTPException(
            status_code=429,
            detail="Too many attempts. Try again later.",
            headers={"Retry-After": str(max(1, int(retry_after.total_seconds())))},
        )
