"""
Worker credentials.

A worker authenticates with a bearer token `m4pw_<random>`. The database
keeps only the token's SHA-256, and the token is shown once when it is made.
The compose deploy's local workers share one token from
`M4P_LOCAL_WORKER_TOKEN` (usually `M4P_LOCAL_WORKER_TOKEN_FILE`), which
`migrate` registers as the worker named "local".
"""

import datetime
import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ml4paleo.protocol import WORKER_TOKEN_PREFIX

from ..auth.deps import DbSession
from ..auth.tokens import token_hash
from ..db import Worker

LOCAL_WORKER_NAME = "local"


def new_worker_token() -> str:
    return WORKER_TOKEN_PREFIX + secrets.token_urlsafe(32)


async def ensure_local_worker(db: AsyncSession, token: str) -> None:
    """
    Register the local workers' shared token. A new token replaces the old
    one and re-activates the worker; the same token leaves it alone, so a
    revoked local worker stays revoked across restarts.
    """
    if not token.startswith(WORKER_TOKEN_PREFIX):
        raise ValueError(f"Worker tokens start with {WORKER_TOKEN_PREFIX}")
    worker = await db.scalar(select(Worker).where(Worker.name == LOCAL_WORKER_NAME))
    if worker is None:
        db.add(
            Worker(
                name=LOCAL_WORKER_NAME,
                pool="local",
                token_hash=token_hash(token),
                caps={},
            )
        )
    elif worker.token_hash != token_hash(token):
        worker.token_hash = token_hash(token)
        worker.pool = "local"
        worker.status = "active"
    await db.commit()


async def current_worker(request: Request, db: DbSession) -> Worker:
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    worker = None
    if scheme.lower() == "bearer" and token.startswith(WORKER_TOKEN_PREFIX):
        worker = await db.scalar(
            select(Worker).where(
                Worker.token_hash == token_hash(token), Worker.status == "active"
            )
        )
    if worker is None or (
        worker.expires_at is not None
        and worker.expires_at < datetime.datetime.now(datetime.UTC)
    ):
        raise HTTPException(
            status_code=401,
            detail="A valid worker token is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return worker


CurrentWorker = Annotated[Worker, Depends(current_worker)]
