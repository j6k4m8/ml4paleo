"""
Per-user limits: how much a user can store and how many trained models they
can keep (plus optional compute-time limits for deploys that want them).

Each limit comes from the user's override if it has that key, otherwise from
the deploy's defaults; None means unlimited. Project storage counts against
the project owner.

Usage is reserved before work starts and released when the work fails or its
output is deleted. A reservation checks the limit and adds to usage in one
conditional UPDATE, so concurrent uploads or trainings can't each see the old
usage and together overshoot the limit. The UPDATE locks the user's usage row
until the caller's transaction ends, so commit soon after reserving.
"""

import uuid
from dataclasses import dataclass

from fastapi import HTTPException
from sqlalchemy import ColumnElement, func, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import InstrumentedAttribute

from .db import User, UserUsage
from .settings import QuotaSettings, Settings

GB = 1024**3


@dataclass(frozen=True)
class Limits:
    storage_bytes: int | None
    trained_models: int | None
    cpu_hours_per_day: float | None
    gpu_hours_per_day: float | None


def limits_for(user: User, settings: Settings) -> Limits:
    defaults = settings.quota.model_dump()
    merged = QuotaSettings.model_validate({**defaults, **(user.quota_override or {})})
    return Limits(
        storage_bytes=None
        if merged.storage_gb is None
        else int(merged.storage_gb * GB),
        trained_models=merged.trained_models,
        cpu_hours_per_day=merged.cpu_hours_per_day,
        gpu_hours_per_day=merged.gpu_hours_per_day,
    )


async def usage_for(db: AsyncSession, user_id: uuid.UUID) -> UserUsage:
    usage = await db.get(UserUsage, user_id, populate_existing=True)
    return usage or UserUsage(user_id=user_id, storage_bytes=0, trained_models=0)


async def reserve_storage(
    db: AsyncSession, settings: Settings, owner: User, nbytes: int
) -> None:
    """
    Count `nbytes` more storage against `owner`, or raise 403 if that would put
    them over their limit.
    """
    limit = limits_for(owner, settings).storage_bytes
    if not await _reserve(db, owner.id, UserUsage.storage_bytes, nbytes, limit):
        raise HTTPException(status_code=403, detail="storage_quota_exceeded")


async def check_storage(db: AsyncSession, settings: Settings, owner: User) -> None:
    """
    Raise 403 if `owner` already uses all the storage they're allowed, so
    work whose results couldn't be kept doesn't start. This only looks: the
    results are counted when they're committed.
    """
    limit = limits_for(owner, settings).storage_bytes
    if limit is not None and (await usage_for(db, owner.id)).storage_bytes >= limit:
        raise HTTPException(status_code=403, detail="storage_quota_exceeded")


async def release_storage(db: AsyncSession, owner_id: uuid.UUID, nbytes: int) -> None:
    await _release(db, owner_id, UserUsage.storage_bytes, nbytes)


async def reserve_trained_model(
    db: AsyncSession, settings: Settings, owner: User
) -> None:
    """
    Count one more trained model against `owner`, or raise 403 if they already
    keep as many as allowed.
    """
    limit = limits_for(owner, settings).trained_models
    if not await _reserve(db, owner.id, UserUsage.trained_models, 1, limit):
        raise HTTPException(status_code=403, detail="trained_model_quota_exceeded")


async def release_trained_model(db: AsyncSession, owner_id: uuid.UUID) -> None:
    await _release(db, owner_id, UserUsage.trained_models, 1)


async def _reserve(
    db: AsyncSession,
    user_id: uuid.UUID,
    column: InstrumentedAttribute[int],
    amount: int,
    limit: int | None,
) -> bool:
    if amount < 0:
        raise ValueError(f"Cannot reserve a negative amount ({amount})")
    await db.execute(insert(UserUsage).values(user_id=user_id).on_conflict_do_nothing())
    condition: ColumnElement[bool] = UserUsage.user_id == user_id
    if limit is not None:
        condition = condition & (column + amount <= limit)
    reserved = await db.execute(
        update(UserUsage)
        .where(condition)
        .values({column: column + amount})
        .returning(UserUsage.user_id)
        .execution_options(synchronize_session=False)
    )
    return reserved.first() is not None


async def _release(
    db: AsyncSession,
    user_id: uuid.UUID,
    column: InstrumentedAttribute[int],
    amount: int,
) -> None:
    if amount < 0:
        raise ValueError(f"Cannot release a negative amount ({amount})")
    await db.execute(
        update(UserUsage)
        .where(UserUsage.user_id == user_id)
        .values({column: func.greatest(column - amount, 0)})
        .execution_options(synchronize_session=False)
    )
