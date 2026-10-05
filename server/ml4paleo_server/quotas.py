"""
Per-user limits: how much a user can store and how many trained models they
can keep (plus optional compute-time limits for deploys that want them).

Each limit comes from the user's override if it has that key, otherwise from
the deploy's defaults; None means unlimited. Project storage counts against
the project owner.
"""

import uuid
from dataclasses import dataclass

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

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
    usage = await db.get(UserUsage, user_id)
    return usage or UserUsage(user_id=user_id, storage_bytes=0, trained_models=0)


async def check_storage(
    db: AsyncSession, settings: Settings, owner: User, additional_bytes: int
) -> None:
    """
    Raise 403 if storing `additional_bytes` more would put `owner` over their
    storage limit.
    """
    limit = limits_for(owner, settings).storage_bytes
    if limit is None:
        return
    usage = await usage_for(db, owner.id)
    if usage.storage_bytes + additional_bytes > limit:
        raise HTTPException(status_code=403, detail="storage_quota_exceeded")


async def check_trained_models(
    db: AsyncSession, settings: Settings, owner: User
) -> None:
    """
    Raise 403 if `owner` already keeps as many trained models as allowed.
    """
    limit = limits_for(owner, settings).trained_models
    if limit is None:
        return
    usage = await usage_for(db, owner.id)
    if usage.trained_models >= limit:
        raise HTTPException(status_code=403, detail="trained_model_quota_exceeded")
