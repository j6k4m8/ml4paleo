"""
The signed-in user's own limits, usage, and requests for more.
"""

import datetime

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field
from sqlalchemy import select

from .. import audit, quotas
from ..auth import ratelimit
from ..auth.deps import CurrentAuth, DbSession, EngineDep, SettingsDep
from ..db import QuotaRequest, User
from ..email import queue_email

router = APIRouter(prefix="/api/me", tags=["me"])

DAY = datetime.timedelta(days=1)


class QuotaOut(BaseModel):
    storage_bytes_limit: int | None
    storage_bytes_used: int
    trained_models_limit: int | None
    trained_models_used: int
    open_request: bool


@router.get("/quota")
async def my_quota(auth: CurrentAuth, db: DbSession, settings: SettingsDep) -> QuotaOut:
    limits = quotas.limits_for(auth.user, settings)
    usage = await quotas.usage_for(db, auth.user.id)
    open_request = await db.scalar(
        select(QuotaRequest.id).where(
            QuotaRequest.user_id == auth.user.id, QuotaRequest.status == "open"
        )
    )
    return QuotaOut(
        storage_bytes_limit=limits.storage_bytes,
        storage_bytes_used=usage.storage_bytes,
        trained_models_limit=limits.trained_models,
        trained_models_used=usage.trained_models,
        open_request=open_request is not None,
    )


class QuotaRequestIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)


@router.post("/quota-requests", status_code=202)
async def request_more(
    body: QuotaRequestIn,
    request: Request,
    auth: CurrentAuth,
    db: DbSession,
    engine: EngineDep,
    settings: SettingsDep,
) -> None:
    """
    Ask the admins for higher limits. Admins with a verified email are told
    by email (when email is set up) and can see open requests in the admin
    pages either way.
    """
    await ratelimit.hit(
        engine, f"quota-request:user:{auth.user.id}", limit=3, window=DAY
    )
    quota_request = QuotaRequest(user_id=auth.user.id, message=body.message.strip())
    db.add(quota_request)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="quota.request",
        target_type="user",
        target_id=auth.user.id,
        request=request,
    )
    if settings.smtp.enabled:
        admins = (
            await db.scalars(
                select(User).where(
                    User.is_admin,
                    User.status == "active",
                    User.email_verified_at.is_not(None),
                )
            )
        ).all()
        for admin in admins:
            queue_email(
                db,
                settings,
                admin.email or "",
                f"ml4paleo: {auth.user.username} asked for more space",
                f"{auth.user.username} asked for higher limits:\n\n"
                f"{quota_request.message}\n\n"
                f"Review it at {settings.public_url}/admin/quota-requests\n",
            )
    await db.commit()
