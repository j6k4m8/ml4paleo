"""
Site administration.
"""

import datetime
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import delete, func, or_, select

from .. import audit
from ..auth.deps import AdminAuth, DbSession, SettingsDep
from ..auth.tokens import new_token, token_hash
from ..db import AuthToken, QuotaRequest, User, UserSession, UserUsage
from ..settings import QuotaSettings
from ..site_settings import SignupMode, get_signup_mode, set_signup_mode
from .auth import Email

router = APIRouter(prefix="/api/admin", tags=["admin"])

INVITE_LIFETIME = datetime.timedelta(days=14)


class SiteSettingsOut(BaseModel):
    signup_mode: SignupMode


class SiteSettingsIn(BaseModel):
    signup_mode: SignupMode


@router.get("/settings")
async def get_site_settings(
    auth: AdminAuth, db: DbSession, settings: SettingsDep
) -> SiteSettingsOut:
    return SiteSettingsOut(signup_mode=await get_signup_mode(db, settings))


@router.put("/settings")
async def update_site_settings(
    body: SiteSettingsIn, auth: AdminAuth, db: DbSession
) -> SiteSettingsOut:
    await set_signup_mode(db, body.signup_mode)
    await db.commit()
    return SiteSettingsOut(signup_mode=body.signup_mode)


class InviteIn(BaseModel):
    # When set, signing up with this email skips email verification.
    email: Email = None


class InviteOut(BaseModel):
    url: str
    expires_at: datetime.datetime


@router.post("/invites", status_code=201)
async def create_invite(
    body: InviteIn, auth: AdminAuth, db: DbSession, settings: SettingsDep
) -> InviteOut:
    token = new_token()
    expires_at = datetime.datetime.now(datetime.UTC) + INVITE_LIFETIME
    db.add(
        AuthToken(
            token_hash=token_hash(token),
            kind="invite",
            email=body.email,
            expires_at=expires_at,
        )
    )
    await db.commit()
    return InviteOut(
        url=f"{settings.public_url}/signup?invite={token}", expires_at=expires_at
    )


class QuotaRequestOut(BaseModel):
    id: str
    user_id: str
    username: str
    message: str
    status: str
    created_at: datetime.datetime
    quota_override: dict


@router.get("/quota-requests")
async def list_quota_requests(auth: AdminAuth, db: DbSession) -> list[QuotaRequestOut]:
    rows = (
        await db.execute(
            select(QuotaRequest, User)
            .join(User, User.id == QuotaRequest.user_id)
            .where(QuotaRequest.status == "open")
            .order_by(QuotaRequest.created_at)
        )
    ).all()
    return [
        QuotaRequestOut(
            id=str(quota_request.id),
            user_id=str(user.id),
            username=user.username,
            message=quota_request.message,
            status=quota_request.status,
            created_at=quota_request.created_at,
            quota_override=user.quota_override or {},
        )
        for quota_request, user in rows
    ]


class QuotaOverride(QuotaSettings):
    """
    Limits that replace the deploy defaults for one user. Leave a limit out
    to use the default; set it to null for unlimited.
    """


class QuotaDecisionIn(BaseModel):
    decision: Literal["grant", "decline"]
    quota_override: QuotaOverride | None = None


@router.post("/quota-requests/{request_id}")
async def resolve_quota_request(
    request_id: uuid.UUID,
    body: QuotaDecisionIn,
    request: Request,
    auth: AdminAuth,
    db: DbSession,
) -> None:
    quota_request = await db.get(QuotaRequest, request_id)
    if quota_request is None or quota_request.status != "open":
        raise HTTPException(status_code=404, detail="No open request with that ID.")
    user = await db.get(User, quota_request.user_id)
    if body.decision == "grant":
        if body.quota_override is None or user is None:
            raise HTTPException(status_code=422, detail="Say which limits to grant.")
        user.quota_override = body.quota_override.model_dump(exclude_unset=True)
    quota_request.status = "granted" if body.decision == "grant" else "declined"
    quota_request.resolved_by = auth.user.id
    quota_request.resolved_at = datetime.datetime.now(datetime.UTC)
    audit.record(
        db,
        actor_id=auth.user.id,
        action=f"quota.{quota_request.status}",
        target_type="user",
        target_id=quota_request.user_id,
        request=request,
        details={"quota_override": user.quota_override if user else None},
    )
    await db.commit()


@router.put("/users/{user_id}/quota")
async def set_user_quota(
    user_id: uuid.UUID,
    body: QuotaOverride,
    request: Request,
    auth: AdminAuth,
    db: DbSession,
) -> dict:
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="No such user.")
    user.quota_override = body.model_dump(exclude_unset=True)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="quota.set",
        target_type="user",
        target_id=user.id,
        request=request,
        details={"quota_override": user.quota_override},
    )
    await db.commit()
    return user.quota_override


class UserOut(BaseModel):
    id: uuid.UUID
    username: str
    email: str | None
    is_admin: bool
    # "active", "unverified", or "disabled".
    status: str
    created_at: datetime.datetime
    last_login_at: datetime.datetime | None
    storage_bytes_used: int
    trained_models_used: int
    # Limits that replace the deploy's defaults for this user.
    quota_override: dict


@router.get("/users")
async def list_users(
    auth: AdminAuth,
    db: DbSession,
    q: str = "",
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> list[UserOut]:
    """
    Accounts, newest first. `q` matches the start of a username or email.
    """
    query = (
        select(User, UserUsage)
        .outerjoin(UserUsage, UserUsage.user_id == User.id)
        .order_by(User.created_at.desc())
        .limit(limit)
    )
    if prefix := q.strip().lower():
        pattern = (
            prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        )
        query = query.where(
            or_(
                func.lower(User.username).like(pattern, escape="\\"),
                func.lower(User.email).like(pattern, escape="\\"),
            )
        )
    return [
        UserOut(
            id=user.id,
            username=user.username,
            email=user.email,
            is_admin=user.is_admin,
            status=user.status,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
            storage_bytes_used=usage.storage_bytes if usage else 0,
            trained_models_used=usage.trained_models if usage else 0,
            quota_override=user.quota_override or {},
        )
        for user, usage in await db.execute(query)
    ]


class UserStatusIn(BaseModel):
    status: Literal["active", "disabled"]


@router.put("/users/{user_id}/status", status_code=204)
async def set_user_status(
    user_id: uuid.UUID,
    body: UserStatusIn,
    request: Request,
    auth: AdminAuth,
    db: DbSession,
) -> None:
    """
    Disable an account (it is signed out at once and can't sign in again) or
    enable it. Enabling also counts as verifying its email. Admins can't
    disable their own account.
    """
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="No such user.")
    if user.id == auth.user.id and body.status == "disabled":
        raise HTTPException(
            status_code=409, detail="You can't disable your own account."
        )
    user.status = body.status
    if body.status == "disabled":
        await db.execute(delete(UserSession).where(UserSession.user_id == user.id))
    audit.record(
        db,
        actor_id=auth.user.id,
        action=f"user.{'disable' if body.status == 'disabled' else 'enable'}",
        target_type="user",
        target_id=user.id,
        request=request,
    )
    await db.commit()
