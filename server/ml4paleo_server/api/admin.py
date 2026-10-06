"""
Site administration.
"""

import datetime
import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy import delete, func, or_, select

from .. import audit, jobs, quotas
from ..auth.deps import AdminAuth, DbSession, SettingsDep
from ..auth.tokens import new_token, token_hash
from ..db import AuthToken, Job, QuotaRequest, User, UserSession, UserUsage
from ..email import queue_email
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
    url = f"{settings.public_url}/signup?invite={token}"
    # An invite addressed to someone is mailed to them (signing up with it
    # then counts as confirming that address).
    if body.email and settings.smtp.enabled:
        queue_email(
            db,
            settings,
            body.email,
            "You're invited to ml4paleo",
            f"Hi,\n\n{auth.user.username} invited you to ml4paleo. Make your "
            f"account with this link (it works once, for two weeks):\n\n{url}\n",
        )
    await db.commit()
    return InviteOut(url=url, expires_at=expires_at)


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
        # Granting adds to the limits someone has; it never lowers others.
        user.quota_override = {
            **(user.quota_override or {}),
            **body.quota_override.model_dump(exclude_unset=True),
        }
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
    # Their limits (None: unlimited), and the ones that replace the deploy's
    # defaults for them.
    storage_bytes_limit: int | None
    trained_models_limit: int | None
    quota_override: dict


@router.get("/users")
async def list_users(
    auth: AdminAuth,
    db: DbSession,
    settings: SettingsDep,
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
            storage_bytes_limit=limits.storage_bytes,
            trained_models_limit=limits.trained_models,
            quota_override=user.quota_override or {},
        )
        for user, usage in await db.execute(query)
        for limits in [quotas.limits_for(user, settings)]
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
    settings: SettingsDep,
) -> None:
    """
    Disable an account (it is signed out at once, can't sign in again, and
    the jobs it started stop) or enable it again. Admins can't disable their
    own account.
    """
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="No such user.")
    if user.id == auth.user.id and body.status == "disabled":
        raise HTTPException(
            status_code=409, detail="You can't disable your own account."
        )
    if body.status == "disabled":
        await disable(db, user)
    else:
        user.status = enabled_status(user, settings)
    audit.record(
        db,
        actor_id=auth.user.id,
        action=f"user.{'disable' if body.status == 'disabled' else 'enable'}",
        target_type="user",
        target_id=user.id,
        request=request,
    )
    await db.commit()


def enabled_status(user: User, settings) -> str:
    """
    What an account's status becomes when it's enabled: still waiting for
    its email to be confirmed, if mail is on and it never was.
    """
    if user.email and user.email_verified_at is None and settings.smtp.enabled:
        return "unverified"
    return "active"


async def disable(db, user: User) -> None:
    """Sign an account out for good, and stop the jobs it started."""
    user.status = "disabled"
    await db.execute(delete(UserSession).where(UserSession.user_id == user.id))
    roots = await db.scalars(
        select(Job.root_id)
        .where(
            Job.created_by == user.id,
            Job.status.in_(("blocked", "queued", "leased")),
        )
        .distinct()
    )
    for root_id in sorted(set(roots)):
        await jobs.cancel_pipeline(db, root_id)
