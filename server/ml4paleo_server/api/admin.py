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
from ..site_settings import (
    SignupMode,
    get_require_email,
    get_signup_mode,
    set_require_email,
    set_signup_mode,
)
from .auth import Email

router = APIRouter(prefix="/api/admin", tags=["admin"])

INVITE_LIFETIME = datetime.timedelta(days=14)


class SiteSettingsOut(BaseModel):
    signup_mode: SignupMode
    # Whether signing up needs an email address; while it does, accounts
    # whose address isn't confirmed get `unconfirmed_quota` (set at deploy).
    require_email: bool
    unconfirmed_quota: QuotaSettings


class SiteSettingsIn(BaseModel):
    """The settings to change; those left out stay as they are."""

    signup_mode: SignupMode | None = None
    require_email: bool | None = None


async def _site_settings(db, settings) -> SiteSettingsOut:
    return SiteSettingsOut(
        signup_mode=await get_signup_mode(db, settings),
        require_email=await get_require_email(db, settings),
        unconfirmed_quota=settings.unconfirmed_quota,
    )


@router.get("/settings")
async def get_site_settings(
    auth: AdminAuth, db: DbSession, settings: SettingsDep
) -> SiteSettingsOut:
    return await _site_settings(db, settings)


@router.put("/settings")
async def update_site_settings(
    body: SiteSettingsIn,
    request: Request,
    auth: AdminAuth,
    db: DbSession,
    settings: SettingsDep,
) -> SiteSettingsOut:
    if body.signup_mode is not None:
        await set_signup_mode(db, body.signup_mode)
    if body.require_email is not None:
        await set_require_email(db, body.require_email)
    audit.record(
        db,
        actor_id=auth.user.id,
        action="site.settings",
        target_type="site",
        target_id="settings",
        request=request,
        details=body.model_dump(exclude_none=True),
    )
    await db.commit()
    return await _site_settings(db, settings)


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
    email_confirmed: bool
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
            email_confirmed=user.email_verified_at is not None,
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
    email_confirmed: bool
    is_admin: bool
    # "active" or "disabled".
    status: str
    created_at: datetime.datetime
    last_login_at: datetime.datetime | None
    storage_bytes_used: int
    trained_models_used: int
    # Their limits (None: unlimited), whether those are the starter limits
    # until they confirm their email, and the limits that replace the
    # deploy's defaults for them.
    starter_limits: bool
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
    require_email = await get_require_email(db, settings)
    return [
        UserOut(
            id=user.id,
            username=user.username,
            email=user.email,
            email_confirmed=user.email_verified_at is not None,
            is_admin=user.is_admin,
            status=user.status,
            created_at=user.created_at,
            last_login_at=user.last_login_at,
            storage_bytes_used=usage.storage_bytes if usage else 0,
            trained_models_used=usage.trained_models if usage else 0,
            starter_limits=quotas.has_starter_limits(user, require_email=require_email),
            storage_bytes_limit=limits.storage_bytes,
            trained_models_limit=limits.trained_models,
            quota_override=user.quota_override or {},
        )
        for user, usage in await db.execute(query)
        for limits in [quotas.limits_of(user, settings, require_email=require_email)]
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
        user.status = "active"
    audit.record(
        db,
        actor_id=auth.user.id,
        action=f"user.{'disable' if body.status == 'disabled' else 'enable'}",
        target_type="user",
        target_id=user.id,
        request=request,
    )
    await db.commit()


class ConfirmEmailIn(BaseModel):
    # The address the admin is vouching for, as they saw it.
    email: Email


@router.post("/users/{user_id}/confirm-email", status_code=204)
async def confirm_user_email(
    user_id: uuid.UUID,
    body: ConfirmEmailIn,
    request: Request,
    auth: AdminAuth,
    db: DbSession,
) -> None:
    """
    Count an account's email address as confirmed, vouching for it (for
    example when email isn't set up, so people can't confirm it themselves).
    Refused if the account's address isn't the one given, say because it
    changed since the admin looked.
    """
    user = await db.get(User, user_id, with_for_update=True, populate_existing=True)
    if user is None:
        raise HTTPException(status_code=404, detail="No such user.")
    if not user.email:
        raise HTTPException(status_code=409, detail="They have no email address.")
    if user.email != body.email:
        raise HTTPException(
            status_code=409, detail="Their address changed; look again."
        )
    if user.email_verified_at is None:
        user.email_verified_at = datetime.datetime.now(datetime.UTC)
        audit.record(
            db,
            actor_id=auth.user.id,
            action="user.confirm_email",
            target_type="user",
            target_id=user.id,
            request=request,
            details={"email": user.email},
        )
    await db.commit()


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
