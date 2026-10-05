"""
Site administration.
"""

import datetime

from fastapi import APIRouter
from pydantic import BaseModel

from ..auth.deps import AdminAuth, DbSession, SettingsDep
from ..auth.tokens import new_token, token_hash
from ..db import AuthToken
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
