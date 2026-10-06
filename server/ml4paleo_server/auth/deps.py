"""
FastAPI dependencies that identify the signed-in user.

Use `CurrentAuth` on every route that needs a signed-in user. It refuses
users who still have a required setup step: a forced password change, or
two-factor setup for admins. The few routes that complete those steps use
`SetupAuth` instead. (An unconfirmed email address isn't a step: it only
means starter limits, see `quotas`.)
"""

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from ..db import User, UserSession, get_session
from ..settings import Settings
from .sessions import cookie_name, load_session


@dataclass
class Auth:
    user: User
    session: UserSession
    # The raw session token, needed to derive the CSRF token.
    token: str


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_engine(request: Request) -> AsyncEngine:
    return request.app.state.engine


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbSession = Annotated[AsyncSession, Depends(get_session)]
EngineDep = Annotated[AsyncEngine, Depends(get_engine)]


async def optional_auth(
    request: Request, db: DbSession, engine: EngineDep, settings: SettingsDep
) -> Auth | None:
    token = request.cookies.get(cookie_name(settings))
    if not token:
        return None
    loaded = await load_session(db, engine, settings, token)
    if loaded is None:
        return None
    user_session, user = loaded
    return Auth(user=user, session=user_session, token=token)


async def setup_auth(auth: Annotated[Auth | None, Depends(optional_auth)]) -> Auth:
    if auth is None:
        raise HTTPException(status_code=401, detail="Sign in first.")
    return auth


async def current_auth(auth: Annotated[Auth, Depends(setup_auth)]) -> Auth:
    if auth.user.must_change_password:
        raise HTTPException(status_code=403, detail="password_change_required")
    if auth.user.is_admin and auth.user.totp_secret_enc is None:
        raise HTTPException(status_code=403, detail="two_factor_required")
    return auth


async def admin_auth(auth: Annotated[Auth, Depends(current_auth)]) -> Auth:
    if not auth.user.is_admin:
        raise HTTPException(status_code=403, detail="Admins only.")
    return auth


OptionalAuth = Annotated[Auth | None, Depends(optional_auth)]
SetupAuth = Annotated[Auth, Depends(setup_auth)]
CurrentAuth = Annotated[Auth, Depends(current_auth)]
AdminAuth = Annotated[Auth, Depends(admin_auth)]
