"""
Browser sessions: a random token in an HttpOnly cookie, and a row in the
`sessions` table holding its hash.

Sessions expire after `session_idle_days` without use, and always after
`session_max_days`.
"""

import datetime
import uuid

from fastapi import Request, Response
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import User, UserSession
from ..settings import Settings
from .tokens import new_token, token_hash

# The __Host- prefix makes browsers refuse the cookie unless it is Secure,
# host-only, and for path "/". It only works over HTTPS.
SECURE_COOKIE_NAME = "__Host-m4p_session"
PLAIN_COOKIE_NAME = "m4p_session"

# Refresh `last_seen_at` (and slide the idle expiry) at most this often.
_TOUCH_INTERVAL = datetime.timedelta(minutes=5)


def cookie_name(settings: Settings) -> str:
    return SECURE_COOKIE_NAME if settings.is_https else PLAIN_COOKIE_NAME


def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.UTC)


def _idle(settings: Settings) -> datetime.timedelta:
    return datetime.timedelta(days=settings.auth.session_idle_days)


def _max_age(settings: Settings) -> datetime.timedelta:
    return datetime.timedelta(days=settings.auth.session_max_days)


async def create_session(
    db: AsyncSession, settings: Settings, user: User, request: Request
) -> str:
    """
    Start a session for `user` and return its token. The caller commits.
    """
    token = new_token()
    now = _now()
    db.add(
        UserSession(
            token_hash=token_hash(token),
            user_id=user.id,
            created_at=now,
            last_seen_at=now,
            expires_at=now + min(_idle(settings), _max_age(settings)),
            ip=request.client.host if request.client else None,
            user_agent=(request.headers.get("user-agent") or "")[:512],
        )
    )
    user.last_login_at = now
    return token


def set_session_cookie(response: Response, settings: Settings, token: str) -> None:
    response.set_cookie(
        cookie_name(settings),
        token,
        max_age=int(_max_age(settings).total_seconds()),
        path="/",
        secure=settings.is_https,
        httponly=True,
        samesite="lax",
    )


def clear_session_cookie(response: Response, settings: Settings) -> None:
    response.delete_cookie(
        cookie_name(settings),
        path="/",
        secure=settings.is_https,
        httponly=True,
        samesite="lax",
    )


async def load_session(
    db: AsyncSession, settings: Settings, token: str
) -> tuple[UserSession, User] | None:
    """
    Return the session and its user for a token, or None if the token is
    unknown, expired, or belongs to a disabled user.
    """
    row = (
        await db.execute(
            select(UserSession, User)
            .join(User, User.id == UserSession.user_id)
            .where(UserSession.token_hash == token_hash(token))
        )
    ).one_or_none()
    if row is None:
        return None
    user_session, user = row
    now = _now()
    if (
        user_session.expires_at <= now
        or user_session.created_at + _max_age(settings) <= now
        or user.status == "disabled"
    ):
        return None
    if now - user_session.last_seen_at > _TOUCH_INTERVAL:
        user_session.last_seen_at = now
        user_session.expires_at = min(
            user_session.created_at + _max_age(settings), now + _idle(settings)
        )
        await db.commit()
    return user_session, user


async def delete_session(db: AsyncSession, token: str) -> None:
    await db.execute(
        delete(UserSession).where(UserSession.token_hash == token_hash(token))
    )


async def delete_user_sessions(
    db: AsyncSession, user_id: uuid.UUID, keep_token: str | None = None
) -> None:
    """
    Sign a user out everywhere, except (optionally) the session for `keep_token`.
    """
    statement = delete(UserSession).where(UserSession.user_id == user_id)
    if keep_token is not None:
        statement = statement.where(UserSession.token_hash != token_hash(keep_token))
    await db.execute(statement)
