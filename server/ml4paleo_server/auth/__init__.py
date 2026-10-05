"""
Accounts, sessions, and sign-in.
"""

import datetime
import secrets
import uuid

from sqlalchemy import delete, func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import AuthToken, User
from .passwords import hash_password
from .sessions import delete_user_sessions


async def expire_reset_tokens(db: AsyncSession, user_id: uuid.UUID) -> None:
    """
    Use up every outstanding password-reset link for a user. Call this
    whenever their password changes, by any route.
    """
    await db.execute(
        update(AuthToken)
        .where(
            AuthToken.user_id == user_id,
            AuthToken.kind == "reset",
            AuthToken.used_at.is_(None),
        )
        .values(used_at=datetime.datetime.now(datetime.UTC))
    )


def random_password() -> str:
    """
    A random 24-character password for bootstrapped or reset accounts.
    """
    return secrets.token_urlsafe(18)


async def ensure_admin(db: AsyncSession, password: str | None = None) -> str | None:
    """
    If no admin exists, make the `admin` account an admin with `password` (or
    a random one), and return the random password so the caller can show it
    once. Returns None if an admin already exists or `password` was given.

    The admin must change the password and set up two-factor sign-in at
    first login. If someone already holds the `admin` username, that account
    is taken over: its two-factor setup is cleared and its sessions end.
    """
    admins = await db.scalar(
        select(func.count()).select_from(User).where(User.is_admin)
    )
    if admins:
        return None
    generated = password is None
    password = password or random_password()
    user = await db.scalar(select(User).where(User.username == "admin"))
    if user is None:
        user = User(username="admin")
        db.add(user)
    else:
        # Someone else signed up as "admin" before the real admin existed:
        # take the account over completely, so they can't get it back by
        # email or an old session.
        await delete_user_sessions(db, user.id)
        await db.execute(delete(AuthToken).where(AuthToken.user_id == user.id))
        user.email = None
        user.email_verified_at = None
    user.is_admin = True
    user.status = "active"
    user.password_hash = await hash_password(password)
    user.must_change_password = True
    user.totp_secret_enc = None
    user.totp_pending_enc = None
    user.totp_last_step = None
    await db.commit()
    return password if generated else None
