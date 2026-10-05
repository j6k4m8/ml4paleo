"""
Accounts, sessions, and sign-in.
"""

import secrets

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import User
from .passwords import hash_password
from .sessions import delete_user_sessions


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
        await delete_user_sessions(db, user.id)
    user.is_admin = True
    user.status = "active"
    user.password_hash = await hash_password(password)
    user.must_change_password = True
    user.totp_secret_enc = None
    user.totp_pending_enc = None
    user.totp_last_step = None
    await db.commit()
    return password if generated else None
