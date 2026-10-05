"""
Accounts, sessions, and sign-in.
"""

import secrets

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..db import User
from .passwords import hash_password


def random_password() -> str:
    """
    A random 24-character password for bootstrapped or reset accounts.
    """
    return secrets.token_urlsafe(18)


async def ensure_admin(db: AsyncSession) -> str | None:
    """
    If no admin exists, create the `admin` account with a random password and
    return that password (so the caller can show it once). The admin must
    change the password and set up two-factor sign-in at first login.
    """
    admins = await db.scalar(
        select(func.count()).select_from(User).where(User.is_admin)
    )
    if admins:
        return None
    password = random_password()
    existing = await db.scalar(select(User).where(User.username == "admin"))
    if existing is not None:
        existing.is_admin = True
        existing.password_hash = await hash_password(password)
        existing.must_change_password = True
        existing.status = "active"
    else:
        db.add(
            User(
                username="admin",
                password_hash=await hash_password(password),
                is_admin=True,
                must_change_password=True,
            )
        )
    await db.commit()
    return password
