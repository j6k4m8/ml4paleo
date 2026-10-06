"""
Settings that admins change at runtime, stored in the `site_settings` table.
Each falls back to its deploy-time default from `Settings` until an admin
sets it.
"""

from typing import Any, Literal

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from .db import SiteSetting
from .settings import Settings

SignupMode = Literal["open", "invite"]


async def get_signup_mode(db: AsyncSession, settings: Settings) -> SignupMode:
    value = await _get(db, "signup_mode")
    if value in ("open", "invite"):
        return value
    return settings.auth.signup_mode


async def set_signup_mode(db: AsyncSession, mode: SignupMode) -> None:
    await _set(db, "signup_mode", mode)


async def get_require_email(db: AsyncSession, settings: Settings) -> bool:
    """Whether signing up needs an email address (see `AuthSettings`)."""
    value = await _get(db, "require_email")
    if isinstance(value, bool):
        return value
    return settings.auth.require_email


async def set_require_email(db: AsyncSession, required: bool) -> None:
    await _set(db, "require_email", required)


async def _get(db: AsyncSession, key: str) -> Any:
    row = await db.get(SiteSetting, key, populate_existing=True)
    return None if row is None else row.value


async def _set(db: AsyncSession, key: str, value: Any) -> None:
    statement = insert(SiteSetting).values(key=key, value=value)
    await db.execute(
        statement.on_conflict_do_update(
            index_elements=[SiteSetting.key], set_={"value": statement.excluded.value}
        )
    )
