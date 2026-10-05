"""
Settings that admins change at runtime, stored in the `site_settings` table.
Each falls back to its deploy-time default from `Settings` until an admin
sets it.
"""

from typing import Literal

from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from .db import SiteSetting
from .settings import Settings

SignupMode = Literal["open", "invite"]


async def get_signup_mode(db: AsyncSession, settings: Settings) -> SignupMode:
    row = await db.get(SiteSetting, "signup_mode")
    if row is not None and row.value in ("open", "invite"):
        return row.value
    return settings.auth.signup_mode


async def set_signup_mode(db: AsyncSession, mode: SignupMode) -> None:
    statement = insert(SiteSetting).values(key="signup_mode", value=mode)
    await db.execute(
        statement.on_conflict_do_update(
            index_elements=[SiteSetting.key], set_={"value": statement.excluded.value}
        )
    )
