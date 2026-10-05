"""
Database tables. Each build step adds its tables here, with an Alembic
migration in `ml4paleo_server/migrations/versions`.
"""

from typing import Any

from sqlalchemy import String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin


class SiteSetting(TimestampMixin, Base):
    """
    Runtime settings that admins change from the web UI (for example whether
    signup is open). Deploy-time settings live in environment variables.
    """

    __tablename__ = "site_settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[Any] = mapped_column(JSONB, nullable=False)
