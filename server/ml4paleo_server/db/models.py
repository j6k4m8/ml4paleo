"""
Database tables. Each build step adds its tables here, with an Alembic
migration in `ml4paleo_server/migrations/versions`.
"""

import datetime
import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    String,
    Text,
    false,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base, TimestampMixin, uuid7


class SiteSetting(TimestampMixin, Base):
    """
    Runtime settings that admins change from the web UI (for example whether
    signup is open). Deploy-time settings live in environment variables.
    """

    __tablename__ = "site_settings"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[Any] = mapped_column(JSONB, nullable=False)


class User(TimestampMixin, Base):
    """
    A person with an account. Usernames and emails are stored lowercase.
    """

    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    username: Mapped[str] = mapped_column(String(32), unique=True)
    email: Mapped[str | None] = mapped_column(String(254), unique=True)
    email_verified_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    password_hash: Mapped[str | None] = mapped_column(String(255))
    is_admin: Mapped[bool] = mapped_column(default=False, server_default=false())
    # "active", "unverified" (waiting for email verification), or "disabled".
    status: Mapped[str] = mapped_column(String(16), default="active")
    must_change_password: Mapped[bool] = mapped_column(
        default=False, server_default=false()
    )
    # TOTP secrets are encrypted with a key derived from the server secret.
    totp_secret_enc: Mapped[str | None] = mapped_column(String(255))
    totp_pending_enc: Mapped[str | None] = mapped_column(String(255))
    # The time step of the last accepted code, so a code can't be reused.
    totp_last_step: Mapped[int | None] = mapped_column(BigInteger)
    last_login_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    __table_args__ = (
        CheckConstraint(
            "status IN ('active', 'unverified', 'disabled')", name="status"
        ),
    )


class UserSession(Base):
    """
    A signed-in browser. Only a hash of the session token is stored.
    """

    __tablename__ = "sessions"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_seen_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), index=True
    )
    ip: Mapped[str | None] = mapped_column(String(64))
    user_agent: Mapped[str | None] = mapped_column(String(512))


class AuthToken(Base):
    """
    A single-use token sent by email or shared as a link: email verification,
    password reset, or a signup invite. Only a hash of the token is stored.
    """

    __tablename__ = "auth_tokens"

    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[str] = mapped_column(String(16))
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    email: Mapped[str | None] = mapped_column(String(254))
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    expires_at: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    used_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("kind IN ('verify', 'reset', 'invite')", name="kind"),
    )


class RateLimit(Base):
    """
    Fixed-window request counters. The table is unlogged: losing it in a
    crash only resets the counters.
    """

    __tablename__ = "rate_limits"
    __table_args__ = {"prefixes": ["UNLOGGED"]}

    key: Mapped[str] = mapped_column(String(255), primary_key=True)
    window_start: Mapped[datetime.datetime] = mapped_column(DateTime(timezone=True))
    count: Mapped[int]


class EmailOutbox(Base):
    """
    Email waiting to be sent by the housekeeper.
    """

    __tablename__ = "email_outbox"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    to_address: Mapped[str] = mapped_column(String(254))
    subject: Mapped[str] = mapped_column(String(255))
    # The message text, sealed (`ml4paleo_server.sealing`): it holds
    # single-use links that must not sit readable in the database.
    body_sealed: Mapped[str] = mapped_column(Text)
    # "queued", "sent", or "failed".
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(default=0)
    # Failed sends wait longer before each retry.
    next_attempt_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    sent_at: Mapped[datetime.datetime | None] = mapped_column(DateTime(timezone=True))
