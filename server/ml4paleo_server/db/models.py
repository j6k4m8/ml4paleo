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
    Index,
    String,
    Text,
    false,
    func,
    text,
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
    # Per-user limits that replace the deploy defaults, for example
    # {"storage_gb": 100}. A null value means unlimited.
    quota_override: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
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


class Project(TimestampMixin, Base):
    """
    One scan and everything made from it. The owner and collaborators all
    have full access; only the owner can delete the project. Storage used by
    the project counts against the owner's quota.
    """

    __tablename__ = "projects"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    name: Mapped[str] = mapped_column(String(100))
    owner_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="RESTRICT"), index=True
    )
    settings: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )
    # The v1 job this project was imported from, if any.
    v1_job_id: Mapped[str | None] = mapped_column(String(6), unique=True)
    deleted_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )


class ProjectMember(Base):
    """
    A person with access to a project. The owner has a row too.
    """

    __tablename__ = "project_members"

    project_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    added_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class AuditEvent(Base):
    """
    Who did what, for actions on accounts and projects.
    """

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), index=True
    )
    ip: Mapped[str | None] = mapped_column(String(64))
    action: Mapped[str] = mapped_column(String(64))
    target_type: Mapped[str] = mapped_column(String(32))
    target_id: Mapped[str] = mapped_column(String(64))
    details: Mapped[dict[str, Any]] = mapped_column(
        JSONB, default=dict, server_default=text("'{}'::jsonb")
    )

    __table_args__ = (Index("ix_audit_log_target", "target_type", "target_id"),)


class UserUsage(Base):
    """
    What each user currently uses against their quota. Change it only through
    `ml4paleo_server.quotas`, which reserves usage before work starts and
    releases it when the work fails or its output is deleted.
    """

    __tablename__ = "user_usage"

    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    storage_bytes: Mapped[int] = mapped_column(
        BigInteger, default=0, server_default=text("0")
    )
    trained_models: Mapped[int] = mapped_column(default=0, server_default=text("0"))


class QuotaRequest(Base):
    """
    A user's request for higher limits, for an admin to grant or decline.
    """

    __tablename__ = "quota_requests"

    id: Mapped[uuid.UUID] = mapped_column(primary_key=True, default=uuid7)
    user_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), index=True
    )
    message: Mapped[str] = mapped_column(Text)
    # "open", "granted", or "declined".
    status: Mapped[str] = mapped_column(String(16), default="open", index=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    resolved_by: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL")
    )
    resolved_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True)
    )

    __table_args__ = (
        CheckConstraint("status IN ('open', 'granted', 'declined')", name="status"),
    )
