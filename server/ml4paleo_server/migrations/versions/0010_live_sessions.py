"""Expiring Live capacity reservations."""

import sqlalchemy as sa
from alembic import op

revision = "0010_live_sessions"
down_revision = "0009_starter_limits"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "live_sessions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "project_id",
            sa.Uuid(),
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.Uuid(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "image_id",
            sa.Uuid(),
            sa.ForeignKey("artifacts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(op.f("ix_live_sessions_user_id"), "live_sessions", ["user_id"])
    op.create_index(
        op.f("ix_live_sessions_expires_at"), "live_sessions", ["expires_at"]
    )


def downgrade():
    op.drop_table("live_sessions")
