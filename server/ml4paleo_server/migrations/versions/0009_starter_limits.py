"""
Accounts no longer wait in an "unverified" status until their email is
confirmed; they get starter limits instead.

Revision ID: 0009_starter_limits
Revises: 0008_models
Create Date: 2026-10-06
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0009_starter_limits"
down_revision: str | None = "0008_models"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("UPDATE users SET status = 'active' WHERE status = 'unverified'")
    op.drop_constraint(op.f("ck_users_status"), "users", type_="check")
    op.create_check_constraint(
        op.f("ck_users_status"), "users", "status IN ('active', 'disabled')"
    )


def downgrade() -> None:
    op.drop_constraint(op.f("ck_users_status"), "users", type_="check")
    op.create_check_constraint(
        op.f("ck_users_status"),
        "users",
        "status IN ('active', 'unverified', 'disabled')",
    )
