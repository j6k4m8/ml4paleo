"""
Uploads: files sent from browsers straight to object storage.

Revision ID: 0006_uploads
Revises: 0005_artifacts
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006_uploads"
down_revision: str | None = "0005_artifacts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "uploads",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("created_by", sa.Uuid(), nullable=True),
        sa.Column("filename", sa.String(length=255), nullable=False),
        sa.Column("size", sa.BigInteger(), nullable=False),
        sa.Column("part_size", sa.BigInteger(), nullable=False),
        sa.Column("multipart_id", sa.String(length=1024), nullable=True),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint(
            "state IN ('uploading', 'complete', 'aborted', 'deleting', 'deleted')",
            name=op.f("ck_uploads_state"),
        ),
        sa.CheckConstraint("size > 0 AND part_size > 0", name=op.f("ck_uploads_sizes")),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.id"],
            name=op.f("fk_uploads_created_by_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_uploads_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_uploads")),
    )
    op.create_index(
        op.f("ix_uploads_expires_at"), "uploads", ["expires_at"], unique=False
    )
    op.create_index(
        op.f("ix_uploads_project_id"), "uploads", ["project_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_uploads_project_id"), table_name="uploads")
    op.drop_index(op.f("ix_uploads_expires_at"), table_name="uploads")
    op.drop_table("uploads")
