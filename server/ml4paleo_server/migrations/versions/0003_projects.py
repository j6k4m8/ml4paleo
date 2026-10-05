"""
Projects, collaborators, the audit log, and quotas.

Revision ID: 0003_projects
Revises: 0002_auth
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_projects"
down_revision: str | None = "0002_auth"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("actor_user_id", sa.Uuid(), nullable=True),
        sa.Column("ip", sa.String(length=64), nullable=True),
        sa.Column("action", sa.String(length=64), nullable=False),
        sa.Column("target_type", sa.String(length=32), nullable=False),
        sa.Column("target_id", sa.String(length=64), nullable=False),
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["actor_user_id"],
            ["users.id"],
            name=op.f("fk_audit_log_actor_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_audit_log")),
    )
    op.create_index(
        op.f("ix_audit_log_actor_user_id"), "audit_log", ["actor_user_id"], unique=False
    )
    op.create_index(op.f("ix_audit_log_at"), "audit_log", ["at"], unique=False)
    op.create_index(
        "ix_audit_log_target", "audit_log", ["target_type", "target_id"], unique=False
    )
    op.create_table(
        "projects",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("owner_id", sa.Uuid(), nullable=False),
        sa.Column(
            "settings",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("v1_job_id", sa.String(length=6), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["owner_id"],
            ["users.id"],
            name=op.f("fk_projects_owner_id_users"),
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_projects")),
        sa.UniqueConstraint("v1_job_id", name=op.f("uq_projects_v1_job_id")),
    )
    op.create_index(
        op.f("ix_projects_owner_id"), "projects", ["owner_id"], unique=False
    )
    op.create_table(
        "quota_requests",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("message", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("resolved_by", sa.Uuid(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "status IN ('open', 'granted', 'declined')",
            name=op.f("ck_quota_requests_status"),
        ),
        sa.ForeignKeyConstraint(
            ["resolved_by"],
            ["users.id"],
            name=op.f("fk_quota_requests_resolved_by_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_quota_requests_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_quota_requests")),
    )
    op.create_index(
        op.f("ix_quota_requests_status"), "quota_requests", ["status"], unique=False
    )
    op.create_index(
        op.f("ix_quota_requests_user_id"), "quota_requests", ["user_id"], unique=False
    )
    op.create_table(
        "user_usage",
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column(
            "storage_bytes",
            sa.BigInteger(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "trained_models", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_user_usage_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("user_id", name=op.f("pk_user_usage")),
    )
    op.create_table(
        "project_members",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=False),
        sa.Column("added_by", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["added_by"],
            ["users.id"],
            name=op.f("fk_project_members_added_by_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_project_members_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_project_members_user_id_users"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "project_id", "user_id", name=op.f("pk_project_members")
        ),
    )
    op.create_index(
        op.f("ix_project_members_user_id"), "project_members", ["user_id"], unique=False
    )
    op.add_column(
        "users",
        sa.Column(
            "quota_override",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "quota_override")
    op.drop_index(op.f("ix_project_members_user_id"), table_name="project_members")
    op.drop_table("project_members")
    op.drop_table("user_usage")
    op.drop_index(op.f("ix_quota_requests_user_id"), table_name="quota_requests")
    op.drop_index(op.f("ix_quota_requests_status"), table_name="quota_requests")
    op.drop_table("quota_requests")
    op.drop_index(op.f("ix_projects_owner_id"), table_name="projects")
    op.drop_table("projects")
    op.drop_index("ix_audit_log_target", table_name="audit_log")
    op.drop_index(op.f("ix_audit_log_at"), table_name="audit_log")
    op.drop_index(op.f("ix_audit_log_actor_user_id"), table_name="audit_log")
    op.drop_table("audit_log")
