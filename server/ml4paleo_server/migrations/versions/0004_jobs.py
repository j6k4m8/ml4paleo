"""
The job queue: workers, jobs, their dependencies, and attempts.

Revision ID: 0004_jobs
Revises: 0003_projects
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0004_jobs"
down_revision: str | None = "0003_projects"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "workers",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("pool", sa.String(length=16), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column(
            "caps",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_by", sa.Uuid(), nullable=True),
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
        sa.CheckConstraint(
            "pool IN ('local', 'remote', 'burst')", name=op.f("ck_workers_pool")
        ),
        sa.CheckConstraint(
            "status IN ('active', 'revoked')", name=op.f("ck_workers_status")
        ),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.id"],
            name=op.f("fk_workers_created_by_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_workers")),
        sa.UniqueConstraint("name", name=op.f("uq_workers_name")),
        sa.UniqueConstraint("token_hash", name=op.f("uq_workers_token_hash")),
    )
    op.create_table(
        "jobs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("root_id", sa.Uuid(), nullable=False),
        sa.Column("parent_id", sa.Uuid(), nullable=True),
        sa.Column("project_id", sa.Uuid(), nullable=True),
        sa.Column("created_by", sa.Uuid(), nullable=True),
        sa.Column("kind", sa.String(length=64), nullable=False),
        sa.Column("tier", sa.SmallInteger(), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("result", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column(
            "required_labels",
            postgresql.ARRAY(sa.String(length=64)),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "min_vram_gb", sa.Float(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column("weight", sa.Float(), server_default=sa.text("1"), nullable=False),
        sa.Column("progress", sa.Float(), server_default=sa.text("0"), nullable=False),
        sa.Column("message", sa.String(length=200), nullable=True),
        sa.Column(
            "attempts", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column(
            "max_attempts", sa.Integer(), server_default=sa.text("3"), nullable=False
        ),
        sa.Column(
            "not_before",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("submitted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("scale_trigger", sa.Boolean(), nullable=False),
        sa.Column("lease_worker_id", sa.Uuid(), nullable=True),
        sa.Column("lease_token_hash", sa.String(length=64), nullable=True),
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("cancel_requested", sa.Boolean(), nullable=False),
        sa.Column("idempotency_key", sa.String(length=200), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint(
            "status IN ('blocked', 'queued', 'leased', 'succeeded', 'failed', 'cancelled')",
            name=op.f("ck_jobs_status"),
        ),
        sa.CheckConstraint(
            "progress >= 0 AND progress <= 1", name=op.f("ck_jobs_progress")
        ),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.id"],
            name=op.f("fk_jobs_created_by_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["lease_worker_id"],
            ["workers.id"],
            name=op.f("fk_jobs_lease_worker_id_workers"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"],
            ["jobs.id"],
            name=op.f("fk_jobs_parent_id_jobs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_jobs_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["root_id"],
            ["jobs.id"],
            name=op.f("fk_jobs_root_id_jobs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
        sa.UniqueConstraint("idempotency_key", name=op.f("uq_jobs_idempotency_key")),
    )
    op.create_index(
        "ix_jobs_claim_order",
        "jobs",
        ["tier", "submitted_at", "id"],
        unique=False,
        postgresql_where=sa.text("status = 'queued'"),
    )
    op.create_index(
        "ix_jobs_lease_expiry",
        "jobs",
        ["lease_expires_at"],
        unique=False,
        postgresql_where=sa.text("status = 'leased'"),
    )
    op.create_index(op.f("ix_jobs_parent_id"), "jobs", ["parent_id"], unique=False)
    op.create_index(op.f("ix_jobs_project_id"), "jobs", ["project_id"], unique=False)
    op.create_index(op.f("ix_jobs_root_id"), "jobs", ["root_id"], unique=False)
    op.create_table(
        "job_attempts",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("worker_id", sa.Uuid(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("outcome", sa.String(length=16), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name=op.f("fk_job_attempts_job_id_jobs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["worker_id"],
            ["workers.id"],
            name=op.f("fk_job_attempts_worker_id_workers"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_job_attempts")),
    )
    op.create_index(
        op.f("ix_job_attempts_job_id"), "job_attempts", ["job_id"], unique=False
    )
    op.create_table(
        "job_deps",
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("depends_on", sa.Uuid(), nullable=False),
        sa.ForeignKeyConstraint(
            ["depends_on"],
            ["jobs.id"],
            name=op.f("fk_job_deps_depends_on_jobs"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name=op.f("fk_job_deps_job_id_jobs"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("job_id", "depends_on", name=op.f("pk_job_deps")),
    )
    op.create_index(
        op.f("ix_job_deps_depends_on"), "job_deps", ["depends_on"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_job_deps_depends_on"), table_name="job_deps")
    op.drop_table("job_deps")
    op.drop_index(op.f("ix_job_attempts_job_id"), table_name="job_attempts")
    op.drop_table("job_attempts")
    op.drop_index(op.f("ix_jobs_root_id"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_project_id"), table_name="jobs")
    op.drop_index(op.f("ix_jobs_parent_id"), table_name="jobs")
    op.drop_index(
        "ix_jobs_lease_expiry",
        table_name="jobs",
        postgresql_where=sa.text("status = 'leased'"),
    )
    op.drop_index(
        "ix_jobs_claim_order",
        table_name="jobs",
        postgresql_where=sa.text("status = 'queued'"),
    )
    op.drop_table("jobs")
    op.drop_table("workers")
