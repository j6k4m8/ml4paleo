"""
Artifacts, artifact heads, and the storage each job may use.

Revision ID: 0005_artifacts
Revises: 0004_jobs
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0005_artifacts"
down_revision: str | None = "0004_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "artifacts",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column("state", sa.String(length=16), nullable=False),
        sa.Column(
            "bytes", sa.BigInteger(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column(
            "inputs",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("manifest", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("produced_by_job", sa.Uuid(), nullable=True),
        sa.Column("head_slot", sa.String(length=32), nullable=True),
        sa.Column("cache_key", sa.String(length=200), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "state_changed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "state IN ('staging', 'committed', 'superseded', 'failed', 'deleted')",
            name=op.f("ck_artifacts_state"),
        ),
        sa.ForeignKeyConstraint(
            ["produced_by_job"],
            ["jobs.id"],
            name=op.f("fk_artifacts_produced_by_job_jobs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_artifacts_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_artifacts")),
    )
    op.create_index(
        "ix_artifacts_cache_key",
        "artifacts",
        ["project_id", "cache_key"],
        unique=True,
        postgresql_where=sa.text("cache_key IS NOT NULL AND state = 'committed'"),
    )
    op.create_index(
        op.f("ix_artifacts_produced_by_job"),
        "artifacts",
        ["produced_by_job"],
        unique=False,
    )
    op.create_index(
        op.f("ix_artifacts_project_id"), "artifacts", ["project_id"], unique=False
    )
    op.create_index(
        "ix_artifacts_state", "artifacts", ["state", "state_changed_at"], unique=False
    )
    op.create_table(
        "artifact_heads",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("slot", sa.String(length=32), nullable=False),
        sa.Column("artifact_id", sa.Uuid(), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["artifact_id"],
            ["artifacts.id"],
            name=op.f("fk_artifact_heads_artifact_id_artifacts"),
            ondelete="RESTRICT",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_artifact_heads_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("project_id", "slot", name=op.f("pk_artifact_heads")),
    )
    op.create_index(
        op.f("ix_artifact_heads_artifact_id"),
        "artifact_heads",
        ["artifact_id"],
        unique=False,
    )
    op.add_column(
        "jobs",
        sa.Column(
            "grants",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
    )


def downgrade() -> None:
    op.drop_column("jobs", "grants")
    op.drop_index(op.f("ix_artifact_heads_artifact_id"), table_name="artifact_heads")
    op.drop_table("artifact_heads")
    op.drop_index("ix_artifacts_state", table_name="artifacts")
    op.drop_index(op.f("ix_artifacts_project_id"), table_name="artifacts")
    op.drop_index(op.f("ix_artifacts_produced_by_job"), table_name="artifacts")
    op.drop_index(
        "ix_artifacts_cache_key",
        table_name="artifacts",
        postgresql_where=sa.text("cache_key IS NOT NULL AND state = 'committed'"),
    )
    op.drop_table("artifacts")
