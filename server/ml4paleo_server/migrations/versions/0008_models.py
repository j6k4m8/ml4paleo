"""
Models and the training sets they were trained on.

Revision ID: 0008_models
Revises: 0007_labels
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0008_models"
down_revision: str | None = "0007_labels"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "training_sets",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("summary", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_training_sets_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_training_sets")),
    )
    op.create_index(
        op.f("ix_training_sets_project_id"),
        "training_sets",
        ["project_id"],
        unique=False,
    )
    op.create_table(
        "models",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("plugin", sa.String(length=32), nullable=False),
        sa.Column("plugin_version", sa.String(length=32), nullable=True),
        sa.Column("params", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("training_set_id", sa.String(length=64), nullable=False),
        sa.Column("class_values", postgresql.ARRAY(sa.SmallInteger()), nullable=False),
        sa.Column("job_id", sa.Uuid(), nullable=True),
        sa.Column("artifact_id", sa.Uuid(), nullable=True),
        sa.Column("metrics", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "holds_slot", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("created_by", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["artifact_id"],
            ["artifacts.id"],
            name=op.f("fk_models_artifact_id_artifacts"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.id"],
            name=op.f("fk_models_created_by_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name=op.f("fk_models_job_id_jobs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_models_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["training_set_id"],
            ["training_sets.id"],
            name=op.f("fk_models_training_set_id_training_sets"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_models")),
    )
    op.create_index(
        op.f("ix_models_project_id"), "models", ["project_id"], unique=False
    )
    op.create_index(
        op.f("ix_models_training_set_id"), "models", ["training_set_id"], unique=False
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_models_training_set_id"), table_name="models")
    op.drop_index(op.f("ix_models_project_id"), table_name="models")
    op.drop_table("models")
    op.drop_index(op.f("ix_training_sets_project_id"), table_name="training_sets")
    op.drop_table("training_sets")
