"""
Labels: classes, chunks, the op log with claims, and ROIs.

Revision ID: 0007_labels
Revises: 0006_uploads
Create Date: 2026-10-05
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007_labels"
down_revision: str | None = "0006_uploads"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "label_chunks",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("cz", sa.Integer(), nullable=False),
        sa.Column("cy", sa.Integer(), nullable=False),
        sa.Column("cx", sa.Integer(), nullable=False),
        sa.Column("version", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("class_sha", sa.String(length=64), nullable=True),
        sa.Column("source_sha", sa.String(length=64), nullable=True),
        sa.Column(
            "labeled_voxels", sa.Integer(), server_default=sa.text("0"), nullable=False
        ),
        sa.Column(
            "class_counts",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_label_chunks_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "project_id", "cz", "cy", "cx", name=op.f("pk_label_chunks")
        ),
    )
    op.create_table(
        "label_classes",
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("value", sa.SmallInteger(), nullable=False),
        sa.Column("name", sa.String(length=100), nullable=False),
        sa.Column("color", sa.String(length=7), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "value BETWEEN 2 AND 254", name=op.f("ck_label_classes_value")
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_label_classes_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("project_id", "value", name=op.f("pk_label_classes")),
    )
    op.create_table(
        "rois",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("created_by", sa.Uuid(), nullable=True),
        sa.Column("bbox", postgresql.ARRAY(sa.BigInteger()), nullable=False),
        sa.Column("kind", sa.String(length=8), nullable=False),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("split", sa.String(length=8), nullable=False),
        sa.Column("origin", sa.String(length=16), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
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
        sa.CheckConstraint("kind IN ('cube', 'slice')", name=op.f("ck_rois_kind")),
        sa.CheckConstraint("split IN ('train', 'val')", name=op.f("ck_rois_split")),
        sa.CheckConstraint(
            "status IN ('open', 'complete', 'skipped')", name=op.f("ck_rois_status")
        ),
        sa.ForeignKeyConstraint(
            ["created_by"],
            ["users.id"],
            name=op.f("fk_rois_created_by_users"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_rois_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_rois")),
    )
    op.create_index(op.f("ix_rois_project_id"), "rois", ["project_id"], unique=False)
    op.create_table(
        "label_ops",
        sa.Column("seq", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("client_op_id", sa.Uuid(), nullable=False),
        sa.Column("kind", sa.String(length=8), nullable=False),
        sa.Column("user_id", sa.Uuid(), nullable=True),
        sa.Column("job_id", sa.Uuid(), nullable=True),
        sa.Column("source", sa.SmallInteger(), nullable=False),
        sa.Column(
            "tool",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column("bbox", postgresql.ARRAY(sa.BigInteger()), nullable=False),
        sa.Column("target_seq", sa.BigInteger(), nullable=True),
        sa.Column("live", sa.Boolean(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "kind IN ('edit', 'undo', 'redo')", name=op.f("ck_label_ops_kind")
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["jobs.id"],
            name=op.f("fk_label_ops_job_id_jobs"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_label_ops_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["target_seq"],
            ["label_ops.seq"],
            name=op.f("fk_label_ops_target_seq_label_ops"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            name=op.f("fk_label_ops_user_id_users"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("seq", name=op.f("pk_label_ops")),
    )
    op.create_index(
        "ix_label_ops_client_op",
        "label_ops",
        ["project_id", "client_op_id"],
        unique=True,
    )
    op.create_index(
        op.f("ix_label_ops_project_id"), "label_ops", ["project_id"], unique=False
    )
    op.create_table(
        "label_op_chunks",
        sa.Column("seq", sa.BigInteger(), nullable=False),
        sa.Column("project_id", sa.Uuid(), nullable=False),
        sa.Column("cz", sa.Integer(), nullable=False),
        sa.Column("cy", sa.Integer(), nullable=False),
        sa.Column("cx", sa.Integer(), nullable=False),
        sa.Column("base_version", sa.Integer(), nullable=False),
        sa.Column("new_version", sa.Integer(), nullable=False),
        sa.Column("class_sha", sa.String(length=64), nullable=True),
        sa.Column("claim_box", postgresql.ARRAY(sa.SmallInteger()), nullable=True),
        sa.Column("claim_mask", postgresql.BYTEA(), nullable=True),
        sa.Column("claim_values", postgresql.BYTEA(), nullable=True),
        sa.ForeignKeyConstraint(
            ["project_id"],
            ["projects.id"],
            name=op.f("fk_label_op_chunks_project_id_projects"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["seq"],
            ["label_ops.seq"],
            name=op.f("fk_label_op_chunks_seq_label_ops"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint(
            "seq", "cz", "cy", "cx", name=op.f("pk_label_op_chunks")
        ),
    )
    op.create_index(
        "ix_label_op_chunks_chunk",
        "label_op_chunks",
        ["project_id", "cz", "cy", "cx", "seq"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_label_op_chunks_chunk", table_name="label_op_chunks")
    op.drop_table("label_op_chunks")
    op.drop_index(op.f("ix_label_ops_project_id"), table_name="label_ops")
    op.drop_index("ix_label_ops_client_op", table_name="label_ops")
    op.drop_table("label_ops")
    op.drop_index(op.f("ix_rois_project_id"), table_name="rois")
    op.drop_table("rois")
    op.drop_table("label_classes")
    op.drop_table("label_chunks")
