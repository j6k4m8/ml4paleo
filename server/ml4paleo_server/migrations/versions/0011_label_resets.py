"""Irreversible, class-wide annotation resets in the label change feed."""

from alembic import op

revision = "0011_label_resets"
down_revision = "0010_live_sessions"
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint(op.f("ck_label_ops_kind"), "label_ops", type_="check")
    op.create_check_constraint(
        op.f("ck_label_ops_kind"),
        "label_ops",
        "kind IN ('edit', 'undo', 'redo', 'reset')",
    )


def downgrade():
    # Do not discard reset barriers: rolling back code with resets would let
    # undo resurrect erased annotations. An empty/new database can roll back.
    op.execute(
        "DO $$ BEGIN IF EXISTS (SELECT 1 FROM label_ops WHERE kind = 'reset') "
        "THEN RAISE EXCEPTION 'Cannot downgrade after permanent label resets'; "
        "END IF; END $$"
    )
    op.drop_constraint(op.f("ck_label_ops_kind"), "label_ops", type_="check")
    op.create_check_constraint(
        op.f("ck_label_ops_kind"), "label_ops", "kind IN ('edit', 'undo', 'redo')"
    )
