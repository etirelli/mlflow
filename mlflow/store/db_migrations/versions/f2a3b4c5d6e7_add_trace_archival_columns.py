"""add trace archival columns

Create Date: 2026-03-04 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy import inspect

# revision identifiers, used by Alembic.
revision = "f2a3b4c5d6e7"
down_revision = "e1f2a3b4c5d6"
branch_labels = None
depends_on = None


def upgrade():
    conn = op.get_bind()

    with op.batch_alter_table("spans", recreate="auto") as batch_op:
        batch_op.add_column(
            sa.Column("content_size", sa.BigInteger(), nullable=False, server_default="0")
        )

    inspector = inspect(conn)
    if "workspaces" in inspector.get_table_names():
        with op.batch_alter_table("workspaces", recreate="auto") as batch_op:
            batch_op.add_column(sa.Column("traces_destination", sa.Text(), nullable=True))


def downgrade():
    conn = op.get_bind()
    inspector = inspect(conn)

    if "workspaces" in inspector.get_table_names():
        with op.batch_alter_table("workspaces", recreate="auto") as batch_op:
            batch_op.drop_column("traces_destination")

    with op.batch_alter_table("spans", recreate="auto") as batch_op:
        batch_op.drop_column("content_size")
