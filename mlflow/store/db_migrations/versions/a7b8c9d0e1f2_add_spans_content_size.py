"""add content_size column to spans table

Supports size-based trace archival policy by storing byte length of span content
at write time, avoiding SUM(LENGTH(content)) at query time.

Create Date: 2026-03-04 00:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "a7b8c9d0e1f2"
down_revision = "e1f2a3b4c5d6"
branch_labels = None
depends_on = None

_SPANS_TABLE = "spans"


def upgrade():
    op.add_column(
        _SPANS_TABLE,
        sa.Column(
            "content_size",
            sa.BigInteger(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    # Backfill content_size for existing spans so size-based archival policies
    # see accurate totals from the start.
    op.execute(
        sa.text(
            f"UPDATE {_SPANS_TABLE} SET content_size = LENGTH(content)"
            f" WHERE content IS NOT NULL AND content != ''"
        )
    )


def downgrade():
    op.drop_column(_SPANS_TABLE, "content_size")
