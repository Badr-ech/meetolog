"""add cancelled state to job_records

Revision ID: f1a2b3c4d5e6
Revises: c7d9e1f3a5b2
Create Date: 2026-05-23 00:00:00.000000

Adds a ``cancelled_at`` timestamp column to ``job_records`` so that
cancellation events carry an authoritative timestamp for observability
and audit purposes.

The ``status`` column is already defined as ``String(32)`` with no
database-level CHECK constraint, so the ``"cancelled"`` value is
accepted without any DDL change to the column itself.  This migration
only adds the new timestamp column.

After applying this migration the following job lifecycle is in effect:

    pending → processing → completed
                       ↘ failed (auto-retry eligible)
    pending → cancelled   (API-initiated, pre-claim)
    processing → cancelled (API-initiated, worker polls and aborts)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "c7d9e1f3a5b2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add cancelled_at timestamp column to job_records."""
    op.add_column(
        "job_records",
        sa.Column(
            "cancelled_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
    )


def downgrade() -> None:
    """Remove the cancelled_at column."""
    op.drop_column("job_records", "cancelled_at")
