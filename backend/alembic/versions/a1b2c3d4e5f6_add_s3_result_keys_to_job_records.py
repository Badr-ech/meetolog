"""add s3 result keys to job_records

Revision ID: a1b2c3d4e5f6
Revises: ed48278a6f66
Create Date: 2026-03-09 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = 'ed48278a6f66'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'job_records',
        sa.Column('pdf_s3_key', sa.String(length=512), nullable=True),
    )
    op.add_column(
        'job_records',
        sa.Column('artifacts_s3_key', sa.String(length=512), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('job_records', 'artifacts_s3_key')
    op.drop_column('job_records', 'pdf_s3_key')
