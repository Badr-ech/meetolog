"""add transcript_embeddings table for pgvector

Revision ID: c7d9e1f3a5b2
Revises: a1b2c3d4e5f6
Create Date: 2026-03-11 10:00:00.000000

This migration enables the ``vector`` PostgreSQL extension and creates
the ``transcript_embeddings`` table used by the pgvector RAG storage
backend (``RAG_STORAGE_BACKEND=pgvector``).

Prerequisites:
  - The ``vector`` extension must be available on the PostgreSQL server.
    On RDS, enable it via the ``shared_preload_libraries`` parameter.
    On self-hosted Postgres, install ``pgvector`` from source or packages.
  - If you are NOT using the pgvector backend, this migration is a no-op
    that can safely run (the table will simply remain empty).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c7d9e1f3a5b2'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Enable pgvector extension and create the transcript embeddings table."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "transcript_embeddings",
        sa.Column("id", sa.Uuid(), primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("job_id", sa.Uuid(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        # vector column — dimensionless to support both OpenAI (1536) and Gemini (768).
        # pgvector stores the actual dimension per row; cosine distance (<=>)
        # works correctly across rows with the same dimension.
        sa.Column("embedding", sa.LargeBinary(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    # The embedding column type must be set via raw DDL because
    # SQLAlchemy's Column() does not natively support pgvector's
    # ``vector`` type without the pgvector Python package.
    op.execute(
        "ALTER TABLE transcript_embeddings "
        "ALTER COLUMN embedding TYPE vector USING embedding::vector"
    )

    op.create_index(
        "ix_transcript_embeddings_job_id",
        "transcript_embeddings",
        ["job_id"],
    )


def downgrade() -> None:
    """Drop the transcript embeddings table."""
    op.drop_index("ix_transcript_embeddings_job_id", table_name="transcript_embeddings")
    op.drop_table("transcript_embeddings")
    # Note: we intentionally do NOT drop the vector extension since
    # other objects in the database may depend on it.
