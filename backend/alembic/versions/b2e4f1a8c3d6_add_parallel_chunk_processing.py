"""add parallel chunk processing

Revision ID: b2e4f1a8c3d6
Revises: f1a2b3c4d5e6
Create Date: 2026-05-23 00:00:00.000000

Adds the ``job_chunks`` table that backs the parallel transcription
pipeline.  Each row represents one 5-minute audio segment belonging to a
parent job.  Chunk workers claim rows via ``SELECT … FOR UPDATE SKIP
LOCKED``, transcribe the audio, and write the result back.  When the last
worker finishes it transitions the parent job to ``assembling`` so a
separate assembler task can reassemble the transcript and run extraction.

Also adds ``detected_language`` to ``job_records`` so the language
detected on the first chunk is propagated to all workers, preventing
Whisper from re-running language detection on every chunk (which was the
root cause of the stuck-on-chunk-12 bug with long recordings).
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "b2e4f1a8c3d6"
down_revision: Union[str, Sequence[str], None] = "f1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "job_records",
        sa.Column("detected_language", sa.String(10), nullable=True),
    )

    op.create_table(
        "job_chunks",
        sa.Column("id", sa.dialects.postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "job_id",
            sa.dialects.postgresql.UUID(as_uuid=True),
            sa.ForeignKey("job_records.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("chunk_index", sa.Integer, nullable=False),
        sa.Column("audio_s3_key", sa.String(512), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("transcript", sa.Text, nullable=True),
        sa.Column("detected_language", sa.String(10), nullable=True),
        sa.Column("locked_by", sa.String(128), nullable=True),
        sa.Column("locked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    op.create_index("ix_job_chunks_job_id", "job_chunks", ["job_id"])
    op.create_index(
        "ix_job_chunks_queue_poll",
        "job_chunks",
        ["job_id", "status", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_job_chunks_queue_poll", table_name="job_chunks")
    op.drop_index("ix_job_chunks_job_id", table_name="job_chunks")
    op.drop_table("job_chunks")
    op.drop_column("job_records", "detected_language")
