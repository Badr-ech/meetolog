"""SQLAlchemy ORM models for job tracking in PostgreSQL."""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Index, Integer, String, Text, Uuid
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from ..infrastructure.db import Base


class JobRecord(Base):
    __tablename__ = "job_records"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(), primary_key=True, default=uuid.uuid4,
    )
    s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    file_name: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    file_size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending", index=True,
    )
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    pdf_url: Mapped[str | None] = mapped_column(String(512), nullable=True)
    pdf_s3_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    artifacts_s3_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    artifacts: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    # Persistent queue fields
    locked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    locked_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_retries: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    next_retry_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    cancelled_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    detected_language: Mapped[str | None] = mapped_column(
        String(10), nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index(
            "ix_job_records_queue_poll",
            "status",
            "next_retry_at",
            "created_at",
        ),
    )


class JobChunk(Base):
    """One 5-minute audio segment belonging to a parent :class:`JobRecord`.

    The parallel transcription pipeline splits each uploaded recording into
    fixed-duration chunks and stores one row here per chunk.  Multiple
    ``chunk_worker`` Fargate tasks claim rows concurrently via
    ``SELECT … FOR UPDATE SKIP LOCKED``, transcribe their audio segment,
    and write the result back.  When the last worker marks its chunk
    ``completed`` it transitions the parent job to ``assembling``.
    """

    __tablename__ = "job_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(), primary_key=True, default=uuid.uuid4,
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        ForeignKey("job_records.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    audio_s3_key: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(
        String(32), nullable=False, default="pending",
    )
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Language detected while transcribing this chunk (ISO 639-1, e.g. "en").
    detected_language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    locked_by: Mapped[str | None] = mapped_column(String(128), nullable=True)
    locked_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    __table_args__ = (
        Index("ix_job_chunks_queue_poll", "job_id", "status", "created_at"),
    )
