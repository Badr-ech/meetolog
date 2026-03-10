"""
SQLAlchemy ORM model for file metadata stored in PostgreSQL.

Tracks every audio file uploaded to S3 with its associated job,
original filename, S3 object key, and file size.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import BigInteger, DateTime, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column

from ..infrastructure.db import Base


class FileMetadata(Base):
    """Persisted metadata for audio files stored in S3."""

    __tablename__ = "file_metadata"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid(),
        primary_key=True,
        default=uuid.uuid4,
    )
    job_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        index=True,
    )
    s3_key: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        unique=True,
    )
    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
