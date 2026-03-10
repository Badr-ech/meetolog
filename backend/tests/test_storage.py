"""
Tests for S3StorageService and FileMetadata persistence.

Uses moto (aiobotocore-compatible) to mock S3 and an in-memory SQLite
database to validate the metadata ORM layer without external services.
"""

import io
import os
import uuid
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import pytest_asyncio
from botocore.exceptions import ClientError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.infrastructure.db import Base
from app.models.metadata import FileMetadata
from app.services.storage import S3StorageService


# ---------------------------------------------------------------------------
# Database fixtures (in-memory SQLite via aiosqlite)
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture
async def db_engine():
    """Create an in-memory async SQLite engine and provision tables."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(db_engine):
    """Yield a fresh async session scoped to each test."""
    factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    async with factory() as session:
        yield session


# ---------------------------------------------------------------------------
# S3 fixtures
# ---------------------------------------------------------------------------

TEST_BUCKET = "test-meetolog-bucket"
TEST_REGION = "us-east-1"


def _make_s3_service() -> S3StorageService:
    """Create an S3StorageService pointing at the test bucket."""
    return S3StorageService(
        bucket=TEST_BUCKET,
        region=TEST_REGION,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


# ---------------------------------------------------------------------------
# FileMetadata ORM Tests
# ---------------------------------------------------------------------------

class TestFileMetadata:
    """Verify the SQLAlchemy ORM model for file metadata."""

    @pytest.mark.asyncio
    async def test_insert_and_query(self, db_session: AsyncSession):
        job_id = str(uuid.uuid4())
        record = FileMetadata(
            job_id=job_id,
            s3_key=f"uploads/{job_id}.mp3",
            original_filename="standup.mp3",
            file_size_bytes=1_048_576,
        )
        db_session.add(record)
        await db_session.commit()

        result = await db_session.execute(
            select(FileMetadata).where(FileMetadata.job_id == job_id)
        )
        row = result.scalar_one()

        assert row.job_id == job_id
        assert row.original_filename == "standup.mp3"
        assert row.file_size_bytes == 1_048_576
        assert row.s3_key.endswith(".mp3")
        assert row.created_at is not None

    @pytest.mark.asyncio
    async def test_unique_s3_key_constraint(self, db_session: AsyncSession):
        s3_key = "uploads/duplicate.mp3"
        for _ in range(2):
            db_session.add(
                FileMetadata(
                    job_id=str(uuid.uuid4()),
                    s3_key=s3_key,
                    original_filename="dup.mp3",
                    file_size_bytes=100,
                )
            )
        with pytest.raises(Exception):
            await db_session.commit()

    @pytest.mark.asyncio
    async def test_default_uuid_generation(self, db_session: AsyncSession):
        record = FileMetadata(
            job_id=str(uuid.uuid4()),
            s3_key="uploads/auto-id.mp3",
            original_filename="auto.mp3",
            file_size_bytes=500,
        )
        db_session.add(record)
        await db_session.commit()

        assert record.id is not None
        assert isinstance(record.id, uuid.UUID)


# ---------------------------------------------------------------------------
# S3StorageService Tests
# ---------------------------------------------------------------------------

class TestS3StorageService:
    """Test S3 upload/download via mocked aioboto3."""

    @pytest.mark.asyncio
    async def test_upload_stream_success(self):
        service = _make_s3_service()
        payload = b"fake-audio-content" * 100
        stream = io.BytesIO(payload)

        mock_s3_client = AsyncMock()
        mock_s3_client.upload_fileobj = AsyncMock()
        mock_s3_client.__aenter__ = AsyncMock(return_value=mock_s3_client)
        mock_s3_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(service, "_client_ctx", return_value=mock_s3_client):
            result = await service.upload_stream(stream, "uploads/test.mp3")

        assert result == f"s3://{TEST_BUCKET}/uploads/test.mp3"
        mock_s3_client.upload_fileobj.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_download_to_file_success(self, tmp_path):
        service = _make_s3_service()
        dest = str(tmp_path / "downloaded.mp3")

        mock_s3_client = AsyncMock()
        mock_s3_client.download_file = AsyncMock()
        mock_s3_client.__aenter__ = AsyncMock(return_value=mock_s3_client)
        mock_s3_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(service, "_client_ctx", return_value=mock_s3_client):
            await service.download_to_file("uploads/test.mp3", dest)

        mock_s3_client.download_file.assert_awaited_once_with(
            TEST_BUCKET, "uploads/test.mp3", dest
        )

    @pytest.mark.asyncio
    async def test_delete_object_success(self):
        service = _make_s3_service()

        mock_s3_client = AsyncMock()
        mock_s3_client.delete_object = AsyncMock()
        mock_s3_client.__aenter__ = AsyncMock(return_value=mock_s3_client)
        mock_s3_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(service, "_client_ctx", return_value=mock_s3_client):
            await service.delete_object("uploads/old.mp3")

        mock_s3_client.delete_object.assert_awaited_once_with(
            Bucket=TEST_BUCKET, Key="uploads/old.mp3"
        )

    @pytest.mark.asyncio
    async def test_upload_retries_on_transient_error(self):
        """Verify that a transient ClientError triggers retry and eventually succeeds."""
        service = _make_s3_service()
        payload = io.BytesIO(b"data")

        error_response = {"Error": {"Code": "InternalError", "Message": "Transient"}}
        transient_error = ClientError(error_response, "PutObject")

        mock_s3_client = AsyncMock()
        mock_s3_client.upload_fileobj = AsyncMock(
            side_effect=[transient_error, transient_error, None]
        )
        mock_s3_client.__aenter__ = AsyncMock(return_value=mock_s3_client)
        mock_s3_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(service, "_client_ctx", return_value=mock_s3_client):
            result = await service.upload_stream(payload, "uploads/retry.mp3")

        assert result == f"s3://{TEST_BUCKET}/uploads/retry.mp3"
        assert mock_s3_client.upload_fileobj.await_count == 3

    @pytest.mark.asyncio
    async def test_upload_fails_after_max_retries(self):
        """After exhausting retries the original ClientError should propagate."""
        service = _make_s3_service()
        payload = io.BytesIO(b"data")

        error_response = {"Error": {"Code": "InternalError", "Message": "Persistent"}}
        persistent_error = ClientError(error_response, "PutObject")

        mock_s3_client = AsyncMock()
        mock_s3_client.upload_fileobj = AsyncMock(side_effect=persistent_error)
        mock_s3_client.__aenter__ = AsyncMock(return_value=mock_s3_client)
        mock_s3_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(service, "_client_ctx", return_value=mock_s3_client):
            with pytest.raises(ClientError):
                await service.upload_stream(payload, "uploads/fail.mp3")

        # 4 attempts (1 initial + 3 retries)
        assert mock_s3_client.upload_fileobj.await_count == 4

    @pytest.mark.asyncio
    async def test_download_retries_on_connection_error(self, tmp_path):
        """ConnectionError should trigger retry logic."""
        service = _make_s3_service()
        dest = str(tmp_path / "retry-dl.mp3")

        mock_s3_client = AsyncMock()
        mock_s3_client.download_file = AsyncMock(
            side_effect=[ConnectionError("reset"), None]
        )
        mock_s3_client.__aenter__ = AsyncMock(return_value=mock_s3_client)
        mock_s3_client.__aexit__ = AsyncMock(return_value=False)

        with patch.object(service, "_client_ctx", return_value=mock_s3_client):
            await service.download_to_file("uploads/test.mp3", dest)

        assert mock_s3_client.download_file.await_count == 2
