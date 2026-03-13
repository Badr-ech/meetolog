"""Tests for FastAPI endpoints: GET /, POST /upload, GET /status/{job_id}."""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app, get_job_store, get_job_queue, get_s3_service
from app.infrastructure.db import get_async_session
from app.models import JobResponse, ProcessingStatus
from app.models.db_models import JobRecord


SAMPLE_JOB_ID = UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def mock_job_store(sample_job_response):
    """In-memory mock implementing the PostgresJobStore interface."""
    store = AsyncMock()
    store.save = AsyncMock()
    store.load = AsyncMock(return_value=sample_job_response)
    store.update = AsyncMock(return_value=sample_job_response)
    store.exists = AsyncMock(return_value=True)
    store.delete = AsyncMock(return_value=True)
    store.get_cached_artifacts = AsyncMock(return_value=None)
    return store


@pytest.fixture
def mock_job_queue():
    """Mock the PostgresJobQueue used for job enqueueing."""
    queue = AsyncMock()
    record = MagicMock(spec=JobRecord)
    record.id = SAMPLE_JOB_ID
    record.message = "Job queued"
    record.progress = 0
    queue.enqueue_job = AsyncMock(return_value=record)
    return queue


@pytest.fixture
def mock_s3_service():
    """Mock S3StorageService for upload operations."""
    s3 = AsyncMock()
    s3.upload_stream = AsyncMock()
    return s3


@pytest.fixture
def mock_db_session():
    """Mock async DB session."""
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    return session


@pytest_asyncio.fixture
async def client(mock_job_store, mock_job_queue, mock_s3_service, mock_db_session):
    """
    Yield an httpx.AsyncClient wired to the FastAPI app with mocked
    dependencies (Postgres job store, queue, S3, and DB session).
    """
    app.dependency_overrides[get_job_store] = lambda: mock_job_store
    app.dependency_overrides[get_job_queue] = lambda: mock_job_queue
    app.dependency_overrides[get_async_session] = lambda: mock_db_session

    with patch("app.main.get_s3_service", return_value=mock_s3_service):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    app.dependency_overrides.clear()


# GET /

@pytest.mark.asyncio
async def test_root_returns_service_info(client):
    response = await client.get("/")
    assert response.status_code == 200

    body = response.json()
    assert body["service"] == "Meetolog API"
    assert body["version"] == "3.0.0"
    assert body["status"] == "healthy"


# POST /upload

@pytest.mark.asyncio
async def test_upload_audio_returns_job_id(client, mock_job_queue):
    audio_bytes = b"\x00" * 1024
    files = {"file": ("test.wav", BytesIO(audio_bytes), "audio/wav")}

    response = await client.post("/upload", files=files)
    assert response.status_code == 200

    body = response.json()
    assert "job_id" in body
    assert body["status"] == "uploading"
    assert body["progress"] == 0

    mock_job_queue.enqueue_job.assert_awaited_once()


@pytest.mark.asyncio
async def test_upload_rejects_unsupported_extension(client):
    files = {"file": ("document.pdf", BytesIO(b"data"), "application/pdf")}

    response = await client.post("/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


@pytest.mark.asyncio
async def test_upload_rejects_missing_filename(client):
    files = {"file": ("", BytesIO(b"data"), "audio/wav")}

    response = await client.post("/upload", files=files)
    assert response.status_code in (400, 422)


@pytest.mark.asyncio
async def test_upload_returns_502_when_s3_unavailable(client, mock_s3_service):
    mock_s3_service.upload_stream.side_effect = ConnectionError("S3 down")

    audio_bytes = b"\x00" * 1024
    files = {"file": ("test.mp3", BytesIO(audio_bytes), "audio/mpeg")}

    response = await client.post("/upload", files=files)
    assert response.status_code == 502
    assert "storage" in response.json()["detail"].lower()


# GET /status/{job_id}

@pytest.mark.asyncio
async def test_get_status_returns_job(client, mock_job_store, sample_job_id):
    response = await client.get(f"/status/{sample_job_id}")
    assert response.status_code == 200

    body = response.json()
    assert body["job_id"] == str(sample_job_id)
    assert body["status"] == "uploading"


@pytest.mark.asyncio
async def test_get_status_returns_404_when_not_found(client, mock_job_store):
    mock_job_store.load.return_value = None

    response = await client.get(f"/status/{SAMPLE_JOB_ID}")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


# GET /download/{job_id}

@pytest.mark.asyncio
async def test_download_rejects_incomplete_job(client, mock_job_store, sample_job_id):
    response = await client.get(f"/download/{sample_job_id}")
    assert response.status_code == 400
    assert "not complete" in response.json()["detail"].lower()


# GET /artifacts/{job_id}

@pytest.mark.asyncio
async def test_artifacts_rejects_incomplete_job(client, mock_job_store, sample_job_id):
    response = await client.get(f"/artifacts/{sample_job_id}")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_artifacts_returns_data_when_complete(
    client, mock_job_store, sample_job_id, sample_artifacts
):
    completed_job = JobResponse(
        job_id=sample_job_id,
        status=ProcessingStatus.COMPLETED,
        progress=100,
        message="Done",
        artifacts=sample_artifacts,
    )
    mock_job_store.load.return_value = completed_job

    response = await client.get(f"/artifacts/{sample_job_id}")
    assert response.status_code == 200

    body = response.json()
    assert body["meeting_title"] == "Sprint Planning - Auth Feature"
    assert len(body["user_stories"]) == 1
