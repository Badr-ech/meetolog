"""Tests for FastAPI endpoints: POST /upload, GET /status/{job_id}."""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app, get_job_store
from app.models import JobResponse, ProcessingStatus


SAMPLE_JOB_ID = UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def mock_job_store(sample_job_response):
    """In-memory mock implementing the RedisJobStore interface."""
    store = AsyncMock()
    store.save = AsyncMock()
    store.load = AsyncMock(return_value=sample_job_response)
    store.update = AsyncMock(return_value=sample_job_response)
    store.exists = AsyncMock(return_value=True)
    store.delete = AsyncMock(return_value=True)
    store.get_cached_artifacts = AsyncMock(return_value=None)
    return store


@pytest.fixture
def mock_arq_pool():
    """Mock the ARQ connection pool used for job enqueueing."""
    pool = AsyncMock()
    arq_job = MagicMock()
    arq_job.job_id = str(SAMPLE_JOB_ID)
    pool.enqueue_job = AsyncMock(return_value=arq_job)
    pool.close = AsyncMock()
    return pool


@pytest_asyncio.fixture
async def client(mock_job_store, mock_arq_pool):
    """
    Yield an httpx.AsyncClient wired to the FastAPI app with mocked
    job store and ARQ pool dependencies.

    Uses FastAPI's dependency_overrides so the real RedisJobStore is
    never instantiated (avoids Redis pipeline calls on the mock).
    """
    app.dependency_overrides[get_job_store] = lambda: mock_job_store

    with (
        patch(
            "app.main.check_redis_health",
            new_callable=AsyncMock,
            return_value={"status": "healthy"},
        ),
        patch(
            "app.main.get_arq_pool",
            new_callable=AsyncMock,
            return_value=mock_arq_pool,
        ),
    ):
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
    assert body["version"] == "2.0.0"
    assert body["status"] == "healthy"


# POST /upload

@pytest.mark.asyncio
async def test_upload_audio_returns_job_id(client, mock_job_store, mock_arq_pool):
    audio_bytes = b"\x00" * 1024
    files = {"file": ("test.wav", BytesIO(audio_bytes), "audio/wav")}

    response = await client.post("/upload", files=files)
    assert response.status_code == 200

    body = response.json()
    assert "job_id" in body
    assert body["status"] == "uploading"
    assert body["progress"] == 0

    mock_job_store.save.assert_awaited_once()
    mock_arq_pool.enqueue_job.assert_awaited_once()


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
async def test_upload_returns_503_when_arq_unavailable(
    client, mock_job_store, mock_arq_pool
):
    mock_arq_pool.enqueue_job.side_effect = ConnectionError("Redis down")

    audio_bytes = b"\x00" * 1024
    files = {"file": ("test.mp3", BytesIO(audio_bytes), "audio/mpeg")}

    response = await client.post("/upload", files=files)
    assert response.status_code == 503
    assert "queue unavailable" in response.json()["detail"].lower()


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
