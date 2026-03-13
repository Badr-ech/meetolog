"""Extended API endpoint tests: PUT /artifacts, GET /export/jira, GET /health,
GET /download edge cases, upload size rejection, and schema validation (422)."""

from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import Row

from app.main import app, get_job_store, get_job_queue, get_s3_service
from app.infrastructure.db import get_async_session
from app.models import (
    JobResponse,
    MeetingArtifacts,
    ProcessingStatus,
    UserStory,
    Task,
    Decision,
    Blocker,
    ActionItem,
    ActionableTask,
    Priority,
    TaskStatus,
)
from app.models.db_models import JobRecord

SAMPLE_JOB_ID = UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def sample_artifacts() -> MeetingArtifacts:
    return MeetingArtifacts(
        meeting_title="Test Meeting",
        participants=["Alice"],
        summary="Brief sync.",
        user_stories=[
            UserStory(
                title="Login",
                as_a="user",
                i_want="to log in",
                so_that="secure",
                priority=Priority.HIGH,
            ),
        ],
        tasks=[Task(title="Build API")],
        decisions=[Decision(title="Timeout", description="30 min")],
        blockers=[Blocker(title="Email down", description="Blocks reset")],
        action_items=[ActionItem(description="Update docs")],
        execution_tasks=[
            ActionableTask(
                title="Deploy",
                description="Ship to prod",
                owner_role="Eng",
                task_source="Explicit",
            ),
        ],
        transcript="Transcript text.",
    )


def _completed_job(artifacts: MeetingArtifacts | None = None) -> JobResponse:
    return JobResponse(
        job_id=SAMPLE_JOB_ID,
        status=ProcessingStatus.COMPLETED,
        progress=100,
        message="Done",
        artifacts=artifacts,
    )


def _processing_job() -> JobResponse:
    return JobResponse(
        job_id=SAMPLE_JOB_ID,
        status=ProcessingStatus.TRANSCRIBING,
        progress=30,
        message="Working",
    )


@pytest.fixture
def mock_job_store():
    store = AsyncMock()
    store.save = AsyncMock()
    store.load = AsyncMock(return_value=_processing_job())
    store.update = AsyncMock()
    store.exists = AsyncMock(return_value=True)
    store.delete = AsyncMock(return_value=True)
    store.get_cached_artifacts = AsyncMock(return_value=None)
    store.update_artifacts = AsyncMock()
    return store


@pytest.fixture
def mock_job_queue():
    queue = AsyncMock()
    record = MagicMock(spec=JobRecord)
    record.id = SAMPLE_JOB_ID
    record.message = "Job queued"
    record.progress = 0
    queue.enqueue_job = AsyncMock(return_value=record)
    return queue


@pytest.fixture
def mock_s3_service():
    s3 = AsyncMock()
    s3.upload_stream = AsyncMock()
    s3.generate_presigned_get_url = AsyncMock(return_value="https://s3.example.com/test.pdf")
    return s3


@pytest.fixture
def mock_db_session():
    session = AsyncMock()
    session.add = MagicMock()
    session.commit = AsyncMock()
    # Default: no pdf_s3_key found for download endpoint
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    session.execute = AsyncMock(return_value=mock_result)
    return session


@pytest_asyncio.fixture
async def client(mock_job_store, mock_job_queue, mock_s3_service, mock_db_session):
    app.dependency_overrides[get_job_store] = lambda: mock_job_store
    app.dependency_overrides[get_job_queue] = lambda: mock_job_queue
    app.dependency_overrides[get_async_session] = lambda: mock_db_session

    with patch("app.main.get_s3_service", return_value=mock_s3_service):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    app.dependency_overrides.clear()


# PUT /artifacts/{job_id}

@pytest.mark.asyncio
async def test_put_artifacts_succeeds_for_completed_job(
    client, mock_job_store, sample_artifacts
):
    mock_job_store.load.return_value = _completed_job(sample_artifacts)
    mock_job_store.update_artifacts.return_value = _completed_job(sample_artifacts)

    response = await client.put(
        f"/artifacts/{SAMPLE_JOB_ID}",
        json=sample_artifacts.model_dump(mode="json"),
    )
    assert response.status_code == 200
    mock_job_store.update_artifacts.assert_awaited_once()


@pytest.mark.asyncio
async def test_put_artifacts_rejects_incomplete_job(client, mock_job_store):
    mock_job_store.load.return_value = _processing_job()

    response = await client.put(
        f"/artifacts/{SAMPLE_JOB_ID}",
        json=MeetingArtifacts().model_dump(mode="json"),
    )
    assert response.status_code == 400
    assert "status" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_put_artifacts_returns_404_for_missing_job(client, mock_job_store):
    mock_job_store.load.return_value = None

    response = await client.put(
        f"/artifacts/{SAMPLE_JOB_ID}",
        json=MeetingArtifacts().model_dump(mode="json"),
    )
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_put_artifacts_returns_422_for_invalid_schema(client, mock_job_store):
    """Sending a body that violates the MeetingArtifacts schema yields 422."""
    mock_job_store.load.return_value = _completed_job()

    response = await client.put(
        f"/artifacts/{SAMPLE_JOB_ID}",
        json={"meeting_title": 12345, "user_stories": "not-a-list"},
    )
    assert response.status_code == 422


# GET /export/jira/{job_id}

@pytest.mark.asyncio
async def test_jira_export_succeeds(client, mock_job_store, sample_artifacts):
    mock_job_store.load.return_value = _completed_job(sample_artifacts)

    response = await client.get(f"/export/jira/{SAMPLE_JOB_ID}")
    assert response.status_code == 200

    body = response.json()
    assert "projects" in body
    assert len(body["projects"]) == 1
    assert len(body["projects"][0]["issues"]) > 0

    # Verify download header
    assert "attachment" in response.headers.get("content-disposition", "")


@pytest.mark.asyncio
async def test_jira_export_rejects_incomplete_job(client, mock_job_store):
    mock_job_store.load.return_value = _processing_job()

    response = await client.get(f"/export/jira/{SAMPLE_JOB_ID}")
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_jira_export_returns_404_for_missing_job(client, mock_job_store):
    mock_job_store.load.return_value = None

    response = await client.get(f"/export/jira/{SAMPLE_JOB_ID}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_jira_export_uses_cached_artifacts(client, mock_job_store, sample_artifacts):
    """When job.artifacts is None, falls back to cached artifacts."""
    mock_job_store.load.return_value = _completed_job(artifacts=None)
    mock_job_store.get_cached_artifacts.return_value = sample_artifacts

    response = await client.get(f"/export/jira/{SAMPLE_JOB_ID}")
    assert response.status_code == 200
    mock_job_store.get_cached_artifacts.assert_awaited()


@pytest.mark.asyncio
async def test_jira_export_404_when_no_artifacts(client, mock_job_store):
    """COMPLETED job with no artifacts in response or cache yields 404."""
    mock_job_store.load.return_value = _completed_job(artifacts=None)
    mock_job_store.get_cached_artifacts.return_value = None

    response = await client.get(f"/export/jira/{SAMPLE_JOB_ID}")
    assert response.status_code == 404


# GET /health

@pytest.mark.asyncio
async def test_health_endpoint(client, mock_db_session):
    """Health endpoint reports database status and job counts."""
    # Mock the DB query result for pending/processing counts
    mock_row = MagicMock()
    mock_row.pending = 2
    mock_row.processing = 1
    mock_result = MagicMock()
    mock_result.one.return_value = mock_row
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert "status" in body
    assert "components" in body


# GET /download/{job_id} — edge cases

@pytest.mark.asyncio
async def test_download_returns_404_for_missing_job(client, mock_job_store):
    mock_job_store.load.return_value = None
    response = await client.get(f"/download/{SAMPLE_JOB_ID}")
    assert response.status_code == 404


@pytest.mark.asyncio
async def test_download_returns_404_when_pdf_not_on_s3(
    client, mock_job_store, mock_db_session
):
    """Job is complete but no PDF S3 key exists."""
    mock_job_store.load.return_value = _completed_job()

    # DB returns no pdf_s3_key
    mock_result = MagicMock()
    mock_result.scalar_one_or_none.return_value = None
    mock_db_session.execute = AsyncMock(return_value=mock_result)

    response = await client.get(f"/download/{SAMPLE_JOB_ID}")
    assert response.status_code == 404


# POST /upload — size rejection

@pytest.mark.asyncio
async def test_upload_rejects_oversized_file(client):
    """Files exceeding MAX_UPLOAD_SIZE_MB are rejected with 400."""
    with patch("app.main.settings") as mock_settings:
        mock_settings.max_upload_size_mb = 0.0001  # ~100 bytes
        mock_settings.allowed_audio_extensions = [".wav"]
        mock_settings.upload_dir = "uploads"

        data = b"\x00" * 1024
        files = {"file": ("big.wav", BytesIO(data), "audio/wav")}
        response = await client.post("/upload", files=files)
        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()


# GET /artifacts/{job_id} — cache fallback

@pytest.mark.asyncio
async def test_get_artifacts_uses_cache_when_job_has_none(
    client, mock_job_store, sample_artifacts
):
    job = _completed_job(artifacts=None)
    mock_job_store.load.return_value = job
    mock_job_store.get_cached_artifacts.return_value = sample_artifacts

    response = await client.get(f"/artifacts/{SAMPLE_JOB_ID}")
    assert response.status_code == 200
    body = response.json()
    assert body["meeting_title"] == "Test Meeting"


@pytest.mark.asyncio
async def test_get_artifacts_404_when_no_cache(client, mock_job_store):
    job = _completed_job(artifacts=None)
    mock_job_store.load.return_value = job
    mock_job_store.get_cached_artifacts.return_value = None

    response = await client.get(f"/artifacts/{SAMPLE_JOB_ID}")
    assert response.status_code == 404
