"""Tests for PostgreSQL worker pipeline: process_job stages."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest

from app.models import MeetingArtifacts, ProcessingStatus
from app.worker import process_job


SAMPLE_JOB_ID = UUID("12345678-1234-5678-1234-567812345678")
SAMPLE_S3_KEY = "uploads/12345678-1234-5678-1234-567812345678.wav"


@pytest.fixture
def mock_queue():
    """Mock PostgresJobQueue with all methods used by process_job."""
    queue = AsyncMock()
    queue.update_job_progress = AsyncMock()
    queue.mark_job_completed = AsyncMock()
    queue.mark_job_failed = AsyncMock()
    return queue


@pytest.fixture
def mock_s3_service(tmp_path):
    """Mock S3StorageService that writes a dummy audio file on download."""
    s3 = AsyncMock()

    async def _fake_download(s3_key: str, dest_path: str) -> None:
        Path(dest_path).write_bytes(b"\x00" * 2048)

    s3.download_to_file = AsyncMock(side_effect=_fake_download)
    s3.upload_pdf = AsyncMock(return_value="results/test.pdf")
    s3.upload_artifacts_json = AsyncMock(return_value="results/test.json")
    return s3


# process_job — full pipeline (mocked services)

@pytest.mark.asyncio
async def test_process_job_completes_pipeline(
    mock_queue, mock_s3_service, sample_artifacts
):
    """Verify the pipeline reaches COMPLETED and persists artifacts."""
    mock_transcriber = AsyncMock()
    mock_transcriber.transcribe = AsyncMock(return_value="Full transcript.")

    mock_extractor = AsyncMock()
    mock_extractor.extract_artifacts = AsyncMock(return_value=sample_artifacts)

    mock_pdf_service = AsyncMock()
    mock_pdf_service.generate = AsyncMock(return_value=Path("/tmp/meeting.pdf"))

    with (
        patch("app.worker._get_transcriber", return_value=mock_transcriber),
        patch("app.worker._get_llm_provider", return_value=mock_extractor),
        patch("app.worker._get_pdf_service", return_value=mock_pdf_service),
    ):
        await process_job(
            job_id=SAMPLE_JOB_ID,
            s3_key=SAMPLE_S3_KEY,
            file_name="test_audio.wav",
            queue=mock_queue,
            s3_service=mock_s3_service,
            worker_id="test-worker",
        )

    mock_queue.mark_job_completed.assert_awaited_once()
    mock_extractor.extract_artifacts.assert_awaited_once()
    mock_pdf_service.generate.assert_awaited_once()
    mock_s3_service.upload_pdf.assert_awaited_once()
    mock_s3_service.upload_artifacts_json.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_job_handles_s3_download_failure(mock_queue, mock_s3_service):
    """Pipeline should mark FAILED when S3 download raises FileNotFoundError."""
    mock_s3_service.download_to_file = AsyncMock(
        side_effect=FileNotFoundError("S3 key not found")
    )

    await process_job(
        job_id=SAMPLE_JOB_ID,
        s3_key=SAMPLE_S3_KEY,
        file_name="test_audio.wav",
        queue=mock_queue,
        s3_service=mock_s3_service,
        worker_id="test-worker",
    )

    mock_queue.mark_job_failed.assert_awaited_once()
    error_msg = mock_queue.mark_job_failed.call_args[0][1]
    assert "not found" in error_msg.lower()


@pytest.mark.asyncio
async def test_process_job_handles_llm_error(mock_queue, mock_s3_service):
    """Pipeline should mark FAILED when LLM extraction raises."""
    mock_transcriber = AsyncMock()
    mock_transcriber.transcribe = AsyncMock(return_value="Full transcript.")

    mock_extractor = AsyncMock()
    mock_extractor.extract_artifacts = AsyncMock(
        side_effect=RuntimeError("API quota exceeded")
    )

    with (
        patch("app.worker._get_transcriber", return_value=mock_transcriber),
        patch("app.worker._get_llm_provider", return_value=mock_extractor),
    ):
        await process_job(
            job_id=SAMPLE_JOB_ID,
            s3_key=SAMPLE_S3_KEY,
            file_name="test.wav",
            queue=mock_queue,
            s3_service=mock_s3_service,
            worker_id="test-worker",
        )

    mock_queue.mark_job_failed.assert_awaited_once()
    error_msg = mock_queue.mark_job_failed.call_args[0][1]
    assert "API quota exceeded" in error_msg


@pytest.mark.asyncio
async def test_process_job_reports_progress_stages(mock_queue, mock_s3_service, sample_artifacts):
    """Pipeline reports progress through transcription → extraction → PDF stages."""
    mock_transcriber = AsyncMock()
    mock_transcriber.transcribe = AsyncMock(return_value="Transcript.")

    mock_extractor = AsyncMock()
    mock_extractor.extract_artifacts = AsyncMock(return_value=sample_artifacts)

    mock_pdf_service = AsyncMock()
    mock_pdf_service.generate = AsyncMock(return_value=Path("/tmp/meeting.pdf"))

    with (
        patch("app.worker._get_transcriber", return_value=mock_transcriber),
        patch("app.worker._get_llm_provider", return_value=mock_extractor),
        patch("app.worker._get_pdf_service", return_value=mock_pdf_service),
    ):
        await process_job(
            job_id=SAMPLE_JOB_ID,
            s3_key=SAMPLE_S3_KEY,
            file_name="test.wav",
            queue=mock_queue,
            s3_service=mock_s3_service,
        )

    # Should have multiple progress updates
    assert mock_queue.update_job_progress.await_count >= 3


# LLM response parsing (GeminiProvider._parse_extraction)

@pytest.mark.asyncio
async def test_gemini_parse_extraction_produces_valid_artifacts(
    deterministic_llm_response,
):
    """Verify _parse_extraction builds correct MeetingArtifacts from raw JSON."""
    from app.services.llm_engine import GeminiProvider

    with patch.object(GeminiProvider, "__init__", lambda self, **kw: None):
        provider = GeminiProvider.__new__(GeminiProvider)

    artifacts = provider._parse_extraction(
        deterministic_llm_response, "Raw transcript text"
    )

    assert isinstance(artifacts, MeetingArtifacts)
    assert artifacts.meeting_title == "Sprint Planning - Auth"
    assert len(artifacts.user_stories) == 1
    assert artifacts.user_stories[0].story_points == 5
    assert len(artifacts.tasks) == 1
    assert artifacts.tasks[0].assignee == "Mike"
    assert len(artifacts.action_items) == 1
    assert artifacts.transcript == "Raw transcript text"


@pytest.mark.asyncio
async def test_gemini_parse_handles_empty_categories(deterministic_llm_response):
    """Parser should handle missing or empty optional categories gracefully."""
    from app.services.llm_engine import GeminiProvider

    minimal = {
        "meeting_title": "Quick Sync",
        "summary": "Brief sync.",
        "participants": [],
        "user_stories": [],
        "tasks": [],
        "decisions": [],
        "blockers": [],
        "action_items": [],
    }

    with patch.object(GeminiProvider, "__init__", lambda self, **kw: None):
        provider = GeminiProvider.__new__(GeminiProvider)

    artifacts = provider._parse_extraction(minimal, "")

    assert artifacts.meeting_title == "Quick Sync"
    assert len(artifacts.user_stories) == 0
    assert len(artifacts.tasks) == 0
    assert len(artifacts.blockers) == 0
