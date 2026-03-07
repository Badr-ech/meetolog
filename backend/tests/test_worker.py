"""Tests for ARQ worker tasks: process_audio_job pipeline stages."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import pytest
import pytest_asyncio

from app.models import MeetingArtifacts, ProcessingStatus


SAMPLE_JOB_ID = "12345678-1234-5678-1234-567812345678"


@pytest.fixture
def mock_job_store():
    """Mock RedisJobStore with all methods used by the worker pipeline."""
    store = AsyncMock()
    store.update = AsyncMock(return_value=None)
    store.update_job_stage = AsyncMock(return_value=None)
    store.get_cached_transcript = AsyncMock(return_value=None)
    store.get_cached_artifacts = AsyncMock(return_value=None)
    store.cache_transcript = AsyncMock()
    store.cache_artifacts = AsyncMock()
    store.store_audio = AsyncMock(return_value=True)
    store.get_stored_audio = AsyncMock(return_value=None)
    store.delete_stored_audio = AsyncMock()
    store.get_completed_chunk_indices = AsyncMock(return_value=(set(), 0))
    store.save_chunk_transcript = AsyncMock()
    store.assemble_transcript_from_chunks = AsyncMock(return_value="Full transcript.")
    store.delete_chunk_data = AsyncMock()
    return store


@pytest.fixture
def worker_ctx(mock_job_store):
    """Minimal ARQ context dict injected into each task."""
    return {"job_store": mock_job_store, "redis": AsyncMock()}


@pytest.fixture
def tmp_audio(tmp_path) -> Path:
    """Create a temporary audio file on disk."""
    audio_file = tmp_path / "test_audio.wav"
    audio_file.write_bytes(b"\x00" * 2048)
    return audio_file


# process_audio_job â€” full pipeline (mocked services)

@pytest.mark.asyncio
async def test_process_audio_job_completes_pipeline(
    worker_ctx,
    mock_job_store,
    tmp_audio,
    sample_artifacts,
):
    """Verify the pipeline reaches COMPLETED and persists artifacts."""

    mock_transcriber = AsyncMock()
    mock_transcriber.transcribe = AsyncMock(return_value="Full transcript.")

    mock_llm = AsyncMock()
    mock_llm.extract_artifacts = AsyncMock(return_value=sample_artifacts)

    mock_pdf_service = AsyncMock()
    mock_pdf_service.generate = AsyncMock(return_value=Path("/tmp/meeting.pdf"))

    with (
        patch("app.worker.get_transcriber", return_value=mock_transcriber),
        patch("app.worker.get_llm_provider", return_value=mock_llm),
        patch("app.worker.get_pdf_service", return_value=mock_pdf_service),
        patch("app.services.transcription.compress_audio_for_storage", return_value=b"\x00" * 100),
    ):
        from app.worker import process_audio_job

        result = await process_audio_job(
            ctx=worker_ctx,
            job_id=SAMPLE_JOB_ID,
            file_path=str(tmp_audio),
            file_name="test_audio.wav",
            file_size=2048,
        )

    assert result["status"] == "completed"
    assert result["job_id"] == SAMPLE_JOB_ID

    completion_calls = [
        c
        for c in mock_job_store.update_job_stage.call_args_list
        if c.kwargs.get("status") == ProcessingStatus.COMPLETED
    ]
    assert len(completion_calls) == 1

    mock_llm.extract_artifacts.assert_awaited_once()
    mock_pdf_service.generate.assert_awaited_once()


@pytest.mark.asyncio
async def test_process_audio_job_uses_cached_transcript(
    worker_ctx, mock_job_store, tmp_audio, sample_artifacts
):
    """When a cached transcript exists, transcription is skipped."""
    mock_job_store.get_cached_transcript.return_value = "Previously cached transcript"

    mock_llm = AsyncMock()
    mock_llm.extract_artifacts = AsyncMock(return_value=sample_artifacts)

    mock_pdf_service = AsyncMock()
    mock_pdf_service.generate = AsyncMock(return_value=Path("/tmp/meeting.pdf"))

    with (
        patch("app.worker.get_transcriber") as mock_get_transcriber,
        patch("app.worker.get_llm_provider", return_value=mock_llm),
        patch("app.worker.get_pdf_service", return_value=mock_pdf_service),
    ):
        from app.worker import process_audio_job

        result = await process_audio_job(
            ctx=worker_ctx,
            job_id=SAMPLE_JOB_ID,
            file_path=str(tmp_audio),
            file_name="test_audio.wav",
            file_size=2048,
        )

    assert result["status"] == "completed"
    mock_get_transcriber.assert_not_called()


@pytest.mark.asyncio
async def test_process_audio_job_uses_cached_artifacts(
    worker_ctx, mock_job_store, tmp_audio, sample_artifacts
):
    """When cached artifacts exist, LLM extraction is skipped."""
    mock_job_store.get_cached_transcript.return_value = "Cached transcript"
    mock_job_store.get_cached_artifacts.return_value = sample_artifacts

    mock_pdf_service = AsyncMock()
    mock_pdf_service.generate = AsyncMock(return_value=Path("/tmp/meeting.pdf"))

    with (
        patch("app.worker.get_llm_provider") as mock_get_llm,
        patch("app.worker.get_pdf_service", return_value=mock_pdf_service),
    ):
        from app.worker import process_audio_job

        result = await process_audio_job(
            ctx=worker_ctx,
            job_id=SAMPLE_JOB_ID,
            file_path=str(tmp_audio),
            file_name="test_audio.wav",
            file_size=2048,
        )

    assert result["status"] == "completed"
    mock_get_llm.assert_not_called()


@pytest.mark.asyncio
async def test_process_audio_job_handles_missing_file(
    worker_ctx, mock_job_store
):
    """Pipeline should set FAILED when audio file does not exist and has no Redis backup."""
    mock_job_store.get_cached_transcript.return_value = None
    mock_job_store.get_stored_audio.return_value = None

    from app.worker import process_audio_job

    result = await process_audio_job(
        ctx=worker_ctx,
        job_id=SAMPLE_JOB_ID,
        file_path="/nonexistent/path/audio.wav",
        file_name="audio.wav",
        file_size=0,
    )

    assert result["status"] == "failed"

    failure_calls = [
        c
        for c in mock_job_store.update_job_stage.call_args_list
        if c.kwargs.get("status") == ProcessingStatus.FAILED
    ]
    assert len(failure_calls) >= 1


@pytest.mark.asyncio
async def test_process_audio_job_handles_llm_error(
    worker_ctx, mock_job_store, tmp_audio
):
    """Pipeline should set FAILED when LLM extraction raises."""
    mock_transcriber = AsyncMock()
    mock_transcriber.transcribe = AsyncMock(return_value="Full transcript.")

    mock_llm = AsyncMock()
    mock_llm.extract_artifacts = AsyncMock(side_effect=RuntimeError("API quota exceeded"))

    with (
        patch("app.worker.get_transcriber", return_value=mock_transcriber),
        patch("app.worker.get_llm_provider", return_value=mock_llm),
        patch("app.services.transcription.compress_audio_for_storage", return_value=b"\x00"),
    ):
        from app.worker import process_audio_job

        result = await process_audio_job(
            ctx=worker_ctx,
            job_id=SAMPLE_JOB_ID,
            file_path=str(tmp_audio),
            file_name="test.wav",
            file_size=2048,
        )

    assert result["status"] == "failed"
    assert "API quota exceeded" in result["error"]


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
