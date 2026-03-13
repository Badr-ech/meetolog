"""Tests for app.services.mock_services — deterministic mock behaviour."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID

import pytest

from app.services.mock_services import MockTranscriber, MockExtractor
from app.models import MeetingArtifacts, Priority


# ---------------------------------------------------------------------------
# MockTranscriber
# ---------------------------------------------------------------------------

class TestMockTranscriber:
    """MockTranscriber contract compliance."""

    @pytest.fixture()
    def transcriber(self) -> MockTranscriber:
        return MockTranscriber(simulated_delay=0.0)

    async def test_returns_string(self, transcriber: MockTranscriber, tmp_path: Path):
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"\x00")
        result = await transcriber.transcribe(audio)
        assert isinstance(result, str)
        assert len(result) > 0

    async def test_transcript_mentions_participants(self, transcriber: MockTranscriber, tmp_path: Path):
        audio = tmp_path / "test.mp3"
        audio.write_bytes(b"\x00")
        result = await transcriber.transcribe(audio)
        assert "Sarah" in result
        assert "Mike" in result

    async def test_missing_file_raises(self, transcriber: MockTranscriber):
        with pytest.raises(FileNotFoundError):
            await transcriber.transcribe(Path("/nonexistent/audio.wav"))

    async def test_preprocess_removes_blank_lines(self, transcriber: MockTranscriber):
        raw = "Line one.\n\n\nLine two.\n   \n"
        cleaned = await transcriber.preprocess_transcript(raw)
        assert "\n\n" not in cleaned
        assert "Line one." in cleaned
        assert "Line two." in cleaned

    async def test_preprocess_collapses_whitespace(self, transcriber: MockTranscriber):
        raw = "too   many   spaces"
        cleaned = await transcriber.preprocess_transcript(raw)
        assert cleaned == "too many spaces"

    async def test_transcribe_chunk_delegates(self, transcriber: MockTranscriber, tmp_path: Path):
        audio = tmp_path / "chunk.wav"
        audio.write_bytes(b"\x00")
        result = await transcriber.transcribe_chunk(audio)
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# MockExtractor
# ---------------------------------------------------------------------------

class TestMockExtractor:
    """MockExtractor returns valid, deterministic MeetingArtifacts."""

    @pytest.fixture()
    def extractor(self) -> MockExtractor:
        return MockExtractor(simulated_delay=0.0)

    async def test_returns_meeting_artifacts(self, extractor: MockExtractor):
        result = await extractor.extract_artifacts("test transcript")
        assert isinstance(result, MeetingArtifacts)

    async def test_meeting_title_present(self, extractor: MockExtractor):
        result = await extractor.extract_artifacts("test")
        assert result.meeting_title
        assert isinstance(result.meeting_title, str)

    async def test_participants_non_empty(self, extractor: MockExtractor):
        result = await extractor.extract_artifacts("test")
        assert len(result.participants) > 0

    async def test_user_stories_present(self, extractor: MockExtractor):
        result = await extractor.extract_artifacts("test")
        assert len(result.user_stories) > 0
        story = result.user_stories[0]
        assert story.title
        assert story.as_a
        assert story.i_want
        assert story.so_that

    async def test_tasks_present(self, extractor: MockExtractor):
        result = await extractor.extract_artifacts("test")
        assert len(result.tasks) > 0
        task = result.tasks[0]
        assert task.title
        assert task.assignee

    async def test_decisions_present(self, extractor: MockExtractor):
        result = await extractor.extract_artifacts("test")
        assert len(result.decisions) > 0

    async def test_blockers_present(self, extractor: MockExtractor):
        result = await extractor.extract_artifacts("test")
        assert len(result.blockers) > 0
        blocker = result.blockers[0]
        assert blocker.title
        assert blocker.owner

    async def test_action_items_present(self, extractor: MockExtractor):
        result = await extractor.extract_artifacts("test")
        assert len(result.action_items) > 0

    async def test_ideas_present(self, extractor: MockExtractor):
        result = await extractor.extract_artifacts("test")
        assert len(result.ideas) > 0

    async def test_execution_tasks_present(self, extractor: MockExtractor):
        result = await extractor.extract_artifacts("test")
        assert len(result.execution_tasks) > 0

    async def test_is_mock_true(self, extractor: MockExtractor):
        assert extractor.is_mock is True

    async def test_provider_name(self, extractor: MockExtractor):
        assert extractor.provider_name == "Mock"

    async def test_generate_text_returns_summary(self, extractor: MockExtractor):
        result = await extractor.generate_text("tell me about testing")
        assert isinstance(result, str)
        assert "Mock summary" in result

    async def test_priorities_are_valid(self, extractor: MockExtractor):
        result = await extractor.extract_artifacts("test")
        for task in result.tasks:
            assert isinstance(task.priority, Priority)
        for story in result.user_stories:
            assert isinstance(story.priority, Priority)

    async def test_deterministic(self, extractor: MockExtractor):
        """Two calls produce identical structure."""
        r1 = await extractor.extract_artifacts("test")
        r2 = await extractor.extract_artifacts("test")
        assert r1.meeting_title == r2.meeting_title
        assert len(r1.tasks) == len(r2.tasks)
        assert len(r1.user_stories) == len(r2.user_stories)
