"""Tests for LLM engine: JSON cleaning, parsing, factory, and provider init."""

import json
import pytest
from unittest.mock import patch, MagicMock

from app.services.llm_engine import (
    GeminiProvider,
    OpenAIProvider,
    LLMProvider,
    get_llm_provider,
)
from app.models import MeetingArtifacts


# _clean_json_response

class TestCleanJsonResponse:
    """Verifies markdown code-block stripping from LLM output."""

    def _make_provider(self) -> LLMProvider:
        with patch.object(GeminiProvider, "__init__", lambda self, **kw: None):
            return GeminiProvider.__new__(GeminiProvider)

    def test_plain_json_unchanged(self):
        p = self._make_provider()
        raw = '{"meeting_title": "Sync"}'
        assert p._clean_json_response(raw) == raw

    def test_strips_json_code_block(self):
        p = self._make_provider()
        raw = '```json\n{"meeting_title": "Sync"}\n```'
        result = p._clean_json_response(raw)
        assert result == '{"meeting_title": "Sync"}'

    def test_strips_plain_code_block(self):
        p = self._make_provider()
        raw = '```\n{"key": "val"}\n```'
        result = p._clean_json_response(raw)
        assert result == '{"key": "val"}'

    def test_whitespace_trimmed(self):
        p = self._make_provider()
        raw = '  {"key": "val"}  '
        assert p._clean_json_response(raw) == '{"key": "val"}'


# _parse_extraction â€” edge cases

class TestParseExtraction:
    def _make_provider(self) -> LLMProvider:
        with patch.object(GeminiProvider, "__init__", lambda self, **kw: None):
            return GeminiProvider.__new__(GeminiProvider)

    def test_missing_optional_fields(self):
        """Parser handles missing fields gracefully with defaults."""
        p = self._make_provider()
        data = {
            "meeting_title": "Quick Sync",
            "summary": "Brief.",
            "participants": [],
            "user_stories": [],
            "tasks": [],
            "decisions": [],
            "blockers": [],
            "action_items": [],
        }
        artifacts = p._parse_extraction(data, "transcript")
        assert artifacts.meeting_title == "Quick Sync"
        assert artifacts.transcript == "transcript"

    def test_malformed_priority_defaults_to_medium(self):
        """Unknown priority strings default to MEDIUM, not crash."""
        p = self._make_provider()
        data = {
            "meeting_title": "Meeting",
            "summary": "",
            "participants": [],
            "user_stories": [{
                "title": "Story",
                "as_a": "u",
                "i_want": "w",
                "so_that": "s",
                "acceptance_criteria": [],
                "priority": "ASAP",
                "story_points": None,
            }],
            "tasks": [],
            "decisions": [],
            "blockers": [],
            "action_items": [],
        }
        artifacts = p._parse_extraction(data, "")
        from app.models import Priority
        assert artifacts.user_stories[0].priority == Priority.MEDIUM

    def test_entirely_empty_response(self):
        """Completely empty LLM response dict produces valid defaults."""
        p = self._make_provider()
        artifacts = p._parse_extraction({}, "")
        assert isinstance(artifacts, MeetingArtifacts)
        assert artifacts.meeting_title == "Untitled Meeting"

    def test_null_assignees_handled(self):
        p = self._make_provider()
        data = {
            "meeting_title": "M",
            "summary": "",
            "participants": [],
            "user_stories": [],
            "tasks": [{"title": "Task", "description": "", "assignee": None, "priority": None, "due_date": None}],
            "decisions": [],
            "blockers": [],
            "action_items": [{"description": "Do thing", "assignee": None, "due_date": None}],
        }
        artifacts = p._parse_extraction(data, "")
        assert artifacts.tasks[0].assignee is None
        assert artifacts.action_items[0].assignee is None


# get_llm_provider â€” factory

class TestGetLlmProvider:
    def test_test_mode_returns_mock(self):
        settings = MagicMock()
        settings.test_mode = True
        provider = get_llm_provider(settings)
        assert provider.is_mock is True

    def test_gemini_without_key_returns_mock(self):
        settings = MagicMock()
        settings.test_mode = False
        settings.llm_provider = "gemini"
        settings.gemini_api_key = ""
        provider = get_llm_provider(settings)
        assert provider.is_mock is True

    def test_openai_without_key_returns_mock(self):
        settings = MagicMock()
        settings.test_mode = False
        settings.llm_provider = "openai"
        settings.openai_api_key = ""
        provider = get_llm_provider(settings)
        assert provider.is_mock is True

    def test_gemini_with_key_returns_gemini(self):
        settings = MagicMock()
        settings.test_mode = False
        settings.llm_provider = "gemini"
        settings.gemini_api_key = "test-key-123"
        provider = get_llm_provider(settings)
        assert isinstance(provider, GeminiProvider)
        assert provider.is_mock is False

    def test_openai_with_key_returns_openai(self):
        settings = MagicMock()
        settings.test_mode = False
        settings.llm_provider = "openai"
        settings.openai_api_key = "sk-test-123"
        provider = get_llm_provider(settings)
        assert isinstance(provider, OpenAIProvider)
        assert provider.is_mock is False


# Provider initialization validation

class TestProviderInit:
    def test_gemini_rejects_empty_key(self):
        with pytest.raises(ValueError, match="GEMINI_API_KEY"):
            GeminiProvider(api_key="")

    def test_openai_rejects_empty_key(self):
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIProvider(api_key="")
