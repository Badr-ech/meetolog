"""Tests for app.models.artifacts — parsing, sanitization, validation, and domain conversion."""

import json

import pytest
from pydantic import ValidationError

from app.models.artifacts import (
    LLMExtractionResponse,
    LLMTask,
    LLMActionItem,
    sanitize_json_string,
    strip_markdown_fencing,
    to_meeting_artifacts,
    validate_llm_response,
)
from app.models.schemas import Priority


# ---------------------------------------------------------------------------
# strip_markdown_fencing
# ---------------------------------------------------------------------------

class TestStripMarkdownFencing:
    def test_removes_json_fence(self):
        text = '```json\n{"key": "value"}\n```'
        assert strip_markdown_fencing(text) == '{"key": "value"}'

    def test_removes_plain_fence(self):
        text = '```\n{"key": "value"}\n```'
        assert strip_markdown_fencing(text) == '{"key": "value"}'

    def test_no_fence_passthrough(self):
        text = '{"key": "value"}'
        assert strip_markdown_fencing(text) == '{"key": "value"}'

    def test_strips_surrounding_whitespace(self):
        text = '  ```json\n  {"a": 1}  \n```  '
        result = strip_markdown_fencing(text)
        assert '"a"' in result


# ---------------------------------------------------------------------------
# sanitize_json_string
# ---------------------------------------------------------------------------

class TestSanitizeJsonString:
    def test_removes_trailing_commas(self):
        text = '{"items": [1, 2, 3,], "key": "val",}'
        result = sanitize_json_string(text)
        # Should be valid JSON after sanitization
        parsed = json.loads(result)
        assert parsed["items"] == [1, 2, 3]
        assert parsed["key"] == "val"

    def test_strips_markdown_and_trailing_commas(self):
        text = '```json\n{"a": 1,}\n```'
        result = sanitize_json_string(text)
        parsed = json.loads(result)
        assert parsed["a"] == 1

    def test_strips_preamble_text(self):
        text = 'Here is the JSON output:\n{"meeting_title": "Test"}'
        result = sanitize_json_string(text)
        parsed = json.loads(result)
        assert parsed["meeting_title"] == "Test"

    def test_strips_postamble_text(self):
        text = '{"meeting_title": "Test"}\nI hope this helps!'
        result = sanitize_json_string(text)
        parsed = json.loads(result)
        assert parsed["meeting_title"] == "Test"

    def test_empty_string(self):
        result = sanitize_json_string("")
        assert result == ""


# ---------------------------------------------------------------------------
# validate_llm_response
# ---------------------------------------------------------------------------

class TestValidateLLMResponse:
    def test_valid_json_parsed(self):
        raw = json.dumps({
            "meeting_title": "Sprint Planning",
            "summary": "Discussed tasks.",
            "participants": ["Alice"],
            "tasks": [{"title": "Build API", "description": "JWT"}],
        })
        result = validate_llm_response(raw)
        assert isinstance(result, LLMExtractionResponse)
        assert result.meeting_title == "Sprint Planning"
        assert len(result.tasks) == 1

    def test_fenced_json_parsed(self):
        payload = {
            "meeting_title": "Standup",
            "summary": "Quick sync.",
            "participants": [],
        }
        raw = f"```json\n{json.dumps(payload)}\n```"
        result = validate_llm_response(raw)
        assert result.meeting_title == "Standup"

    def test_trailing_comma_repaired(self):
        raw = '{"meeting_title": "Test", "participants": ["Alice",],}'
        result = validate_llm_response(raw)
        assert result.meeting_title == "Test"

    def test_empty_object_uses_defaults(self):
        result = validate_llm_response("{}")
        assert result.meeting_title == "Untitled Meeting"
        assert result.summary == ""
        assert result.participants == []

    def test_invalid_json_raises_value_error(self):
        with pytest.raises((ValueError, ValidationError)):
            validate_llm_response("not json at all without braces")

    def test_preamble_stripped_before_parse(self):
        raw = 'Sure! Here is the extraction:\n{"meeting_title": "Retro"}\nLet me know if you need more.'
        result = validate_llm_response(raw)
        assert result.meeting_title == "Retro"

    def test_extra_fields_ignored(self):
        raw = json.dumps({
            "meeting_title": "Test",
            "unknown_field": "should be ignored",
        })
        result = validate_llm_response(raw)
        assert result.meeting_title == "Test"


# ---------------------------------------------------------------------------
# LLM model null priority handling
# ---------------------------------------------------------------------------

class TestLLMNullPriority:
    def test_llm_task_null_priority_defaults(self):
        task = LLMTask.model_validate({"title": "Test", "priority": None})
        assert task.priority is None  # stored as None

    def test_llm_task_missing_priority_defaults(self):
        task = LLMTask.model_validate({"title": "Test"})
        assert task.priority == "medium"

    def test_llm_action_item_null_priority_defaults(self):
        item = LLMActionItem.model_validate({"title": "Test", "priority": None})
        assert item.priority is None

    def test_llm_action_item_missing_priority_defaults(self):
        item = LLMActionItem.model_validate({"title": "Test"})
        assert item.priority == "medium"


# ---------------------------------------------------------------------------
# Confidence clamping in LLM models
# ---------------------------------------------------------------------------

class TestLLMConfidenceClamping:
    def test_score_above_one_clamped(self):
        task = LLMTask.model_validate({"title": "T", "confidence_score": 1.5})
        assert task.confidence_score == 1.0

    def test_score_below_zero_clamped(self):
        task = LLMTask.model_validate({"title": "T", "confidence_score": -0.3})
        assert task.confidence_score == 0.0

    def test_valid_score_unchanged(self):
        task = LLMTask.model_validate({"title": "T", "confidence_score": 0.85})
        assert task.confidence_score == 0.85

    def test_null_score_remains_none(self):
        task = LLMTask.model_validate({"title": "T", "confidence_score": None})
        assert task.confidence_score is None


# ---------------------------------------------------------------------------
# to_meeting_artifacts
# ---------------------------------------------------------------------------

class TestToMeetingArtifacts:
    def test_basic_conversion(self):
        response = LLMExtractionResponse.model_validate({
            "meeting_title": "Sprint Planning",
            "summary": "Planned sprint work.",
            "participants": ["Alice", "Bob"],
            "tasks": [{"title": "Build API", "description": "REST endpoints", "priority": "high"}],
            "decisions": [{"title": "Use Postgres", "description": "For persistence", "made_by": "Alice"}],
            "blockers": [{"title": "Waiting on infra", "description": "Need VPC setup"}],
            "action_items": [{"description": "Update docs", "assignee": "Bob", "priority": "low"}],
            "ideas": [{"idea_description": "Add caching", "proposed_by": "Alice"}],
            "user_stories": [{"title": "Login", "as_a": "user", "i_want": "to log in", "so_that": "secure"}],
        })
        artifacts = to_meeting_artifacts(response, transcript="Sample transcript.")

        assert artifacts.meeting_title == "Sprint Planning"
        assert len(artifacts.tasks) == 1
        assert artifacts.tasks[0].priority == Priority.HIGH
        assert len(artifacts.decisions) == 1
        assert len(artifacts.blockers) == 1
        assert len(artifacts.action_items) == 1
        assert artifacts.action_items[0].priority == Priority.LOW
        assert len(artifacts.ideas) == 1
        assert len(artifacts.user_stories) == 1
        assert artifacts.transcript == "Sample transcript."

    def test_null_priority_maps_to_medium(self):
        response = LLMExtractionResponse.model_validate({
            "tasks": [{"title": "Task", "description": "D", "priority": None}],
            "action_items": [{"description": "AI", "priority": None}],
        })
        artifacts = to_meeting_artifacts(response, transcript="")
        assert artifacts.tasks[0].priority == Priority.MEDIUM
        assert artifacts.action_items[0].priority == Priority.MEDIUM

    def test_unknown_priority_maps_to_medium(self):
        response = LLMExtractionResponse.model_validate({
            "tasks": [{"title": "T", "description": "D", "priority": "urgent"}],
        })
        artifacts = to_meeting_artifacts(response, transcript="")
        assert artifacts.tasks[0].priority == Priority.MEDIUM

    def test_empty_response_produces_empty_artifacts(self):
        response = LLMExtractionResponse.model_validate({})
        artifacts = to_meeting_artifacts(response, transcript="")
        assert artifacts.meeting_title == "Untitled Meeting"
        assert artifacts.tasks == []
        assert artifacts.decisions == []
        assert artifacts.user_stories == []
