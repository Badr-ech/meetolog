"""Tests for Pydantic models, ProcessingStatus parsing, and schema edge cases."""

import pytest
from datetime import datetime
from uuid import uuid4

from pydantic import ValidationError

from app.models import (
    ActionableTask,
    ActionItem,
    Blocker,
    Decision,
    JobResponse,
    MeetingArtifacts,
    Priority,
    ProcessingStatus,
    Task,
    TaskStatus,
    UserStory,
    parse_processing_status,
)


# parse_processing_status â€” legacy mapping

class TestParseProcessingStatus:
    @pytest.mark.parametrize("value,expected", [
        ("uploading", ProcessingStatus.UPLOADING),
        ("transcribing", ProcessingStatus.TRANSCRIBING),
        ("extracting", ProcessingStatus.EXTRACTING),
        ("generating_pdf", ProcessingStatus.GENERATING_PDF),
        ("completed", ProcessingStatus.COMPLETED),
        ("failed", ProcessingStatus.FAILED),
    ])
    def test_current_status_values(self, value, expected):
        assert parse_processing_status(value) == expected

    def test_legacy_pending_maps_to_uploading(self):
        assert parse_processing_status("pending") == ProcessingStatus.UPLOADING

    def test_legacy_processing_maps_to_transcribing(self):
        assert parse_processing_status("processing") == ProcessingStatus.TRANSCRIBING

    def test_unknown_value_falls_back_to_uploading(self):
        assert parse_processing_status("garbage") == ProcessingStatus.UPLOADING

    def test_empty_string_falls_back_to_uploading(self):
        assert parse_processing_status("") == ProcessingStatus.UPLOADING


# confidence_score validation

class TestConfidenceScoreValidation:
    def test_valid_score_accepted(self):
        task = Task(title="Test", confidence_score=0.75)
        assert task.confidence_score == 0.75

    def test_zero_score_accepted(self):
        task = Task(title="Test", confidence_score=0.0)
        assert task.confidence_score == 0.0

    def test_one_score_accepted(self):
        task = Task(title="Test", confidence_score=1.0)
        assert task.confidence_score == 1.0

    def test_none_score_accepted(self):
        task = Task(title="Test", confidence_score=None)
        assert task.confidence_score is None

    def test_score_above_one_rejected(self):
        with pytest.raises(ValidationError):
            Task(title="Test", confidence_score=1.5)

    def test_negative_score_rejected(self):
        with pytest.raises(ValidationError):
            Task(title="Test", confidence_score=-0.1)

    def test_score_applies_to_all_artifact_types(self):
        """confidence_score validation is consistent across all artifact models."""
        for cls, kwargs in [
            (UserStory, {"title": "T", "as_a": "u", "i_want": "w", "so_that": "s"}),
            (Task, {"title": "T"}),
            (Decision, {"title": "T", "description": "D"}),
            (Blocker, {"title": "T", "description": "D"}),
            (ActionItem, {"description": "D"}),
            (ActionableTask, {"title": "T", "description": "D", "owner_role": "R", "task_source": "Explicit"}),
        ]:
            obj = cls(**kwargs, confidence_score=0.5)
            assert obj.confidence_score == 0.5


# MeetingArtifacts defaults and edge cases

class TestMeetingArtifacts:
    def test_defaults(self):
        a = MeetingArtifacts()
        assert a.meeting_title == "Untitled Meeting"
        assert a.participants == []
        assert a.user_stories == []
        assert a.execution_tasks == []
        assert a.transcript == ""

    def test_empty_transcript_allowed(self):
        a = MeetingArtifacts(transcript="")
        assert a.transcript == ""

    def test_all_list_fields_accept_empty(self):
        a = MeetingArtifacts(
            user_stories=[],
            tasks=[],
            decisions=[],
            blockers=[],
            action_items=[],
            execution_tasks=[],
        )
        assert len(a.user_stories) == 0


# ActionableTask â€” Literal fields

class TestActionableTask:
    def test_valid_explicit_source(self):
        at = ActionableTask(
            title="T", description="D", owner_role="Eng", task_source="Explicit"
        )
        assert at.task_source == "Explicit"

    def test_valid_inferred_source(self):
        at = ActionableTask(
            title="T", description="D", owner_role="Eng", task_source="Inferred"
        )
        assert at.task_source == "Inferred"

    def test_invalid_source_rejected(self):
        with pytest.raises(ValidationError):
            ActionableTask(
                title="T", description="D", owner_role="Eng", task_source="Maybe"
            )

    def test_valid_priority_values(self):
        for p in ("High", "Medium", "Low"):
            at = ActionableTask(
                title="T", description="D", owner_role="Eng",
                task_source="Explicit", priority=p,
            )
            assert at.priority == p

    def test_invalid_priority_rejected(self):
        with pytest.raises(ValidationError):
            ActionableTask(
                title="T", description="D", owner_role="Eng",
                task_source="Explicit", priority="Critical",
            )


# JobResponse

class TestJobResponse:
    def test_progress_clamped_low(self):
        with pytest.raises(ValidationError):
            JobResponse(
                job_id=uuid4(),
                status=ProcessingStatus.UPLOADING,
                progress=-1,
            )

    def test_progress_clamped_high(self):
        with pytest.raises(ValidationError):
            JobResponse(
                job_id=uuid4(),
                status=ProcessingStatus.UPLOADING,
                progress=101,
            )

    def test_valid_job_response(self):
        jr = JobResponse(
            job_id=uuid4(),
            status=ProcessingStatus.COMPLETED,
            progress=100,
            message="Done",
        )
        assert jr.progress == 100
