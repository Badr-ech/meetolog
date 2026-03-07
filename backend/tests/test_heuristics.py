"""Tests for deterministic confidence scoring engine (heuristics.py)."""

import pytest

from app.services.heuristics import (
    calculate_artifact_confidence,
    backfill_confidence_scores,
    _field_populated,
)
from app.models import (
    MeetingArtifacts,
    UserStory,
    Task,
    Decision,
    Blocker,
    ActionItem,
    ActionableTask,
    Priority,
    TaskStatus,
)


# _field_populated helper

class TestFieldPopulated:
    def test_none_returns_false(self):
        assert _field_populated(None) is False

    def test_empty_string_returns_false(self):
        assert _field_populated("") is False

    def test_whitespace_only_returns_false(self):
        assert _field_populated("   ") is False

    def test_empty_list_returns_false(self):
        assert _field_populated([]) is False

    def test_nonempty_string_returns_true(self):
        assert _field_populated("hello") is True

    def test_nonempty_list_returns_true(self):
        assert _field_populated(["a"]) is True

    def test_integer_returns_true(self):
        assert _field_populated(0) is True

    def test_boolean_false_returns_true(self):
        assert _field_populated(False) is True


# calculate_artifact_confidence â€” base scoring

class TestCalculateArtifactConfidence:
    def test_minimal_artifact_gets_base_score(self):
        """Artifact with no meaningful fields scores 0.2 (base only)."""
        score = calculate_artifact_confidence(
            {"title": "", "description": ""},
            "task",
        )
        assert score == 0.2

    def test_owner_bonus(self):
        """Assignee present adds +0.2."""
        score = calculate_artifact_confidence(
            {"title": "", "description": "", "assignee": "Alice"},
            "task",
        )
        assert score >= 0.4

    def test_priority_bonus(self):
        """Explicit priority adds +0.2."""
        score = calculate_artifact_confidence(
            {"title": "", "description": "", "priority": "high"},
            "task",
        )
        assert score >= 0.4

    def test_action_verb_bonus(self):
        """Strong action verb in title adds +0.2."""
        score = calculate_artifact_confidence(
            {"title": "Implement auth API", "description": ""},
            "task",
        )
        assert score >= 0.4

    def test_ambiguous_phrase_penalty(self):
        """Hedging language subtracts 0.2."""
        score = calculate_artifact_confidence(
            {"title": "Maybe implement something", "description": ""},
            "task",
        )
        # base + verb(implement) - ambiguity(maybe) = 0.2 + 0.2 - 0.2 = 0.2
        assert score <= 0.4

    def test_all_fields_populated_bonus(self):
        """All expected fields filled adds +0.2."""
        score = calculate_artifact_confidence(
            {
                "title": "Build API",
                "description": "Implement JWT endpoints",
                "assignee": "Mike",
                "priority": "high",
            },
            "task",
        )
        # base(0.2) + owner(0.2) + priority(0.2) + verb(0.2) + all_fields(0.2) = 1.0
        assert score == 1.0

    def test_score_clamped_to_unit_interval(self):
        """Score never exceeds 1.0 or drops below 0.0."""
        perfect = calculate_artifact_confidence(
            {
                "title": "Deploy microservice",
                "description": "Ship to production",
                "assignee": "Alice",
                "priority": "critical",
            },
            "task",
        )
        assert 0.0 <= perfect <= 1.0

    def test_blocker_uses_owner_not_assignee(self):
        """Blocker type checks 'owner' for the owner bonus, not 'assignee'."""
        score = calculate_artifact_confidence(
            {
                "title": "Server down",
                "description": "Production issue",
                "owner": "Dev Team",
                "resolution_plan": "Restart",
            },
            "blocker",
        )
        assert score >= 0.4

    def test_user_story_scoring(self):
        """User story with rich fields scores high."""
        score = calculate_artifact_confidence(
            {
                "title": "User wants to create account",
                "as_a": "visitor",
                "i_want": "to create an account",
                "so_that": "I can save preferences",
                "acceptance_criteria": ["Form validates email"],
                "priority": "high",
            },
            "user_story",
        )
        assert score >= 0.6

    def test_execution_task_scoring(self):
        score = calculate_artifact_confidence(
            {
                "title": "Configure CI/CD",
                "description": "Set up pipeline",
                "owner_role": "DevOps",
                "priority": "High",
                "dependencies": [],
            },
            "execution_task",
        )
        assert score >= 0.6

    def test_unknown_artifact_type_uses_empty_spec(self):
        """Unknown type still returns a valid score based on text heuristics."""
        score = calculate_artifact_confidence(
            {"title": "Implement fix", "description": ""},
            "unknown_type",
        )
        assert 0.0 <= score <= 1.0


# backfill_confidence_scores â€” mutation in-place

class TestBackfillConfidenceScores:
    def test_fills_none_scores(self):
        artifacts = MeetingArtifacts(
            user_stories=[
                UserStory(
                    title="Login",
                    as_a="user",
                    i_want="to log in",
                    so_that="secure",
                    confidence_score=None,
                ),
            ],
            tasks=[
                Task(title="Build API", confidence_score=None),
            ],
        )
        backfill_confidence_scores(artifacts)

        assert artifacts.user_stories[0].confidence_score is not None
        assert artifacts.tasks[0].confidence_score is not None

    def test_preserves_existing_scores(self):
        artifacts = MeetingArtifacts(
            tasks=[
                Task(title="Build API", confidence_score=0.95),
            ],
        )
        backfill_confidence_scores(artifacts)
        assert artifacts.tasks[0].confidence_score == 0.95

    def test_handles_empty_artifact_lists(self):
        artifacts = MeetingArtifacts()
        backfill_confidence_scores(artifacts)
        # No exception raised

    def test_fills_all_artifact_types(self):
        artifacts = MeetingArtifacts(
            decisions=[Decision(title="Use JWT", description="Secure auth")],
            blockers=[Blocker(title="No DB", description="Need provisioning")],
            action_items=[ActionItem(description="Update docs")],
            execution_tasks=[
                ActionableTask(
                    title="Deploy",
                    description="Ship to prod",
                    owner_role="DevOps",
                    task_source="Explicit",
                ),
            ],
        )
        backfill_confidence_scores(artifacts)

        assert artifacts.decisions[0].confidence_score is not None
        assert artifacts.blockers[0].confidence_score is not None
        assert artifacts.action_items[0].confidence_score is not None
        assert artifacts.execution_tasks[0].confidence_score is not None
