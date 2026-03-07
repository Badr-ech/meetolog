"""Tests for Jira bulk-import JSON mapper (jira_mapper.py)."""

import pytest

from app.services.jira_mapper import (
    map_artifacts_to_jira,
    _truncate,
    _safe_summary,
    _map_priority,
    JiraExportPayload,
    JIRA_SUMMARY_MAX,
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


# _truncate

class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("Hello") == "Hello"

    def test_exact_limit_unchanged(self):
        text = "a" * JIRA_SUMMARY_MAX
        assert _truncate(text) == text

    def test_long_text_truncated_with_ellipsis(self):
        text = "x" * 300
        result = _truncate(text)
        assert len(result) == JIRA_SUMMARY_MAX
        assert result.endswith("…")

    def test_custom_limit(self):
        result = _truncate("Hello World", limit=6)
        assert len(result) == 6
        assert result.endswith("…")


# _safe_summary

class TestSafeSummary:
    def test_normal_title(self):
        assert _safe_summary("My Title") == "My Title"

    def test_none_returns_fallback(self):
        assert _safe_summary(None) == "(No title)"

    def test_empty_string_returns_fallback(self):
        assert _safe_summary("") == "(No title)"

    def test_whitespace_returns_fallback(self):
        assert _safe_summary("   ") == "(No title)"

    def test_custom_fallback(self):
        assert _safe_summary(None, "N/A") == "N/A"

    def test_long_title_truncated(self):
        long_title = "T" * 300
        result = _safe_summary(long_title)
        assert len(result) <= JIRA_SUMMARY_MAX


# _map_priority

class TestMapPriority:
    @pytest.mark.parametrize("raw,expected", [
        ("critical", "Highest"),
        ("Critical", "Highest"),
        ("high", "High"),
        ("High", "High"),
        ("medium", "Medium"),
        ("Medium", "Medium"),
        ("low", "Low"),
        ("Low", "Low"),
    ])
    def test_known_priorities(self, raw, expected):
        assert _map_priority(raw) == expected

    def test_none_returns_medium(self):
        assert _map_priority(None) == "Medium"

    def test_unknown_returns_medium(self):
        assert _map_priority("urgent") == "Medium"


# map_artifacts_to_jira â€” full conversion

class TestMapArtifactsToJira:
    @pytest.fixture
    def rich_artifacts(self) -> MeetingArtifacts:
        return MeetingArtifacts(
            meeting_title="Sprint Planning",
            user_stories=[
                UserStory(
                    title="Login Feature",
                    as_a="user",
                    i_want="to log in",
                    so_that="secure access",
                    acceptance_criteria=["Validates email"],
                    priority=Priority.HIGH,
                    story_points=5,
                ),
            ],
            tasks=[
                Task(
                    title="Build API",
                    description="JWT endpoints",
                    assignee="Mike",
                    priority=Priority.HIGH,
                ),
            ],
            decisions=[
                Decision(
                    title="30-min timeout",
                    description="Session policy",
                    made_by="Sarah",
                    rationale="Best practice",
                ),
            ],
            blockers=[
                Blocker(
                    title="Email not configured",
                    description="Blocks password reset",
                    owner="Mike",
                ),
            ],
            action_items=[
                ActionItem(description="Update docs", assignee="John"),
            ],
            execution_tasks=[
                ActionableTask(
                    title="Deploy auth",
                    description="Ship to prod",
                    owner_role="Engineering",
                    task_source="Explicit",
                ),
                ActionableTask(
                    title="Write tests",
                    description="Unit tests for auth",
                    owner_role="QA",
                    task_source="Inferred",
                ),
            ],
        )

    def test_returns_valid_payload(self, rich_artifacts):
        payload = map_artifacts_to_jira(rich_artifacts)
        assert isinstance(payload, JiraExportPayload)
        assert len(payload.projects) == 1
        assert payload.projects[0].key == "MEET"

    def test_issue_count(self, rich_artifacts):
        payload = map_artifacts_to_jira(rich_artifacts)
        # 1 story + 1 task + 1 decision + 1 blocker + 1 action_item + 2 execution_tasks = 7
        assert len(payload.projects[0].issues) == 7

    def test_story_mapped_correctly(self, rich_artifacts):
        payload = map_artifacts_to_jira(rich_artifacts)
        stories = [i for i in payload.projects[0].issues if i.issueType == "Story"]
        assert len(stories) == 1
        assert stories[0].summary == "Login Feature"
        assert "meetolog" in stories[0].labels

    def test_blocker_mapped_as_bug(self, rich_artifacts):
        payload = map_artifacts_to_jira(rich_artifacts)
        bugs = [i for i in payload.projects[0].issues if i.issueType == "Bug"]
        assert len(bugs) == 1
        assert bugs[0].priority == "Highest"

    def test_decision_prefixed(self, rich_artifacts):
        payload = map_artifacts_to_jira(rich_artifacts)
        decisions = [i for i in payload.projects[0].issues if "decision" in i.labels]
        assert len(decisions) == 1
        assert decisions[0].summary.startswith("[Decision]")

    def test_execution_task_labels_include_source(self, rich_artifacts):
        payload = map_artifacts_to_jira(rich_artifacts)
        et_issues = [i for i in payload.projects[0].issues if "execution-task" in i.labels]
        assert len(et_issues) == 2
        sources = {lbl for i in et_issues for lbl in i.labels if lbl in ("explicit", "inferred")}
        assert sources == {"explicit", "inferred"}

    def test_custom_project_key(self, rich_artifacts):
        payload = map_artifacts_to_jira(rich_artifacts, project_key="PROJ")
        assert payload.projects[0].key == "PROJ"

    def test_empty_artifacts_produces_empty_issues(self):
        payload = map_artifacts_to_jira(MeetingArtifacts())
        assert len(payload.projects[0].issues) == 0

    def test_missing_title_uses_prefix_only(self):
        """Artifact with empty title still gets the [Task] prefix."""
        artifacts = MeetingArtifacts(
            tasks=[Task(title="", description="Something")],
        )
        payload = map_artifacts_to_jira(artifacts)
        # _safe_summary receives "[Task] " which is non-empty after strip â†’ "[Task]"
        assert payload.projects[0].issues[0].summary == "[Task]"

    def test_long_title_truncated(self):
        long_title = "A" * 300
        artifacts = MeetingArtifacts(
            tasks=[Task(title=long_title, description="desc")],
        )
        payload = map_artifacts_to_jira(artifacts)
        summary = payload.projects[0].issues[0].summary
        assert len(summary) <= JIRA_SUMMARY_MAX
