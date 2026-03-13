"""
Jira Bulk Import JSON mapper for Meetolog artifacts.

Transforms MeetingArtifacts into the Jira JSON import format:
    { "projects": [{ "key": "MEET", "issues": [...] }] }

Edge-case handling:
    1. Missing titles/descriptions: Falls back to "(No title)" or "(No description)"
       since Jira requires a non-empty summary on every issue.
    2. Summary length: Jira enforces a 255-character limit on the summary field.
       Titles exceeding this are truncated to 252 chars + "…".
    3. Unmapped artifact types: Decisions, ActionItems, and generic Tasks
       do not have a direct Story/Task/Bug counterpart in the mapping spec.
       They are emitted as issueType "Task" with a "[Decision]", "[Action Item]",
       or "[Task]" prefix so that importers can filter/relabel them in Jira.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import (
    MeetingArtifacts,
    UserStory,
    ActionableTask,
    Blocker,
    Decision,
    ActionItem,
    Idea,
    Task,
)

# ---------------------------------------------------------------------------
# Jira validation models
# ---------------------------------------------------------------------------

JIRA_SUMMARY_MAX = 255

_PRIORITY_MAP: dict[str, str] = {
    "critical": "Highest",
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    # ActionableTask uses title-case priorities
    "Critical": "Highest",
    "High": "High",
    "Medium": "Medium",
    "Low": "Low",
}

DEFAULT_PRIORITY = "Medium"


class JiraIssue(BaseModel):
    """Single issue inside the Jira bulk-import payload."""

    summary: str = Field(..., max_length=JIRA_SUMMARY_MAX)
    description: str = ""
    issueType: str = "Task"
    priority: str = DEFAULT_PRIORITY
    assignee: str | None = None
    labels: list[str] = Field(default_factory=list)


class JiraProject(BaseModel):
    key: str = "MEET"
    issues: list[JiraIssue] = Field(default_factory=list)


class JiraExportPayload(BaseModel):
    """Top-level Jira bulk-import JSON structure."""

    projects: list[JiraProject] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _truncate(text: str, limit: int = JIRA_SUMMARY_MAX) -> str:
    """Truncate *text* to *limit* characters, appending '…' if shortened."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _safe_summary(value: str | None, fallback: str = "(No title)") -> str:
    """Return a non-empty, length-safe summary string."""
    text = (value or "").strip()
    return _truncate(text) if text else fallback


def _map_priority(raw: str | None) -> str:
    if raw is None:
        return DEFAULT_PRIORITY
    return _PRIORITY_MAP.get(raw, _PRIORITY_MAP.get(raw.lower(), DEFAULT_PRIORITY))


# ---------------------------------------------------------------------------
# Per-artifact converters
# ---------------------------------------------------------------------------


def _story_to_issue(story: UserStory) -> JiraIssue:
    description_parts: list[str] = []
    if story.as_a or story.i_want or story.so_that:
        description_parts.append(
            f"As a {story.as_a}, I want {story.i_want} so that {story.so_that}"
        )
    if story.acceptance_criteria:
        description_parts.append(
            "Acceptance Criteria:\n"
            + "\n".join(f"- {ac}" for ac in story.acceptance_criteria)
        )
    if story.story_points is not None:
        description_parts.append(f"Story Points: {story.story_points}")

    return JiraIssue(
        summary=_safe_summary(story.title),
        description="\n\n".join(description_parts),
        issueType="Story",
        priority=_map_priority(story.priority.value if hasattr(story.priority, "value") else story.priority),
        labels=["meetolog", "user-story"],
    )


def _execution_task_to_issue(task: ActionableTask) -> JiraIssue:
    description_parts = [task.description or ""]
    if task.owner_role:
        description_parts.append(f"Owner/Role: {task.owner_role}")
    if task.dependencies:
        description_parts.append(
            "Dependencies:\n" + "\n".join(f"- {d}" for d in task.dependencies)
        )
    description_parts.append(f"Source: {task.task_source}")

    return JiraIssue(
        summary=_safe_summary(task.title),
        description="\n\n".join(description_parts),
        issueType="Task",
        priority=_map_priority(task.priority),
        labels=["meetolog", "execution-task", task.task_source.lower()],
    )


def _blocker_to_issue(blocker: Blocker) -> JiraIssue:
    description_parts = [blocker.description or ""]
    if blocker.resolution_plan:
        description_parts.append(f"Resolution Plan: {blocker.resolution_plan}")
    if blocker.affected_tasks:
        description_parts.append(
            "Affected Tasks:\n" + "\n".join(f"- {t}" for t in blocker.affected_tasks)
        )

    return JiraIssue(
        summary=_safe_summary(blocker.title),
        description="\n\n".join(description_parts),
        issueType="Bug",
        priority="Highest",
        assignee=blocker.owner,
        labels=["meetolog", "blocker"],
    )


def _decision_to_issue(decision: Decision) -> JiraIssue:
    """Decisions have no direct Jira type — emitted as Task with a prefix."""
    description_parts = [decision.description or ""]
    if decision.rationale:
        description_parts.append(f"Rationale: {decision.rationale}")
    if decision.made_by:
        description_parts.append(f"Decided by: {decision.made_by}")

    return JiraIssue(
        summary=_safe_summary(f"[Decision] {decision.title}"),
        description="\n\n".join(description_parts),
        issueType="Task",
        labels=["meetolog", "decision"],
    )


def _action_item_to_issue(item: ActionItem) -> JiraIssue:
    return JiraIssue(
        summary=_safe_summary(f"[Action Item] {item.description}"),
        description=item.description or "",
        issueType="Task",
        assignee=item.assignee,
        labels=["meetolog", "action-item"],
    )


def _task_to_issue(task: Task) -> JiraIssue:
    return JiraIssue(
        summary=_safe_summary(f"[Task] {task.title}"),
        description=task.description or "",
        issueType="Task",
        priority=_map_priority(task.priority.value if hasattr(task.priority, "value") else task.priority),
        assignee=task.assignee,
        labels=["meetolog", "task"],
    )


def _idea_to_issue(idea: Idea) -> JiraIssue:
    description_parts = [idea.idea_description or ""]
    if idea.potential_impact:
        description_parts.append(f"Potential Impact: {idea.potential_impact}")
    if idea.proposed_by:
        description_parts.append(f"Proposed by: {idea.proposed_by}")

    return JiraIssue(
        summary=_safe_summary(f"[Idea] {idea.idea_description}"),
        description="\n\n".join(description_parts),
        issueType="Task",
        labels=["meetolog", "idea"],
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def map_artifacts_to_jira(
    artifacts: MeetingArtifacts,
    project_key: str = "MEET",
) -> JiraExportPayload:
    """Convert a ``MeetingArtifacts`` instance into a validated Jira bulk-import payload.

    Args:
        artifacts: The Meetolog artifacts to convert.
        project_key: Jira project key to use in the export (default ``MEET``).

    Returns:
        A ``JiraExportPayload`` that has been validated by Pydantic.
    """
    issues: list[JiraIssue] = []

    for story in artifacts.user_stories:
        issues.append(_story_to_issue(story))

    for et in artifacts.execution_tasks:
        issues.append(_execution_task_to_issue(et))

    for blocker in artifacts.blockers:
        issues.append(_blocker_to_issue(blocker))

    for decision in artifacts.decisions:
        issues.append(_decision_to_issue(decision))

    for item in artifacts.action_items:
        issues.append(_action_item_to_issue(item))

    for task in artifacts.tasks:
        issues.append(_task_to_issue(task))

    for idea in artifacts.ideas:
        issues.append(_idea_to_issue(idea))

    project = JiraProject(key=project_key, issues=issues)
    return JiraExportPayload(projects=[project])
