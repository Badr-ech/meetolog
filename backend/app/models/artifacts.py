"""
LLM response validation models and parsing utilities.

Defines Pydantic models matching the JSON schema expected from the
extraction prompts in ``core.prompts``, and provides utilities to parse,
repair, and validate raw LLM text responses into typed Python objects.

Processing Pipeline
-------------------
1. ``strip_markdown_fencing``  — remove ```json fencing from LLM output.
2. ``sanitize_json_string``    — fix trailing commas and common malformations.
3. ``validate_llm_response``   — parse JSON (with repair fallback) and
   validate against ``LLMExtractionResponse``.
4. ``to_meeting_artifacts``    — convert the validated intermediate model
   into the domain ``MeetingArtifacts`` used by the rest of the system.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from .schemas import (
    ActionableTask,
    ActionItem,
    Blocker,
    Decision,
    Idea,
    MeetingArtifacts,
    Priority,
    Task,
    UserStory,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Shared LLM-model validator helpers
# ======================================================================

def _strip_llm_fields(values: dict) -> dict:
    """Strip whitespace from all string values in the dict."""
    for key, value in values.items():
        if isinstance(value, str):
            values[key] = value.strip()
    return values


def _clamp_confidence_field(values: dict) -> dict:
    """Clamp ``confidence_score`` to [0.0, 1.0]."""
    score = values.get("confidence_score")
    if score is not None:
        try:
            values["confidence_score"] = max(0.0, min(1.0, round(float(score), 2)))
        except (TypeError, ValueError):
            values["confidence_score"] = None
    return values


# ======================================================================
# LLM Output Schema Models
# ======================================================================

class LLMUserStory(BaseModel):
    """Expected LLM output schema for a user story."""
    title: str = ""
    as_a: str = ""
    i_want: str = ""
    so_that: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)
    priority: str = "medium"
    story_points: int | None = None
    confidence_score: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, values: dict) -> dict:
        return _clamp_confidence_field(_strip_llm_fields(values))


class LLMTask(BaseModel):
    """Expected LLM output schema for a task."""
    title: str = ""
    description: str = ""
    assignee: str | None = None
    due_date: str | None = None
    context: str = ""
    priority: str | None = "medium"
    confidence_score: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, values: dict) -> dict:
        return _clamp_confidence_field(_strip_llm_fields(values))


class LLMDecision(BaseModel):
    """Expected LLM output schema for a decision (includes CoT reasoning)."""
    title: str = ""
    description: str = ""
    decision_summary: str = ""
    made_by: str | None = None
    rationale: str = ""
    alternatives_rejected: list[str] = Field(default_factory=list)
    reasoning: str = ""
    confidence_score: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, values: dict) -> dict:
        values = _clamp_confidence_field(_strip_llm_fields(values))
        if not values.get("decision_summary") and values.get("title"):
            values["decision_summary"] = values["title"]
        return values


class LLMBlocker(BaseModel):
    """Expected LLM output schema for a blocker."""
    title: str = ""
    description: str = ""
    affected_tasks: list[str] = Field(default_factory=list)
    owner: str | None = None
    resolution_plan: str = ""
    confidence_score: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, values: dict) -> dict:
        return _clamp_confidence_field(_strip_llm_fields(values))


class LLMActionItem(BaseModel):
    """Expected LLM output schema for an action item."""
    title: str = ""
    description: str = ""
    assignee: str | None = None
    due_date: str | None = None
    context: str = ""
    priority: str | None = "medium"
    confidence_score: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, values: dict) -> dict:
        return _clamp_confidence_field(_strip_llm_fields(values))


class LLMIdea(BaseModel):
    """Expected LLM output schema for an idea or suggestion."""
    idea_description: str = ""
    proposed_by: str | None = None
    potential_impact: str = ""
    confidence_score: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, values: dict) -> dict:
        return _clamp_confidence_field(_strip_llm_fields(values))


class LLMExecutionTask(BaseModel):
    """Expected LLM output schema for an execution task."""
    title: str = ""
    description: str = ""
    owner_role: str = "Engineering"
    priority: str = "Medium"
    task_source: str = "Explicit"
    dependencies: list[str] = Field(default_factory=list)
    confidence_score: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, values: dict) -> dict:
        return _clamp_confidence_field(_strip_llm_fields(values))


class LLMExtractionResponse(BaseModel):
    """Complete LLM extraction output matching the prompt schema."""
    meeting_title: str = "Untitled Meeting"
    summary: str = ""
    participants: list[str] = Field(default_factory=list)
    user_stories: list[LLMUserStory] = Field(default_factory=list)
    tasks: list[LLMTask] = Field(default_factory=list)
    decisions: list[LLMDecision] = Field(default_factory=list)
    blockers: list[LLMBlocker] = Field(default_factory=list)
    action_items: list[LLMActionItem] = Field(default_factory=list)
    ideas: list[LLMIdea] = Field(default_factory=list)
    execution_tasks: list[LLMExecutionTask] = Field(default_factory=list)


# ======================================================================
# Parsing & Validation Utilities
# ======================================================================

_MARKDOWN_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*\n?(.*?)\n?\s*```\s*$",
    re.DOTALL,
)

_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def strip_markdown_fencing(text: str) -> str:
    """Remove markdown code fencing (``\\`json … \\```) from LLM output."""
    text = text.strip()
    match = _MARKDOWN_FENCE_RE.match(text)
    if match:
        return match.group(1).strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = len(lines)
        if lines[-1].strip().startswith("```"):
            end -= 1
        if len(lines) > 1:
            return "\n".join(lines[1:end]).strip()
    return text


def sanitize_json_string(text: str) -> str:
    """Pre-process raw LLM text to fix common JSON malformations.

    - Strips markdown code fencing.
    - Removes trailing commas before ``}`` and ``]``.
    - Strips any leading/trailing non-JSON text.
    """
    text = strip_markdown_fencing(text)
    text = _TRAILING_COMMA_RE.sub(r"\1", text)

    # Strip any conversational preamble/postamble outside the JSON object.
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        text = text[first_brace : last_brace + 1]

    return text


def _attempt_json_repair(text: str) -> dict:
    """Attempt to repair malformed JSON via the ``json_repair`` library."""
    try:
        import json_repair
    except ImportError:
        raise ValueError(
            "json_repair is not installed. "
            "Install it with: pip install json-repair"
        )
    result = json_repair.loads(text)
    if not isinstance(result, dict):
        raise ValueError(
            f"json_repair produced {type(result).__name__}, expected dict"
        )
    return result


def validate_llm_response(raw_text: str) -> LLMExtractionResponse:
    """Parse and validate a raw LLM response string.

    Processing steps:
    1. Sanitize (strip fencing, trailing commas, non-JSON text).
    2. Attempt standard ``json.loads`` parsing.
    3. On failure, attempt repair via ``json_repair``.
    4. Validate the parsed dict against ``LLMExtractionResponse``.

    Parameters
    ----------
    raw_text:
        Raw string response from the LLM.

    Returns
    -------
    LLMExtractionResponse
        Validated extraction data.

    Raises
    ------
    ValueError
        If the response cannot be parsed as JSON even after repair.
    pydantic.ValidationError
        If the parsed JSON does not conform to the schema.
    """
    json_str = sanitize_json_string(raw_text)

    data: dict | None = None
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as parse_err:
        logger.warning(
            "Standard JSON parse failed, attempting repair: %s", parse_err
        )
        try:
            data = _attempt_json_repair(json_str)
        except Exception as repair_err:
            raise ValueError(
                f"LLM response is not valid JSON and could not be repaired. "
                f"Parse error: {parse_err}. Repair error: {repair_err}"
            ) from repair_err

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object from LLM, got {type(data).__name__}"
        )

    return LLMExtractionResponse.model_validate(data)


# ======================================================================
# Conversion to Domain Models
# ======================================================================

_PRIORITY_MAP = {
    "low": Priority.LOW,
    "medium": Priority.MEDIUM,
    "high": Priority.HIGH,
    "critical": Priority.CRITICAL,
}


def _parse_priority(raw: str | None) -> Priority:
    """Convert a raw priority string to the ``Priority`` enum."""
    if not raw:
        return Priority.MEDIUM
    return _PRIORITY_MAP.get(raw.strip().lower(), Priority.MEDIUM)


_VALID_EXEC_PRIORITIES = {"High", "Medium", "Low"}
_VALID_TASK_SOURCES = {"Explicit", "Inferred"}


def to_meeting_artifacts(
    response: LLMExtractionResponse,
    transcript: str,
) -> MeetingArtifacts:
    """Convert a validated ``LLMExtractionResponse`` to the domain model.

    Maps intermediate LLM output models to the canonical Pydantic models
    used throughout the application (``UserStory``, ``Task``, etc.),
    handling priority enum conversion, confidence clamping, and
    ``execution_task`` field normalisation.
    """
    user_stories = [
        UserStory(
            title=s.title,
            as_a=s.as_a,
            i_want=s.i_want,
            so_that=s.so_that,
            acceptance_criteria=s.acceptance_criteria,
            priority=_parse_priority(s.priority),
            story_points=s.story_points,
            confidence_score=s.confidence_score,
        )
        for s in response.user_stories
    ]

    tasks = [
        Task(
            title=t.title,
            description=t.description,
            assignee=t.assignee,
            due_date=t.due_date,
            context=t.context,
            priority=_parse_priority(t.priority),
            confidence_score=t.confidence_score,
        )
        for t in response.tasks
    ]

    decisions = [
        Decision(
            title=d.title,
            description=d.description,
            decision_summary=d.decision_summary,
            made_by=d.made_by,
            rationale=d.rationale,
            alternatives_rejected=d.alternatives_rejected,
            confidence_score=d.confidence_score,
        )
        for d in response.decisions
    ]

    blockers = [
        Blocker(
            title=b.title,
            description=b.description,
            affected_tasks=b.affected_tasks,
            owner=b.owner,
            resolution_plan=b.resolution_plan,
            confidence_score=b.confidence_score,
        )
        for b in response.blockers
    ]

    action_items = [
        ActionItem(
            title=a.title,
            description=a.description,
            assignee=a.assignee,
            due_date=a.due_date,
            context=a.context,
            priority=_parse_priority(a.priority),
            confidence_score=a.confidence_score,
        )
        for a in response.action_items
    ]

    ideas = [
        Idea(
            idea_description=idea.idea_description,
            proposed_by=idea.proposed_by,
            potential_impact=idea.potential_impact,
            confidence_score=idea.confidence_score,
        )
        for idea in response.ideas
    ]

    execution_tasks = [
        ActionableTask(
            title=et.title,
            description=et.description,
            owner_role=et.owner_role,
            priority=et.priority if et.priority in _VALID_EXEC_PRIORITIES else "Medium",
            task_source=et.task_source if et.task_source in _VALID_TASK_SOURCES else "Explicit",
            dependencies=et.dependencies,
            confidence_score=et.confidence_score,
        )
        for et in response.execution_tasks
    ]

    return MeetingArtifacts(
        meeting_title=response.meeting_title,
        meeting_date=datetime.now(),
        participants=response.participants,
        summary=response.summary,
        user_stories=user_stories,
        tasks=tasks,
        decisions=decisions,
        blockers=blockers,
        action_items=action_items,
        ideas=ideas,
        execution_tasks=execution_tasks,
        transcript=transcript,
    )
