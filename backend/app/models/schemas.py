"""
Pydantic models for Agile artifacts extracted from meeting transcripts.
These define the structured output format of the semantic extraction.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Shared validators
# ---------------------------------------------------------------------------

_DATE_FORMATS = [
    "%Y-%m-%d",
    "%m/%d/%Y",
    "%d/%m/%Y",
    "%B %d, %Y",
    "%b %d, %Y",
]


def _strip_str_fields(data: dict) -> dict:
    """Strip leading/trailing whitespace from all string values."""
    for key, value in data.items():
        if isinstance(value, str):
            data[key] = value.strip()
    return data


def _normalize_date(raw: str | None) -> str | None:
    """Attempt to parse and normalise a date string to ISO-8601 (YYYY-MM-DD).

    Returns the original string unchanged when it cannot be parsed into a
    known format (e.g. relative dates like ``"Next Friday"``).
    """
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"


# ---------------------------------------------------------------------------
# Artifact models
# ---------------------------------------------------------------------------

class Task(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., description="Brief task description")
    description: str = Field(default="", description="Detailed task information")
    assignee: str | None = Field(default=None, description="Person assigned to the task")
    due_date: str | None = Field(default=None, description="Expected completion date if mentioned")
    context: str = Field(default="", description="Surrounding discussion context from the transcript")
    priority: Priority = Field(default=Priority.MEDIUM)
    status: TaskStatus = Field(default=TaskStatus.TODO)
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Extraction confidence (0.0–1.0)")

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, values: dict) -> dict:
        values = _strip_str_fields(values)
        values["due_date"] = _normalize_date(values.get("due_date"))
        return values


class UserStory(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., description="Story title")
    as_a: str = Field(..., description="The user role (As a...)")
    i_want: str = Field(..., description="The desired action (I want...)")
    so_that: str = Field(..., description="The benefit (So that...)")
    acceptance_criteria: list[str] = Field(default_factory=list)
    priority: Priority = Field(default=Priority.MEDIUM)
    story_points: int | None = Field(default=None, description="Estimated complexity")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Extraction confidence (0.0–1.0)")

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, values: dict) -> dict:
        values = _strip_str_fields(values)

        return values


class Decision(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., description="Decision title")
    description: str = Field(..., description="Full decision details")
    decision_summary: str = Field(default="", description="Concise summary of what was decided")
    made_by: str | None = Field(default=None, description="Who made the decision")
    rationale: str = Field(default="", description="Why this decision was made")
    alternatives_rejected: list[str] = Field(
        default_factory=list,
        description="Options that were considered but not chosen",
    )
    timestamp: str | None = Field(default=None, description="When in the meeting this was decided")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Extraction confidence (0.0–1.0)")

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, values: dict) -> dict:
        values = _strip_str_fields(values)

        if not values.get("decision_summary") and values.get("title"):
            values["decision_summary"] = values["title"]
        return values


class Blocker(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., description="Blocker summary")
    description: str = Field(..., description="Details about the blocker")
    affected_tasks: list[str] = Field(default_factory=list, description="Tasks impacted by this blocker")
    owner: str | None = Field(default=None, description="Person responsible for resolving")
    resolution_plan: str = Field(default="", description="Proposed solution if discussed")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Extraction confidence (0.0–1.0)")

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, values: dict) -> dict:
        values = _strip_str_fields(values)

        return values


class ActionItem(BaseModel):
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(default="", description="Brief title for the action item")
    description: str = ""
    assignee: str | None = None
    due_date: str | None = None
    context: str = Field(default="", description="Surrounding discussion context from the transcript")
    priority: Priority = Field(default=Priority.MEDIUM)
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Extraction confidence (0.0–1.0)")

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, values: dict) -> dict:
        values = _strip_str_fields(values)
        values["due_date"] = _normalize_date(values.get("due_date"))

        if not values.get("title") and values.get("description"):
            values["title"] = (values["description"][:80] + "…") if len(values.get("description", "")) > 80 else values.get("description", "")
        return values


class Idea(BaseModel):
    """An idea or suggestion raised during the meeting."""
    id: UUID = Field(default_factory=uuid4)
    idea_description: str = Field(..., description="Detailed description of the idea")
    proposed_by: str | None = Field(default=None, description="Person who proposed the idea")
    potential_impact: str = Field(default="", description="Expected impact or benefit of the idea")
    confidence_score: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Extraction confidence (0.0–1.0)")

    @model_validator(mode="before")
    @classmethod
    def _clean(cls, values: dict) -> dict:
        values = _strip_str_fields(values)

        return values


class ActionableTask(BaseModel):
    title: str = Field(..., description="Concise task title")
    description: str = Field(..., description="Detailed description of work required")
    owner_role: str = Field(
        ...,
        description="Responsible role (e.g., Engineering, Design, Product) or specific name if mentioned",
    )
    priority: Literal["High", "Medium", "Low"] = Field(default="Medium")
    task_source: Literal["Explicit", "Inferred"] = Field(
        ...,
        description="Whether this task was directly stated or logically inferred",
    )
    dependencies: list[str] = Field(
        default_factory=list,
        description="Other tasks or conditions this task depends on",
    )
    confidence_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Extraction confidence (0.0–1.0)",
    )


class MeetingArtifacts(BaseModel):
    """Complete extracted artifacts from a meeting."""
    meeting_id: UUID = Field(default_factory=uuid4)
    meeting_title: str = Field(default="Untitled Meeting")
    meeting_date: datetime = Field(default_factory=datetime.now)
    duration_minutes: int | None = None
    participants: list[str] = Field(default_factory=list)
    summary: str = Field(default="", description="Brief meeting summary")
    
    # Extracted Agile Artifacts
    user_stories: list[UserStory] = Field(default_factory=list)
    tasks: list[Task] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    blockers: list[Blocker] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    ideas: list[Idea] = Field(default_factory=list)
    execution_tasks: list[ActionableTask] = Field(default_factory=list)
    
    # Raw data
    transcript: str = Field(default="", description="Full meeting transcript")


class ProcessingStatus(str, Enum):
    """
    Status of the audio processing job.

    v1.1 granular states provide per-stage feedback to the frontend.
    Legacy values ``"pending"`` and ``"processing"`` are accepted on
    read via :func:`parse_processing_status` but should not be written
    by new code.
    """
    UPLOADING = "uploading"
    DIARIZING = "diarizing"
    TRANSCRIBING = "transcribing"
    EXTRACTING = "extracting"
    GENERATING_PDF = "generating_pdf"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Backward-compatibility mapping for Redis records written before v1.1
# ---------------------------------------------------------------------------
_LEGACY_STATUS_MAP: dict[str, ProcessingStatus] = {
    "pending": ProcessingStatus.UPLOADING,
    "processing": ProcessingStatus.TRANSCRIBING,
}


def parse_processing_status(value: str) -> ProcessingStatus:
    """Convert a raw status string to a ``ProcessingStatus`` enum member.

    Handles legacy ``"pending"`` / ``"processing"`` values transparently.
    Falls back to ``UPLOADING`` for completely unknown values.
    """
    try:
        return ProcessingStatus(value)
    except ValueError:
        mapped = _LEGACY_STATUS_MAP.get(value)
        if mapped is not None:
            return mapped
        return ProcessingStatus.UPLOADING


class JobResponse(BaseModel):
    """Response returned when a job is created or queried."""
    job_id: UUID
    status: ProcessingStatus
    message: str = ""
    progress: int = Field(default=0, ge=0, le=100)
    artifacts: MeetingArtifacts | None = None
    pdf_url: str | None = None
    error: str | None = None
