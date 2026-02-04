"""
Pydantic models for Agile artifacts extracted from meeting transcripts.
These define the structured output format of the semantic extraction.
"""

from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Priority(str, Enum):
    """Priority levels for tasks and stories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskStatus(str, Enum):
    """Status options for tasks."""
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"


class Task(BaseModel):
    """A task extracted from meeting discussion."""
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., description="Brief task description")
    description: str = Field(default="", description="Detailed task information")
    assignee: str | None = Field(default=None, description="Person assigned to the task")
    priority: Priority = Field(default=Priority.MEDIUM)
    status: TaskStatus = Field(default=TaskStatus.TODO)
    due_date: str | None = Field(default=None, description="Expected completion date if mentioned")


class UserStory(BaseModel):
    """A user story in standard Agile format."""
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., description="Story title")
    as_a: str = Field(..., description="The user role (As a...)")
    i_want: str = Field(..., description="The desired action (I want...)")
    so_that: str = Field(..., description="The benefit (So that...)")
    acceptance_criteria: list[str] = Field(default_factory=list)
    priority: Priority = Field(default=Priority.MEDIUM)
    story_points: int | None = Field(default=None, description="Estimated complexity")


class Decision(BaseModel):
    """A decision made during the meeting."""
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., description="Decision summary")
    description: str = Field(..., description="Full decision details")
    made_by: str | None = Field(default=None, description="Who made the decision")
    rationale: str = Field(default="", description="Why this decision was made")
    timestamp: str | None = Field(default=None, description="When in the meeting this was decided")


class Blocker(BaseModel):
    """A blocker or impediment identified in the meeting."""
    id: UUID = Field(default_factory=uuid4)
    title: str = Field(..., description="Blocker summary")
    description: str = Field(..., description="Details about the blocker")
    affected_tasks: list[str] = Field(default_factory=list, description="Tasks impacted by this blocker")
    owner: str | None = Field(default=None, description="Person responsible for resolving")
    resolution_plan: str = Field(default="", description="Proposed solution if discussed")


class ActionItem(BaseModel):
    """A general action item that doesn't fit other categories."""
    id: UUID = Field(default_factory=uuid4)
    description: str
    assignee: str | None = None
    due_date: str | None = None


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
    
    # Raw data
    transcript: str = Field(default="", description="Full meeting transcript")


class ProcessingStatus(str, Enum):
    """Status of the audio processing job."""
    PENDING = "pending"
    TRANSCRIBING = "transcribing"
    EXTRACTING = "extracting"
    GENERATING_PDF = "generating_pdf"
    COMPLETED = "completed"
    FAILED = "failed"


class JobResponse(BaseModel):
    """Response returned when a job is created or queried."""
    job_id: UUID
    status: ProcessingStatus
    message: str = ""
    progress: int = Field(default=0, ge=0, le=100)
    artifacts: MeetingArtifacts | None = None
    pdf_url: str | None = None
    error: str | None = None
