"""Shared fixtures for backend test suite."""

import asyncio
from datetime import datetime
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.infrastructure.db import Base


# ---------------------------------------------------------------------------
# SQLite compatibility: map PostgreSQL JSONB to plain JSON for in-memory tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def _register_jsonb_for_sqlite():
    """Teach the SQLite type compiler how to render JSONB columns."""
    from sqlalchemy.dialects.sqlite.base import SQLiteTypeCompiler

    if not hasattr(SQLiteTypeCompiler, "visit_JSONB"):
        def visit_JSONB(self, type_, **kw):
            return self.visit_JSON(type_, **kw)
        SQLiteTypeCompiler.visit_JSONB = visit_JSONB


@pytest_asyncio.fixture
async def sqlite_engine():
    """Create an in-memory async SQLite engine with all tables provisioned."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def sqlite_session(sqlite_engine):
    """Yield a per-test async session backed by in-memory SQLite."""
    factory = async_sessionmaker(bind=sqlite_engine, expire_on_commit=False)
    async with factory() as session:
        yield session

from app.models import (
    JobResponse,
    MeetingArtifacts,
    ProcessingStatus,
    UserStory,
    Task,
    Decision,
    Blocker,
    ActionItem,
    Idea,
    Priority,
    TaskStatus,
)


SAMPLE_JOB_ID = UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def sample_job_id() -> UUID:
    return SAMPLE_JOB_ID


@pytest.fixture
def sample_job_response(sample_job_id: UUID) -> JobResponse:
    return JobResponse(
        job_id=sample_job_id,
        status=ProcessingStatus.UPLOADING,
        message="Job queued for processing",
        progress=0,
    )


@pytest.fixture
def sample_artifacts() -> MeetingArtifacts:
    return MeetingArtifacts(
        meeting_title="Sprint Planning - Auth Feature",
        meeting_date=datetime(2025, 6, 15, 10, 0),
        duration_minutes=45,
        participants=["Sarah", "Mike", "Lisa"],
        summary="Sprint planning focused on authentication.",
        user_stories=[
            UserStory(
                title="User Login",
                as_a="user",
                i_want="to log in with email",
                so_that="my data is secure",
                acceptance_criteria=["Validates email format"],
                priority=Priority.HIGH,
                story_points=5,
            ),
        ],
        tasks=[
            Task(
                title="Backend auth API",
                description="Implement JWT auth endpoints",
                assignee="Mike",
                priority=Priority.HIGH,
                status=TaskStatus.TODO,
            ),
        ],
        decisions=[
            Decision(
                title="30-min session timeout",
                description="Session expires after 30 minutes of inactivity",
                made_by="Sarah",
                rationale="Industry standard for security",
            ),
        ],
        blockers=[
            Blocker(
                title="Email service not configured",
                description="Needed for password reset flow",
                affected_tasks=["Password reset"],
                owner="Mike",
            ),
        ],
        action_items=[
            ActionItem(description="Update docs", assignee="John"),
        ],
        ideas=[
            Idea(
                idea_description="Add OAuth2 social login later",
                proposed_by="Lisa",
                potential_impact="Increase sign-up conversion",
            ),
        ],
        transcript="Sample transcript text.",
    )


@pytest.fixture
def sample_transcript() -> str:
    return (
        "Sarah: Let's discuss the auth feature. "
        "Mike: I'll handle the backend. "
        "Lisa: I'll do the frontend forms."
    )


@pytest.fixture
def deterministic_llm_response() -> dict:
    """Raw JSON matching the LLM response schema."""
    return {
        "meeting_title": "Sprint Planning - Auth",
        "summary": "Discussed authentication feature implementation.",
        "participants": ["Sarah", "Mike"],
        "user_stories": [
            {
                "title": "User Login",
                "as_a": "registered user",
                "i_want": "to log in",
                "so_that": "my data is secure",
                "acceptance_criteria": ["Email validated"],
                "priority": "high",
                "story_points": 5,
            },
        ],
        "tasks": [
            {
                "title": "Build auth API",
                "description": "JWT endpoints",
                "assignee": "Mike",
                "priority": "high",
                "due_date": None,
            },
        ],
        "decisions": [
            {
                "title": "Session timeout",
                "description": "30 minutes",
                "made_by": "Sarah",
                "rationale": "Security best practice",
            },
        ],
        "blockers": [],
        "action_items": [
            {"description": "Update docs", "assignee": "John", "due_date": None},
        ],
        "ideas": [
            {
                "idea_description": "Add OAuth2 social login",
                "proposed_by": "Lisa",
                "potential_impact": "Improve sign-up conversion",
            },
        ],
    }
