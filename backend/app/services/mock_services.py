"""
Mock implementations for testing and CI mode.

Return deterministic, schema-compliant data without external API calls.
"""

import asyncio
import logging
from datetime import datetime
from typing import Awaitable, Callable
from pathlib import Path
from uuid import uuid4

from ..interfaces import Transcriber, LLMExtractor
from ..models import (
    ActionableTask,
    MeetingArtifacts,
    UserStory,
    Task,
    Decision,
    Blocker,
    ActionItem,
    Idea,
    Priority,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class MockTranscriber(Transcriber):
    MOCK_TRANSCRIPT = """
Okay everyone, let's get started with our sprint planning meeting. I'm Sarah, the product owner, 
and we have the dev team here - Mike, Lisa, and John.

So, our main focus this sprint is the user authentication feature. We've had a lot of customer 
requests for this. As a user, I want to be able to log in with my email and password, so that 
my data is secure and personalized.

Mike, can you take the lead on this?

Mike: Sure, I'll handle the backend authentication. I think we'll need about 5 story points for 
the basic login and registration flow. I'll set up JWT tokens and the database schema.

Lisa: I can work on the frontend login forms. That's probably a 3-pointer. I'll also add 
password validation and error handling.

John: I'll take the password reset flow. Users should be able to reset via email. That's about 
3 story points as well.

Sarah: Great. We also need to decide on the session timeout. I suggest 30 minutes of inactivity.

Mike: That works for me. We should also implement refresh tokens.

Sarah: Agreed. So that's a decision - 30 minute session timeout with refresh tokens.

John: One blocker I want to raise - we still don't have the email service configured. I need 
that for password reset emails.

Sarah: Good point. Mike, can you set up the email service this week? That's blocking John's task.

Mike: I'll prioritize that. Should be done by Wednesday.

Lisa: Also, can we get acceptance criteria documented? I need to know the exact validation rules 
for passwords.

Sarah: Yes, acceptance criteria: minimum 8 characters, at least one uppercase, one lowercase, 
one number, and one special character. Also, the login form needs rate limiting after 5 failed 
attempts.

Okay, any other action items? 

John: I'll update the technical documentation once the authentication is in place.

Sarah: Perfect. Let's wrap up. Good meeting everyone!
""".strip()
    
    def __init__(self, simulated_delay: float = 0.5):
        self._delay = simulated_delay
        logger.info("MockTranscriber initialized (TEST_MODE)")
    
    async def transcribe(
        self,
        audio_path: Path,
        progress_callback: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> str:
        """Return a mock transcript after a simulated delay."""
        logger.info("MockTranscriber: Simulating transcription for %s", audio_path)
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        await asyncio.sleep(self._delay)
        
        logger.info("MockTranscriber: Returning mock transcript")
        return self.MOCK_TRANSCRIPT
    
    async def preprocess_transcript(self, raw_transcript: str) -> str:
        """Apply the same whitespace normalisation as the real transcriber."""
        lines = (line.strip() for line in raw_transcript.strip().splitlines())
        return "\n".join(" ".join(line.split()) for line in lines if line)


class MockExtractor(LLMExtractor):
    """Deterministic in-memory extractor used for tests and ``TEST_MODE``."""

    def __init__(self, simulated_delay: float = 0.3):
        self._delay = simulated_delay
        logger.info("MockExtractor initialized (TEST_MODE)")

    @property
    def provider_name(self) -> str:
        return "Mock"

    @property
    def is_mock(self) -> bool:
        return True

    async def generate_text(self, prompt: str) -> str:
        """Return a truncated echo of the prompt for testing."""
        await asyncio.sleep(self._delay)
        return f"Mock summary of input text ({len(prompt)} chars)."

    async def extract_artifacts(self, transcript: str) -> MeetingArtifacts:
        """Return deterministic, schema-valid artifacts after a small delay."""
        logger.info("MockExtractor: Generating mock artifacts")

        await asyncio.sleep(self._delay)

        artifacts = MeetingArtifacts(
            meeting_id=uuid4(),
            meeting_title="Sprint Planning - User Authentication Feature",
            meeting_date=datetime.now(),
            duration_minutes=45,
            participants=["Sarah (Product Owner)", "Mike (Backend)", "Lisa (Frontend)", "John (Full Stack)"],
            summary=(
                "Sprint planning meeting focused on implementing user authentication. "
                "The team estimated user stories for login, registration, and password reset flows. "
                "Key decisions were made regarding session timeout and refresh tokens."
            ),
            user_stories=[
                UserStory(
                    id=uuid4(),
                    title="User Login with Email/Password",
                    as_a="registered user",
                    i_want="to log in with my email and password",
                    so_that="my data is secure and personalized",
                    acceptance_criteria=[
                        "Login form validates email format",
                        "Password minimum 8 characters with complexity requirements",
                        "Rate limiting after 5 failed attempts",
                        "JWT token issued on successful login",
                    ],
                    priority=Priority.HIGH,
                    story_points=5,
                ),
                UserStory(
                    id=uuid4(),
                    title="Password Reset via Email",
                    as_a="user who forgot their password",
                    i_want="to reset my password via email",
                    so_that="I can regain access to my account",
                    acceptance_criteria=[
                        "User receives reset link within 5 minutes",
                        "Reset link expires after 1 hour",
                        "New password must meet complexity requirements",
                    ],
                    priority=Priority.MEDIUM,
                    story_points=3,
                ),
            ],
            tasks=[
                Task(
                    id=uuid4(),
                    title="Implement Backend Authentication",
                    description="Set up JWT tokens, database schema for users, login and registration endpoints",
                    assignee="Mike",
                    priority=Priority.HIGH,
                    status=TaskStatus.TODO,
                    due_date=None,
                ),
                Task(
                    id=uuid4(),
                    title="Create Frontend Login Forms",
                    description="Build login and registration forms with validation and error handling",
                    assignee="Lisa",
                    priority=Priority.HIGH,
                    status=TaskStatus.TODO,
                    due_date=None,
                ),
                Task(
                    id=uuid4(),
                    title="Implement Password Reset Flow",
                    description="Create password reset request and confirmation endpoints with email integration",
                    assignee="John",
                    priority=Priority.MEDIUM,
                    status=TaskStatus.BLOCKED,
                    due_date=None,
                ),
                Task(
                    id=uuid4(),
                    title="Configure Email Service",
                    description="Set up email service for password reset and notification emails",
                    assignee="Mike",
                    priority=Priority.HIGH,
                    status=TaskStatus.TODO,
                    due_date="Wednesday",
                ),
            ],
            decisions=[
                Decision(
                    id=uuid4(),
                    title="Session Timeout Policy",
                    description="User sessions will timeout after 30 minutes of inactivity, with refresh token support",
                    made_by="Sarah",
                    rationale="Balances security with user experience for typical usage patterns",
                ),
                Decision(
                    id=uuid4(),
                    title="Password Complexity Requirements",
                    description="Passwords require minimum 8 characters, uppercase, lowercase, number, and special character",
                    made_by="Sarah",
                    rationale="Industry standard security requirements for user authentication",
                ),
            ],
            blockers=[
                Blocker(
                    id=uuid4(),
                    title="Email Service Not Configured",
                    description="The email service is not yet set up, blocking password reset functionality",
                    affected_tasks=["Implement Password Reset Flow"],
                    owner="Mike",
                    resolution_plan="Mike will configure email service by Wednesday",
                ),
            ],
            action_items=[
                ActionItem(
                    id=uuid4(),
                    description="Update technical documentation after authentication is implemented",
                    assignee="John",
                    due_date=None,
                ),
                ActionItem(
                    id=uuid4(),
                    description="Document acceptance criteria for password validation",
                    assignee="Sarah",
                    due_date=None,
                ),
            ],
            ideas=[
                Idea(
                    id=uuid4(),
                    idea_description="Implement OAuth2 social login as a follow-up to the email/password flow",
                    proposed_by="Lisa",
                    potential_impact="Could increase sign-up conversion by reducing friction for new users",
                ),
            ],
            execution_tasks=[
                ActionableTask(
                    title="Implement Backend Authentication",
                    description="Set up JWT tokens, database schema for users, login and registration endpoints",
                    owner_role="Mike",
                    priority="High",
                    task_source="Explicit",
                    dependencies=[],
                ),
                ActionableTask(
                    title="Create Frontend Login Forms",
                    description="Build login and registration forms with validation and error handling",
                    owner_role="Lisa",
                    priority="High",
                    task_source="Explicit",
                    dependencies=["Implement Backend Authentication"],
                ),
                ActionableTask(
                    title="Configure Email Service",
                    description="Set up email service for password reset and notification emails, blocking password reset flow",
                    owner_role="Mike",
                    priority="High",
                    task_source="Explicit",
                    dependencies=[],
                ),
                ActionableTask(
                    title="Implement Password Reset Flow",
                    description="Create password reset request and confirmation endpoints with email integration",
                    owner_role="John",
                    priority="Medium",
                    task_source="Explicit",
                    dependencies=["Configure Email Service"],
                ),
                ActionableTask(
                    title="Set Up Refresh Token Rotation",
                    description="Implement refresh token issuance and rotation mechanism per the 30-minute session timeout decision",
                    owner_role="Engineering",
                    priority="Medium",
                    task_source="Inferred",
                    dependencies=["Implement Backend Authentication"],
                ),
                ActionableTask(
                    title="Add Login Rate Limiting",
                    description="Implement rate limiting after 5 failed login attempts as required by acceptance criteria",
                    owner_role="Engineering",
                    priority="Medium",
                    task_source="Inferred",
                    dependencies=["Implement Backend Authentication"],
                ),
            ],
            transcript=transcript,
        )

        logger.info(
            "MockExtractor: generated %d stories, %d tasks, %d decisions",
            len(artifacts.user_stories), len(artifacts.tasks), len(artifacts.decisions),
        )
        return artifacts
