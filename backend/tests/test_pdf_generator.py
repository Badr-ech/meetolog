"""Tests for PDF generation service."""

import pytest
from pathlib import Path
from datetime import datetime

from app.services.pdf_generator import PDFGeneratorService, _fmt_confidence
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


class TestFmtConfidence:
    def test_none_returns_na(self):
        assert _fmt_confidence(None) == "N/A"

    def test_zero(self):
        assert _fmt_confidence(0.0) == "0%"

    def test_one(self):
        assert _fmt_confidence(1.0) == "100%"

    def test_mid_value(self):
        assert _fmt_confidence(0.75) == "75%"

    def test_rounding(self):
        assert _fmt_confidence(0.333) == "33%"


class TestPDFGeneratorService:
    @pytest.fixture
    def pdf_service(self, tmp_path) -> PDFGeneratorService:
        return PDFGeneratorService(output_dir=tmp_path)

    @pytest.fixture
    def full_artifacts(self) -> MeetingArtifacts:
        return MeetingArtifacts(
            meeting_title="Sprint Planning",
            meeting_date=datetime(2025, 6, 15, 10, 0),
            duration_minutes=45,
            participants=["Alice", "Bob"],
            summary="Discussed auth feature.",
            user_stories=[
                UserStory(
                    title="Login",
                    as_a="user",
                    i_want="to log in",
                    so_that="data is secure",
                    acceptance_criteria=["Email validated"],
                    priority=Priority.HIGH,
                    story_points=5,
                    confidence_score=0.85,
                ),
            ],
            tasks=[
                Task(
                    title="Build API",
                    description="JWT endpoints",
                    assignee="Bob",
                    priority=Priority.HIGH,
                    confidence_score=0.9,
                ),
            ],
            decisions=[
                Decision(
                    title="30-min timeout",
                    description="Session expires after 30 minutes",
                    made_by="Alice",
                    rationale="Industry standard",
                    confidence_score=0.7,
                ),
            ],
            blockers=[
                Blocker(
                    title="Email service down",
                    description="Blocks password reset",
                    owner="Bob",
                    confidence_score=0.6,
                ),
            ],
            action_items=[
                ActionItem(description="Update docs", assignee="Alice"),
            ],
            execution_tasks=[
                ActionableTask(
                    title="Deploy auth",
                    description="Ship to prod",
                    owner_role="Engineering",
                    task_source="Explicit",
                    confidence_score=0.8,
                ),
            ],
            transcript="Sample transcript.",
        )

    @pytest.mark.asyncio
    async def test_generates_pdf_file(self, pdf_service, full_artifacts, tmp_path):
        path = await pdf_service.generate(full_artifacts, "test.pdf")
        assert path.exists()
        assert path.suffix == ".pdf"
        assert path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_auto_generated_filename(self, pdf_service, full_artifacts):
        path = await pdf_service.generate(full_artifacts)
        assert path.exists()
        assert "meeting_summary" in path.name

    @pytest.mark.asyncio
    async def test_empty_artifacts_still_produces_pdf(self, pdf_service):
        empty = MeetingArtifacts()
        path = await pdf_service.generate(empty, "empty.pdf")
        assert path.exists()
        assert path.stat().st_size > 0

    @pytest.mark.asyncio
    async def test_creates_output_directory(self, tmp_path):
        nested = tmp_path / "deep" / "nested"
        service = PDFGeneratorService(output_dir=nested)
        path = await service.generate(MeetingArtifacts(), "test.pdf")
        assert path.exists()
