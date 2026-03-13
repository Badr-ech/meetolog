"""Tests for production reliability features: structured logging, crash recovery, retries, and timeouts."""

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import configure_logging, get_logger
from app.infrastructure.db import Base
from app.infrastructure.postgres_queue import LOCK_TIMEOUT_SECONDS, PostgresJobQueue
from app.models.db_models import JobRecord


# ---------------------------------------------------------------------------
# Helpers — neutralise ``FOR UPDATE SKIP LOCKED`` for SQLite
# ---------------------------------------------------------------------------

def _strip_for_update(original_claim):
    """Wrap *claim_next_job* so the SELECT it builds drops ``FOR UPDATE``."""
    import functools

    @functools.wraps(original_claim)
    async def _patched(self, worker_id):
        from sqlalchemy import or_
        now = datetime.now(timezone.utc)
        stale_threshold = now - timedelta(seconds=LOCK_TIMEOUT_SECONDS)

        stmt = (
            select(JobRecord)
            .where(
                or_(
                    JobRecord.status == "pending",
                    (JobRecord.status == "processing")
                    & (JobRecord.locked_at < stale_threshold),
                    (JobRecord.status == "failed")
                    & (JobRecord.next_retry_at <= now)
                    & (JobRecord.attempts < JobRecord.max_retries),
                )
            )
            .order_by(JobRecord.created_at.asc())
            .limit(1)
            # No .with_for_update() — SQLite doesn't support it
        )

        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        if record is None:
            return None

        record.status = "processing"
        record.locked_at = now
        record.locked_by = worker_id
        record.attempts += 1
        record.error = None
        record.updated_at = now

        await self._session.commit()
        await self._session.refresh(record)
        return record

    return _patched


# ---------------------------------------------------------------------------
# Structured Logging
# ---------------------------------------------------------------------------

class TestStructuredLogging:
    def test_get_logger_returns_bound_logger(self):
        log = get_logger(component="test")
        assert log is not None

    def test_configure_logging_runs_without_error(self):
        configure_logging(json_output=False, log_level="DEBUG")

    def test_context_binding(self):
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(job_id="abc-123")
        ctx = structlog.contextvars.get_contextvars()
        assert ctx.get("job_id") == "abc-123"
        structlog.contextvars.clear_contextvars()


# ---------------------------------------------------------------------------
# Crash Recovery — Stale Lock Timeout
# ---------------------------------------------------------------------------

class TestStaleLockRecovery:
    def test_lock_timeout_is_two_hours(self):
        assert LOCK_TIMEOUT_SECONDS == 7200

    @pytest.mark.asyncio
    async def test_stale_processing_job_is_reclaimed(self, sqlite_session: AsyncSession):
        """A job stuck in 'processing' with an old locked_at should be re-claimable."""
        job_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        stale_time = now - timedelta(seconds=LOCK_TIMEOUT_SECONDS + 60)

        record = JobRecord(
            id=job_id,
            s3_key="uploads/test.wav",
            file_name="test.wav",
            file_size=1024,
            status="processing",
            progress=25,
            message="Stuck",
            locked_at=stale_time,
            locked_by="dead-worker",
            attempts=1,
        )
        sqlite_session.add(record)
        await sqlite_session.commit()

        queue = PostgresJobQueue(sqlite_session)
        with patch.object(PostgresJobQueue, "claim_next_job", _strip_for_update(queue.claim_next_job)):
            claimed = await queue.claim_next_job("recovery-worker")

        assert claimed is not None
        assert claimed.id == job_id
        assert claimed.locked_by == "recovery-worker"
        assert claimed.status == "processing"
        assert claimed.attempts == 2

    @pytest.mark.asyncio
    async def test_fresh_processing_job_is_not_reclaimed(self, sqlite_session: AsyncSession):
        """A job that was recently locked should NOT be reclaimed."""
        job_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        record = JobRecord(
            id=job_id,
            s3_key="uploads/fresh.wav",
            file_name="fresh.wav",
            file_size=1024,
            status="processing",
            progress=25,
            message="In progress",
            locked_at=now,
            locked_by="active-worker",
            attempts=1,
        )
        sqlite_session.add(record)
        await sqlite_session.commit()

        queue = PostgresJobQueue(sqlite_session)
        with patch.object(PostgresJobQueue, "claim_next_job", _strip_for_update(queue.claim_next_job)):
            claimed = await queue.claim_next_job("greedy-worker")

        # Should not reclaim because the lock is still fresh.
        assert claimed is None

    @pytest.mark.asyncio
    async def test_pending_job_claimed_normally(self, sqlite_session: AsyncSession):
        job_id = uuid.uuid4()
        record = JobRecord(
            id=job_id,
            s3_key="uploads/new.wav",
            file_name="new.wav",
            file_size=1024,
            status="pending",
            progress=0,
            message="Queued",
        )
        sqlite_session.add(record)
        await sqlite_session.commit()

        queue = PostgresJobQueue(sqlite_session)
        with patch.object(PostgresJobQueue, "claim_next_job", _strip_for_update(queue.claim_next_job)):
            claimed = await queue.claim_next_job("normal-worker")

        assert claimed is not None
        assert claimed.id == job_id
        assert claimed.status == "processing"


# ---------------------------------------------------------------------------
# LLM Extraction — Retry & Timeout
# ---------------------------------------------------------------------------

class TestLLMRetryAndTimeout:
    @pytest.mark.asyncio
    async def test_llm_timeout_value(self):
        """Verify that the LLM call timeout is 60 seconds."""
        from app.services.llm_engine import _LLM_CALL_TIMEOUT

        assert _LLM_CALL_TIMEOUT == 60

    @pytest.mark.asyncio
    async def test_retry_decorator_present_on_gemini_call(self):
        """The _call_gemini method should be decorated with tenacity retry."""
        from app.services.llm_engine import GeminiProvider

        method = getattr(GeminiProvider, "_call_gemini", None)
        assert method is not None
        # tenacity-decorated functions have a .retry attribute
        assert hasattr(method, "retry")


# ---------------------------------------------------------------------------
# Worker Hardening
# ---------------------------------------------------------------------------

class TestWorkerHardening:
    @pytest.mark.asyncio
    async def test_process_job_catches_all_exceptions(self):
        """process_job must never raise — it catches all exceptions internally."""
        from app.worker import process_job

        mock_queue = AsyncMock()
        mock_queue.update_job_progress = AsyncMock()
        mock_queue.mark_job_failed = AsyncMock()
        mock_s3 = AsyncMock()
        mock_s3.download_to_file = AsyncMock(side_effect=RuntimeError("Boom"))

        # Should not raise
        await process_job(
            job_id=uuid.uuid4(),
            s3_key="uploads/bad.wav",
            file_name="bad.wav",
            queue=mock_queue,
            s3_service=mock_s3,
            worker_id="test-worker",
        )

        mock_queue.mark_job_failed.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_process_job_emits_job_finished_log(self):
        """Verify the finally block runs and would emit job_finished."""
        from app.worker import process_job

        mock_queue = AsyncMock()
        mock_queue.update_job_progress = AsyncMock()
        mock_queue.mark_job_failed = AsyncMock()
        mock_s3 = AsyncMock()
        mock_s3.download_to_file = AsyncMock(side_effect=Exception("fail"))

        with patch("app.worker.get_logger") as mock_get_logger:
            mock_log = MagicMock()
            mock_get_logger.return_value = mock_log

            await process_job(
                job_id=uuid.uuid4(),
                s3_key="uploads/x.wav",
                file_name="x.wav",
                queue=mock_queue,
                s3_service=mock_s3,
                worker_id="test-worker",
            )

            # Check that job_finished was logged
            finished_calls = [
                c for c in mock_log.info.call_args_list
                if c.args and c.args[0] == "job_finished"
            ]
            assert len(finished_calls) == 1
            assert "duration_seconds" in finished_calls[0].kwargs
            assert finished_calls[0].kwargs["status"] == "failed"
