"""
PostgreSQL-backed persistent job queue.

Uses ``SELECT … FOR UPDATE SKIP LOCKED`` to allow multiple concurrent
workers to claim jobs without double-processing.  Row-level locking
ensures that only one worker ever processes a given job, even under
high concurrency, without blocking the entire table.

Retry semantics: when a job fails and has remaining attempts, it is
returned to ``pending`` with an exponential back-off delay written to
``next_retry_at``.  A crashed worker's lock is considered stale after
``LOCK_TIMEOUT_SECONDS`` and the job becomes claimable again.
"""

import uuid
from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy import or_, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import JobRecord

logger = structlog.get_logger(__name__)

# A lock older than this is treated as abandoned (worker crash recovery).
# 2 hours allows long transcription jobs to finish while still reclaiming
# truly orphaned work after an OOM kill or forced container scale-down.
LOCK_TIMEOUT_SECONDS = 7200  # 2 hours

# Base delay for exponential back-off between retries (doubles per attempt).
RETRY_BASE_DELAY_SECONDS = 30


class PostgresJobQueue:
    """Data-access layer for the Postgres-backed job queue."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def enqueue_job(
        self,
        *,
        job_id: uuid.UUID,
        s3_key: str,
        file_name: str,
        file_size: int,
    ) -> JobRecord:
        """Insert a new job into the queue with ``pending`` status."""
        record = JobRecord(
            id=job_id,
            s3_key=s3_key,
            file_name=file_name,
            file_size=file_size,
            status="pending",
            progress=0,
            message="Job queued for processing",
        )
        self._session.add(record)
        try:
            await self._session.commit()
        except SQLAlchemyError:
            await self._session.rollback()
            raise
        await self._session.refresh(record)
        return record

    async def claim_next_job(self, worker_id: str) -> JobRecord | None:
        """Atomically claim the next eligible job for *worker_id*.

        Eligible rows are, in priority order:
        1. ``pending`` jobs (oldest first).
        2. ``processing`` jobs whose ``locked_at`` exceeds the stale-lock
           timeout (crash recovery).
        3. ``failed`` jobs eligible for retry (``next_retry_at <= now`` and
           ``attempts < max_retries``).

        ``FOR UPDATE SKIP LOCKED`` guarantees that concurrent workers never
        contend on the same row — a locked row is silently skipped rather
        than blocking the caller.
        """
        now = datetime.now(timezone.utc)
        stale_threshold = now - timedelta(seconds=LOCK_TIMEOUT_SECONDS)

        stmt = (
            select(JobRecord)
            .where(
                or_(
                    JobRecord.status == "pending",
                    # Stale lock recovery
                    (JobRecord.status == "processing")
                    & (JobRecord.locked_at < stale_threshold),
                    # Retry eligible
                    (JobRecord.status == "failed")
                    & (JobRecord.next_retry_at <= now)
                    & (JobRecord.attempts < JobRecord.max_retries),
                )
            )
            .order_by(JobRecord.created_at.asc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )

        try:
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
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("Failed to claim next job for worker %s", worker_id)
            raise

    async def mark_job_completed(
        self,
        job_id: uuid.UUID,
        *,
        artifacts: dict | None = None,
        pdf_url: str | None = None,
        pdf_s3_key: str | None = None,
        artifacts_s3_key: str | None = None,
    ) -> None:
        """Transition a job to ``completed`` and store its output artifacts."""
        now = datetime.now(timezone.utc)
        values: dict = {
            "status": "completed",
            "progress": 100,
            "message": "Processing complete!",
            "locked_at": None,
            "updated_at": now,
        }
        if artifacts is not None:
            values["artifacts"] = artifacts
        if pdf_url is not None:
            values["pdf_url"] = pdf_url
        if pdf_s3_key is not None:
            values["pdf_s3_key"] = pdf_s3_key
        if artifacts_s3_key is not None:
            values["artifacts_s3_key"] = artifacts_s3_key

        try:
            await self._session.execute(
                update(JobRecord).where(JobRecord.id == job_id).values(**values)
            )
            await self._session.commit()
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("Failed to mark job %s as completed", job_id)
            raise

    async def mark_job_failed(
        self,
        job_id: uuid.UUID,
        error_msg: str,
    ) -> None:
        """Mark a job as failed, queueing it for retry if attempts remain.

        If ``attempts < max_retries`` the job returns to ``pending`` with a
        ``next_retry_at`` computed via exponential back-off so the worker
        does not immediately re-poll the same broken job.  Otherwise the
        status is set to ``failed`` permanently.
        """
        now = datetime.now(timezone.utc)

        try:
            result = await self._session.execute(
                select(JobRecord.attempts, JobRecord.max_retries)
                .where(JobRecord.id == job_id)
            )
            row = result.one_or_none()
            if row is None:
                logger.warning("mark_job_failed called for unknown job %s", job_id)
                return

            attempts, max_retries = row

            if attempts < max_retries:
                delay = timedelta(
                    seconds=RETRY_BASE_DELAY_SECONDS * (2 ** (attempts - 1)),
                )
                values: dict = {
                    "status": "pending",
                    "progress": 0,
                    "error": error_msg,
                    "message": f"Retrying (attempt {attempts}/{max_retries})",
                    "locked_at": None,
                    "locked_by": None,
                    "next_retry_at": now + delay,
                    "updated_at": now,
                }
            else:
                values = {
                    "status": "failed",
                    "progress": 0,
                    "error": error_msg,
                    "message": "Processing failed after all retries",
                    "locked_at": None,
                    "locked_by": None,
                    "updated_at": now,
                }

            await self._session.execute(
                update(JobRecord).where(JobRecord.id == job_id).values(**values)
            )
            await self._session.commit()
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("Failed to mark job %s as failed", job_id)
            raise

    async def update_job_progress(
        self,
        job_id: uuid.UUID,
        *,
        status: str | None = None,
        progress: int | None = None,
        message: str | None = None,
    ) -> None:
        """Update progress metadata for a running job."""
        values: dict = {"updated_at": datetime.now(timezone.utc)}
        if status is not None:
            values["status"] = status
        if progress is not None:
            values["progress"] = progress
        if message is not None:
            values["message"] = message

        try:
            await self._session.execute(
                update(JobRecord).where(JobRecord.id == job_id).values(**values)
            )
            await self._session.commit()
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("Failed to update progress for job %s", job_id)
            raise
