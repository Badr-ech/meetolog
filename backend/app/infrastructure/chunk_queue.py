"""
PostgreSQL-backed queue for parallel audio chunk processing.

Each ``job_chunks`` row represents one 5-minute audio segment.  Multiple
``chunk_worker`` Fargate tasks claim rows concurrently via
``SELECT … FOR UPDATE SKIP LOCKED`` — the same locking pattern used by
the job queue — so two workers never transcribe the same segment.

The last worker to complete its chunk calls
:meth:`try_transition_to_assembling`, which atomically moves the parent
job to ``assembling`` status exactly once (using a conditional UPDATE).
Only the worker whose UPDATE affects one row launches the assembler task.
"""

import uuid
from datetime import datetime, timezone

import structlog
from sqlalchemy import func, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.db_models import JobChunk, JobRecord

logger = structlog.get_logger(__name__)


class ChunkQueue:
    """Data-access layer for the ``job_chunks`` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    # ------------------------------------------------------------------
    # Write path (splitter creates chunks)
    # ------------------------------------------------------------------

    async def create_chunks(
        self,
        job_id: uuid.UUID,
        audio_s3_keys: list[str],
    ) -> list[JobChunk]:
        """Insert one ``JobChunk`` row per audio segment.

        *audio_s3_keys* must be ordered chronologically; the list index
        becomes the ``chunk_index`` stored on each row.
        """
        chunks = [
            JobChunk(
                id=uuid.uuid4(),
                job_id=job_id,
                chunk_index=i,
                audio_s3_key=key,
                status="pending",
            )
            for i, key in enumerate(audio_s3_keys)
        ]
        self._session.add_all(chunks)
        try:
            await self._session.commit()
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("Failed to create chunks for job %s", job_id)
            raise
        return chunks

    # ------------------------------------------------------------------
    # Claim path (chunk workers)
    # ------------------------------------------------------------------

    async def claim_next_chunk(
        self,
        job_id: uuid.UUID,
        worker_id: str,
    ) -> JobChunk | None:
        """Atomically claim the next pending chunk for *job_id*.

        Returns ``None`` when all chunks have been claimed or completed.
        Uses ``FOR UPDATE SKIP LOCKED`` so concurrent workers never
        contend on the same row.
        """
        now = datetime.now(timezone.utc)

        stmt = (
            select(JobChunk)
            .where(
                JobChunk.job_id == job_id,
                JobChunk.status == "pending",
            )
            .order_by(JobChunk.chunk_index.asc())
            .limit(1)
            .with_for_update(skip_locked=True)
        )

        try:
            result = await self._session.execute(stmt)
            chunk = result.scalar_one_or_none()
            if chunk is None:
                return None

            chunk.status = "processing"
            chunk.locked_by = worker_id
            chunk.locked_at = now
            await self._session.commit()
            await self._session.refresh(chunk)
            return chunk
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception(
                "Failed to claim chunk for job %s / worker %s", job_id, worker_id
            )
            raise

    async def mark_chunk_completed(
        self,
        chunk_id: uuid.UUID,
        transcript: str,
        detected_language: str | None,
    ) -> None:
        """Save the transcript and mark the chunk ``completed``."""
        now = datetime.now(timezone.utc)
        try:
            await self._session.execute(
                update(JobChunk)
                .where(JobChunk.id == chunk_id)
                .values(
                    status="completed",
                    transcript=transcript,
                    detected_language=detected_language,
                    completed_at=now,
                    locked_by=None,
                    locked_at=None,
                )
            )
            await self._session.commit()
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("Failed to mark chunk %s completed", chunk_id)
            raise

    async def mark_chunk_failed(
        self,
        chunk_id: uuid.UUID,
        error: str,
    ) -> None:
        """Mark a chunk as ``failed`` and record the error message."""
        try:
            await self._session.execute(
                update(JobChunk)
                .where(JobChunk.id == chunk_id)
                .values(
                    status="failed",
                    error=error,
                    locked_by=None,
                    locked_at=None,
                )
            )
            await self._session.commit()
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("Failed to mark chunk %s failed", chunk_id)
            raise

    # ------------------------------------------------------------------
    # Read path (assembler + progress tracking)
    # ------------------------------------------------------------------

    async def count_completed_chunks(
        self, job_id: uuid.UUID
    ) -> tuple[int, int]:
        """Return ``(completed_count, total_count)`` for *job_id*'s chunks.

        Used by chunk workers to emit proportional progress updates on the
        parent job record as each chunk finishes.
        """
        try:
            result = await self._session.execute(
                select(
                    func.count().filter(JobChunk.status == "completed"),
                    func.count(),
                )
                .select_from(JobChunk)
                .where(JobChunk.job_id == job_id)
            )
            row = result.one()
            return int(row[0]), int(row[1])
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("Failed to count chunks for job %s", job_id)
            raise

    async def get_completed_transcripts(
        self, job_id: uuid.UUID
    ) -> list[str]:
        """Return all completed chunk transcripts ordered by chunk_index."""
        try:
            result = await self._session.execute(
                select(JobChunk.transcript)
                .where(
                    JobChunk.job_id == job_id,
                    JobChunk.status == "completed",
                    JobChunk.transcript.isnot(None),
                )
                .order_by(JobChunk.chunk_index.asc())
            )
            return [row[0] for row in result.all()]
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("Failed to fetch transcripts for job %s", job_id)
            raise

    # ------------------------------------------------------------------
    # Transition: transcribing → assembling
    # ------------------------------------------------------------------

    async def try_transition_to_assembling(
        self, job_id: uuid.UUID
    ) -> bool:
        """Atomically move the job to ``assembling`` if all chunks are done.

        Returns ``True`` exactly once across all concurrent workers — the
        one whose UPDATE transitions the row from ``transcribing`` to
        ``assembling``.  All other callers get ``False`` and should simply
        exit.

        The check-then-update is done inside a single transaction with a
        ``FOR UPDATE`` lock on the job row, preventing the TOCTOU race
        where two workers both see zero pending chunks and both try to
        launch the assembler.
        """
        try:
            # Lock the job row for the duration of this check.
            result = await self._session.execute(
                select(JobRecord.status)
                .where(JobRecord.id == job_id)
                .with_for_update()
            )
            row = result.one_or_none()
            if row is None or row[0] != "transcribing":
                # Job already transitioned (another worker won the race).
                await self._session.rollback()
                return False

            # Count chunks that are not yet completed.
            count_result = await self._session.execute(
                select(func.count())
                .select_from(JobChunk)
                .where(
                    JobChunk.job_id == job_id,
                    JobChunk.status != "completed",
                )
            )
            remaining = count_result.scalar_one()
            if remaining > 0:
                await self._session.rollback()
                return False

            # All chunks done — transition.
            await self._session.execute(
                update(JobRecord)
                .where(JobRecord.id == job_id, JobRecord.status == "transcribing")
                .values(
                    status="assembling",
                    message="Assembling transcript…",
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await self._session.commit()
            return True
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception(
                "Failed to transition job %s to assembling", job_id
            )
            raise
