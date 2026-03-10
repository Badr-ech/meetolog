"""PostgreSQL-backed job store using async SQLAlchemy sessions."""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession

from ..interfaces import JobStore
from ..models import JobResponse, MeetingArtifacts, ProcessingStatus, parse_processing_status
from ..models.db_models import JobRecord

logger = logging.getLogger(__name__)


class PostgresJobStore(JobStore):
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def save(
        self,
        job_id: UUID,
        job: JobResponse,
        file_path: str | None = None,
        file_name: str | None = None,
        file_size: int | None = None,
    ) -> None:
        try:
            record = JobRecord(
                id=job_id,
                s3_key=file_path or "",
                status=job.status.value,
                progress=job.progress,
                message=job.message or "",
            )
            self._session.add(record)
            await self._session.commit()
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("Failed to save job %s", job_id)
            raise

    async def load(self, job_id: UUID) -> JobResponse | None:
        try:
            result = await self._session.execute(
                select(JobRecord).where(JobRecord.id == job_id)
            )
            record = result.scalar_one_or_none()
            if record is None:
                return None
            return self._to_response(record)
        except SQLAlchemyError:
            logger.exception("Failed to load job %s", job_id)
            raise

    async def update(self, job_id: UUID, **kwargs: Any) -> JobResponse | None:
        try:
            values: dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
            for key, val in kwargs.items():
                if key == "status" and isinstance(val, ProcessingStatus):
                    values["status"] = val.value
                elif key == "artifacts" and isinstance(val, MeetingArtifacts):
                    values["artifacts"] = val.model_dump(mode="json")
                elif key in (
                    "progress", "message", "error",
                    "pdf_url", "worker_id",
                ):
                    values[key] = val

            await self._session.execute(
                update(JobRecord).where(JobRecord.id == job_id).values(**values)
            )
            await self._session.commit()
            return await self.load(job_id)
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("Failed to update job %s", job_id)
            raise

    async def update_job_stage(
        self, job_id: UUID | str, status: ProcessingStatus, progress: int,
    ) -> None:
        uid = job_id if isinstance(job_id, UUID) else UUID(job_id)
        try:
            await self._session.execute(
                update(JobRecord)
                .where(JobRecord.id == uid)
                .values(
                    status=status.value,
                    progress=progress,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await self._session.commit()
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("Failed to update stage for job %s", uid)
            raise

    async def exists(self, job_id: UUID) -> bool:
        result = await self._session.execute(
            select(JobRecord.id).where(JobRecord.id == job_id)
        )
        return result.scalar_one_or_none() is not None

    async def delete(self, job_id: UUID) -> bool:
        record = await self._session.get(JobRecord, job_id)
        if record is None:
            return False
        await self._session.delete(record)
        await self._session.commit()
        return True

    async def update_artifacts(
        self, job_id: UUID, artifacts: MeetingArtifacts,
    ) -> JobResponse:
        try:
            await self._session.execute(
                update(JobRecord)
                .where(JobRecord.id == job_id)
                .values(
                    artifacts=artifacts.model_dump(mode="json"),
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await self._session.commit()
            job = await self.load(job_id)
            if job is None:
                raise ValueError(f"Job {job_id} disappeared after artifact update")
            return job
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("Failed to save artifacts for job %s", job_id)
            raise

    async def save_artifacts(
        self, job_id: UUID, artifacts: dict,
    ) -> None:
        """Save raw artifact dict (used by the worker)."""
        try:
            await self._session.execute(
                update(JobRecord)
                .where(JobRecord.id == job_id)
                .values(
                    artifacts=artifacts,
                    updated_at=datetime.now(timezone.utc),
                )
            )
            await self._session.commit()
        except SQLAlchemyError:
            await self._session.rollback()
            logger.exception("Failed to save raw artifacts for job %s", job_id)
            raise

    async def get_cached_artifacts(self, job_id: UUID) -> MeetingArtifacts | None:
        """Return parsed artifacts for a job, or None if absent."""
        try:
            result = await self._session.execute(
                select(JobRecord.artifacts).where(JobRecord.id == job_id)
            )
            raw = result.scalar_one_or_none()
            if raw is None:
                return None
            return MeetingArtifacts.model_validate(raw)
        except SQLAlchemyError:
            logger.exception("Failed to load cached artifacts for job %s", job_id)
            raise

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _to_response(record: JobRecord) -> JobResponse:
        artifacts = None
        if record.artifacts:
            try:
                artifacts = MeetingArtifacts.model_validate(record.artifacts)
            except Exception:
                logger.warning(
                    "Failed to parse stored artifacts for job %s", record.id,
                )

        return JobResponse(
            job_id=record.id,
            status=parse_processing_status(record.status),
            progress=record.progress,
            message=record.message,
            error=record.error,
            pdf_url=record.pdf_url,
            artifacts=artifacts,
        )
