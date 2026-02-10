"""
Redis-backed Job Store for Meetolog v2.

Implements the JobStore interface using Redis Hashes for persistent
job state storage. Features:

- Redis Hash for job metadata (status, progress, step, error, etc.)
- Separate keys for transcript and artifact caching
- 7-day TTL with automatic refresh on reads
- Pipeline caching for resumability

Redis Key Schema:
- job:{uuid}           - Hash with job metadata
- job:{uuid}:transcript - String with cached transcript
- job:{uuid}:artifacts  - JSON string with extracted artifacts

See TECHNICAL_DESIGN_V2.md Section 2 for full schema details.
"""

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

import ujson

from redis.asyncio import Redis

from ..interfaces import JobStore
from ..models import JobResponse, ProcessingStatus, MeetingArtifacts
from ..config import get_settings
from .redis import get_redis_pool

logger = logging.getLogger(__name__)

# TTL Constants (in seconds)
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days
FAILED_JOB_TTL_SECONDS = 3 * 24 * 60 * 60  # 3 days for failed jobs


class RedisJobStore(JobStore):
    def __init__(self, redis: Redis | None = None):
        self._redis = redis
        self._ttl_seconds = DEFAULT_TTL_SECONDS
        
        settings = get_settings()
        self._ttl_seconds = settings.redis_job_ttl_days * 24 * 60 * 60
        
        logger.debug(f"RedisJobStore initialized (TTL: {settings.redis_job_ttl_days} days)")
    
    async def _get_redis(self) -> Redis:
        if self._redis is not None:
            return self._redis
        return await get_redis_pool()
    
    @staticmethod
    def _job_key(job_id: UUID | str) -> str:
        return f"job:{job_id}"
    
    @staticmethod
    def _transcript_key(job_id: UUID | str) -> str:
        return f"job:{job_id}:transcript"
    
    @staticmethod
    def _artifacts_key(job_id: UUID | str) -> str:
        return f"job:{job_id}:artifacts"
    
    def _get_ttl(self, status: ProcessingStatus) -> int:
        if status == ProcessingStatus.FAILED:
            return FAILED_JOB_TTL_SECONDS
        return self._ttl_seconds
    
    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
    
    async def save(
        self,
        job_id: UUID,
        job: JobResponse,
        file_path: str | None = None,
        file_name: str | None = None,
        file_size: int | None = None,
    ) -> None:
        """
        Save or create a new job in Redis.
        
        Args:
            job_id: Unique identifier for the job
            job: The job response object to persist
            file_path: Optional path to uploaded audio file
            file_name: Optional original filename
            file_size: Optional file size in bytes
        """
        redis = await self._get_redis()
        key = self._job_key(job_id)
        now = self._now_iso()
        
        # Build hash data
        data = {
            "status": job.status.value,
            "progress": str(job.progress),
            "message": job.message or "",
            "error": job.error or "",
            "pdf_url": job.pdf_url or "",
            "created_at": now,
            "updated_at": now,
            "started_at": "",
            "completed_at": "",
            "retry_count": "0",
            "worker_id": "",
            "has_transcript": "0",
            "has_artifacts": "0",
        }
        
        if file_path:
            data["file_path"] = file_path
        if file_name:
            data["file_name"] = file_name
        if file_size is not None:
            data["file_size"] = str(file_size)
        
        # Use pipeline for atomicity
        async with redis.pipeline() as pipe:
            pipe.hset(key, mapping=data)
            pipe.expire(key, self._get_ttl(job.status))
            await pipe.execute()
        
        logger.debug(f"Job {job_id} saved to Redis with status: {job.status.value}")
    
    async def load(self, job_id: UUID) -> JobResponse | None:
        """
        Load a job by its ID with TTL refresh.
        
        Args:
            job_id: The job identifier to look up
            
        Returns:
            The JobResponse if found, None otherwise
        """
        redis = await self._get_redis()
        key = self._job_key(job_id)
        
        # Use pipeline to get data and refresh TTL atomically
        async with redis.pipeline() as pipe:
            pipe.hgetall(key)
            pipe.expire(key, self._ttl_seconds)  # Refresh TTL on read
            results = await pipe.execute()
        
        data = results[0]
        
        if not data:
            logger.debug(f"Job {job_id} not found in Redis")
            return None
        
        # Parse status enum
        try:
            status = ProcessingStatus(data.get("status", "pending"))
        except ValueError:
            status = ProcessingStatus.PENDING
        
        # Load artifacts if cached
        artifacts = None
        if data.get("has_artifacts") == "1":
            artifacts = await self.get_cached_artifacts(job_id)
        
        job = JobResponse(
            job_id=job_id,
            status=status,
            progress=int(data.get("progress", 0)),
            message=data.get("message", ""),
            error=data.get("error") or None,
            pdf_url=data.get("pdf_url") or None,
            artifacts=artifacts,
        )
        
        logger.debug(f"Job {job_id} loaded from Redis with status: {status.value}")
        return job
    
    async def update(self, job_id: UUID, **kwargs: Any) -> JobResponse | None:
        """
        Update specific fields of an existing job.
        
        Args:
            job_id: The job identifier to update
            **kwargs: Fields to update. Supported:
                - status (ProcessingStatus)
                - progress (int)
                - message (str)
                - error (str)
                - pdf_url (str)
                - artifacts (MeetingArtifacts) - also caches separately
                - worker_id (str)
                
        Returns:
            The updated JobResponse if found, None otherwise
        """
        redis = await self._get_redis()
        key = self._job_key(job_id)
        
        # Check if job exists
        if not await redis.exists(key):
            logger.warning(f"Attempted to update non-existent job: {job_id}")
            return None
        
        # Build update data
        updates = {"updated_at": self._now_iso()}
        
        if "status" in kwargs:
            status = kwargs["status"]
            if isinstance(status, ProcessingStatus):
                updates["status"] = status.value
            else:
                updates["status"] = str(status)
            
            # Track timing milestones
            if status == ProcessingStatus.TRANSCRIBING:
                updates["started_at"] = self._now_iso()
            elif status in (ProcessingStatus.COMPLETED, ProcessingStatus.FAILED):
                updates["completed_at"] = self._now_iso()
        
        if "progress" in kwargs:
            updates["progress"] = str(kwargs["progress"])
        
        if "message" in kwargs:
            updates["message"] = kwargs["message"] or ""
        
        if "error" in kwargs:
            updates["error"] = kwargs["error"] or ""
        
        if "pdf_url" in kwargs:
            updates["pdf_url"] = kwargs["pdf_url"] or ""
        
        if "worker_id" in kwargs:
            updates["worker_id"] = kwargs["worker_id"] or ""
        
        # Handle artifacts caching
        if "artifacts" in kwargs and kwargs["artifacts"] is not None:
            await self.cache_artifacts(job_id, kwargs["artifacts"])
            updates["has_artifacts"] = "1"
        
        # Determine TTL based on new status
        ttl = self._ttl_seconds
        if "status" in kwargs:
            status_val = kwargs["status"]
            if isinstance(status_val, ProcessingStatus):
                ttl = self._get_ttl(status_val)
        
        # Update atomically
        async with redis.pipeline() as pipe:
            pipe.hset(key, mapping=updates)
            pipe.expire(key, ttl)
            await pipe.execute()
        
        logger.debug(f"Job {job_id} updated: {list(updates.keys())}")
        
        # Return updated job
        return await self.load(job_id)
    
    async def exists(self, job_id: UUID) -> bool:
        """Check if a job exists."""
        redis = await self._get_redis()
        return bool(await redis.exists(self._job_key(job_id)))
    
    async def delete(self, job_id: UUID) -> bool:
        """
        Delete a job and all associated cached data.
        
        Args:
            job_id: The job identifier to delete
            
        Returns:
            True if deleted, False if not found
        """
        redis = await self._get_redis()
        
        # Delete all related keys
        keys_to_delete = [
            self._job_key(job_id),
            self._transcript_key(job_id),
            self._artifacts_key(job_id),
        ]
        
        deleted = await redis.delete(*keys_to_delete)
        
        if deleted > 0:
            logger.info(f"Job {job_id} deleted from Redis")
            return True
        
        logger.warning(f"Attempted to delete non-existent job: {job_id}")
        return False
    
    # =========================================================================
    # Caching Methods (for pipeline resumability)
    # =========================================================================
    
    async def cache_transcript(self, job_id: UUID, transcript: str) -> None:
        """
        Cache the transcript for a job.
        
        Enables pipeline resumability - if extraction fails, we can
        retry from the cached transcript without re-transcribing.
        
        Args:
            job_id: The job identifier
            transcript: The transcript text to cache
        """
        redis = await self._get_redis()
        
        async with redis.pipeline() as pipe:
            pipe.set(self._transcript_key(job_id), transcript)
            pipe.expire(self._transcript_key(job_id), self._ttl_seconds)
            pipe.hset(self._job_key(job_id), "has_transcript", "1")
            await pipe.execute()
        
        logger.debug(f"Transcript cached for job {job_id}")
    
    async def get_cached_transcript(self, job_id: UUID) -> str | None:
        """
        Get cached transcript for a job.
        
        Args:
            job_id: The job identifier
            
        Returns:
            Cached transcript or None if not cached
        """
        redis = await self._get_redis()
        
        async with redis.pipeline() as pipe:
            pipe.get(self._transcript_key(job_id))
            pipe.expire(self._transcript_key(job_id), self._ttl_seconds)  # Refresh TTL
            results = await pipe.execute()
        
        return results[0]
    
    async def cache_artifacts(self, job_id: UUID, artifacts: MeetingArtifacts) -> None:
        """
        Cache extracted artifacts for a job.
        
        Enables pipeline resumability - if PDF generation fails, we can
        retry from cached artifacts without re-extracting.
        
        Args:
            job_id: The job identifier
            artifacts: The extracted artifacts to cache
        """
        redis = await self._get_redis()
        
        # Serialize artifacts to JSON using ujson for speed
        json_data = ujson.dumps(artifacts.model_dump(mode="json"))
        
        async with redis.pipeline() as pipe:
            pipe.set(self._artifacts_key(job_id), json_data)
            pipe.expire(self._artifacts_key(job_id), self._ttl_seconds)
            pipe.hset(self._job_key(job_id), "has_artifacts", "1")
            await pipe.execute()
        
        logger.debug(f"Artifacts cached for job {job_id}")
    
    async def get_cached_artifacts(self, job_id: UUID) -> MeetingArtifacts | None:
        """
        Get cached artifacts for a job.
        
        Args:
            job_id: The job identifier
            
        Returns:
            Cached MeetingArtifacts or None if not cached
        """
        redis = await self._get_redis()
        
        async with redis.pipeline() as pipe:
            pipe.get(self._artifacts_key(job_id))
            pipe.expire(self._artifacts_key(job_id), self._ttl_seconds)  # Refresh TTL
            results = await pipe.execute()
        
        json_data = results[0]
        
        if not json_data:
            return None
        
        try:
            data = ujson.loads(json_data)
            return MeetingArtifacts.model_validate(data)
        except Exception as e:
            logger.error(f"Failed to deserialize cached artifacts for {job_id}: {e}")
            return None
    
    async def get_file_path(self, job_id: UUID) -> str | None:
        """
        Get the stored file path for a job.
        
        Args:
            job_id: The job identifier
            
        Returns:
            File path or None if not found
        """
        redis = await self._get_redis()
        return await redis.hget(self._job_key(job_id), "file_path")
