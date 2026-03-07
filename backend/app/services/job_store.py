"""
Local Job Store — in-memory dict with optional JSON file persistence.

Suited for single-instance deployments and development.
For production, use RedisJobStore.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import aiofiles

from ..interfaces import JobStore
from ..models import JobResponse, ProcessingStatus, MeetingArtifacts

logger = logging.getLogger(__name__)


class LocalJobStore(JobStore):
    """
    Job store using in-memory dictionary with optional JSON file backup.
    
    Architecture:
    - Primary: In-memory dict for fast access
    - Secondary: JSON file backup for crash recovery (optional)
    
    Thread Safety:
    - Uses asyncio.Lock for concurrent access protection
    - Safe for single-worker FastAPI deployment
    
    Limitations:
    - Single instance only (no horizontal scaling)
    - Memory-bound (large job counts may cause issues)
    - File I/O is async to not block event loop
    """
    
    def __init__(
        self,
        persist_to_file: bool = True,
        storage_dir: Path | str = "outputs",
        jobs_filename: str = "jobs_state.json",
    ):
        """
        Initialize the local job store.
        
        Args:
            persist_to_file: Whether to backup state to JSON file
            storage_dir: Directory for the jobs state file
            jobs_filename: Name of the JSON backup file
        """
        self._jobs: dict[UUID, JobResponse] = {}
        self._lock = asyncio.Lock()
        self._persist = persist_to_file
        
        if persist_to_file:
            self._storage_dir = Path(storage_dir)
            self._storage_dir.mkdir(parents=True, exist_ok=True)
            self._jobs_file = self._storage_dir / jobs_filename
            logger.info(f"LocalJobStore initialized with persistence: {self._jobs_file}")
        else:
            self._jobs_file = None
            logger.info("LocalJobStore initialized (in-memory only, no persistence)")
    
    async def save(self, job_id: UUID, job: JobResponse) -> None:
        """Save or create a new job."""
        async with self._lock:
            self._jobs[job_id] = job
            logger.debug(f"Job {job_id} saved with status: {job.status.value}")
            
            if self._persist:
                await self._persist_to_disk()
    
    async def load(self, job_id: UUID) -> JobResponse | None:
        """Load a job by its ID."""
        async with self._lock:
            job = self._jobs.get(job_id)
            if job:
                logger.debug(f"Job {job_id} loaded with status: {job.status.value}")
            else:
                logger.debug(f"Job {job_id} not found")
            return job
    
    async def update(self, job_id: UUID, **kwargs: Any) -> JobResponse | None:
        """Update specific fields of an existing job."""
        async with self._lock:
            if job_id not in self._jobs:
                logger.warning(f"Attempted to update non-existent job: {job_id}")
                return None
            
            job = self._jobs[job_id]
            
            # Update allowed fields
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
                else:
                    logger.warning(f"Attempted to set unknown field '{key}' on job {job_id}")
            
            logger.debug(f"Job {job_id} updated: {list(kwargs.keys())}")
            
            if self._persist:
                await self._persist_to_disk()
            
            return job
    
    async def exists(self, job_id: UUID) -> bool:
        """Check if a job exists."""
        async with self._lock:
            return job_id in self._jobs
    
    async def delete(self, job_id: UUID) -> bool:
        """Delete a job from the store."""
        async with self._lock:
            if job_id in self._jobs:
                del self._jobs[job_id]
                logger.info(f"Job {job_id} deleted")
                
                if self._persist:
                    await self._persist_to_disk()
                
                return True
            
            logger.warning(f"Attempted to delete non-existent job: {job_id}")
            return False
    
    async def _persist_to_disk(self) -> None:
        """
        Persist current state to JSON file.
        
        Uses aiofiles for non-blocking file I/O.
        """
        if not self._jobs_file:
            return
        
        try:
            # Serialize jobs to JSON-safe format
            serialized = {}
            for job_id, job in self._jobs.items():
                # Convert to dict, handling nested models
                job_dict = job.model_dump(mode="json")
                serialized[str(job_id)] = job_dict
            
            # Write atomically via temp file
            temp_file = self._jobs_file.with_suffix(".tmp")
            
            async with aiofiles.open(temp_file, "w", encoding="utf-8") as f:
                await f.write(json.dumps(serialized, indent=2, default=str))
            
            # Atomic rename
            temp_file.replace(self._jobs_file)
            logger.debug(f"Job state persisted to {self._jobs_file}")
            
        except Exception as e:
            logger.error(f"Failed to persist job state: {e}")
    
    async def load_from_disk(self) -> int:
        """
        Load job state from disk on startup.
        
        Returns:
            Number of jobs loaded
        """
        if not self._jobs_file or not self._jobs_file.exists():
            logger.info("No persisted job state found")
            return 0
        
        try:
            async with aiofiles.open(self._jobs_file, "r", encoding="utf-8") as f:
                content = await f.read()
            
            data = json.loads(content)
            
            loaded_count = 0
            for job_id_str, job_dict in data.items():
                try:
                    job_id = UUID(job_id_str)
                    
                    # Reconstruct JobResponse
                    # Handle nested MeetingArtifacts if present
                    if job_dict.get("artifacts"):
                        job_dict["artifacts"] = MeetingArtifacts(**job_dict["artifacts"])
                    
                    # Convert status string to enum
                    if isinstance(job_dict.get("status"), str):
                        job_dict["status"] = ProcessingStatus(job_dict["status"])
                    
                    job = JobResponse(**job_dict)
                    self._jobs[job_id] = job
                    loaded_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to load job {job_id_str}: {e}")
            
            logger.info(f"Loaded {loaded_count} jobs from disk")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Failed to load job state from disk: {e}")
            return 0
    
    def get_all_jobs(self) -> dict[UUID, JobResponse]:
        """
        Get all jobs (for debugging/admin purposes).
        
        Note: This is a synchronous method that returns a copy.
        """
        return dict(self._jobs)
    
    def job_count(self) -> int:
        """Get the current number of jobs in the store."""
        return len(self._jobs)
