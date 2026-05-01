"""Abstract base classes for the storage and service layers.

The concrete implementations live in ``app.infrastructure`` and
``app.services``. Defining the contract here keeps the rest of the
codebase decoupled from the specific backends and makes it easy to
substitute mocks during testing.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import UUID

from .models import JobResponse, MeetingArtifacts, ProcessingStatus


class JobStore(ABC):
    """Persistence layer for job state and extracted artifacts."""

    @abstractmethod
    async def save(self, job_id: UUID, job: JobResponse) -> None: ...

    @abstractmethod
    async def load(self, job_id: UUID) -> JobResponse | None: ...

    @abstractmethod
    async def update(self, job_id: UUID, **kwargs: Any) -> JobResponse | None: ...

    @abstractmethod
    async def update_job_stage(
        self, job_id: UUID | str, status: ProcessingStatus, progress: int,
    ) -> None:
        """Atomically set ``status`` and ``progress`` in a single write."""

    @abstractmethod
    async def exists(self, job_id: UUID) -> bool: ...

    @abstractmethod
    async def delete(self, job_id: UUID) -> bool: ...

    @abstractmethod
    async def update_artifacts(
        self, job_id: UUID, artifacts: MeetingArtifacts,
    ) -> JobResponse: ...


class Transcriber(ABC):
    """Speech-to-text transcription contract."""

    @abstractmethod
    async def transcribe(
        self,
        audio_path: Path,
        progress_callback: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> str:
        """Transcribe ``audio_path`` and return the raw text."""

    @abstractmethod
    async def preprocess_transcript(self, raw_transcript: str) -> str:
        """Clean and format the raw transcript for LLM processing."""

    async def transcribe_chunk(self, chunk_path: Path) -> str:
        """Transcribe a single chunk; defaults to :meth:`transcribe`."""
        return await self.transcribe(chunk_path)


class LLMExtractor(ABC):
    """Semantic artifact extraction contract."""

    @abstractmethod
    async def extract_artifacts(self, transcript: str) -> MeetingArtifacts:
        """Extract structured Agile artifacts from a meeting transcript."""

    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """True if this implementation is a deterministic mock."""
