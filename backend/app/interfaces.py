"""
Abstract base classes (interfaces) for Meetolog services.

Defines contracts for storage and service layers, enabling
dependency injection and test mocking.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import UUID

from .models import JobResponse, MeetingArtifacts, ProcessingStatus


class JobStore(ABC):
    @abstractmethod
    async def save(self, job_id: UUID, job: JobResponse) -> None:
        ...
    
    @abstractmethod
    async def load(self, job_id: UUID) -> JobResponse | None:
        ...
    
    @abstractmethod
    async def update(self, job_id: UUID, **kwargs: Any) -> JobResponse | None:
        ...

    @abstractmethod
    async def update_job_stage(
        self, job_id: UUID | str, status: ProcessingStatus, progress: int
    ) -> None:
        """Atomically set *status* and *progress* in a single write."""
        ...
    
    @abstractmethod
    async def exists(self, job_id: UUID) -> bool:
        ...
    
    @abstractmethod
    async def delete(self, job_id: UUID) -> bool:
        ...

    @abstractmethod
    async def update_artifacts(self, job_id: UUID, artifacts: MeetingArtifacts) -> JobResponse:
        ...


class Transcriber(ABC):
    """Abstract interface for speech-to-text transcription."""
    
    @abstractmethod
    async def transcribe(
        self,
        audio_path: Path,
        progress_callback: Callable[[int, int], Awaitable[None]] | None = None,
    ) -> str:
        """
        Transcribe an audio file to text.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text content
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            RuntimeError: If transcription fails
        """
        ...
    
    @abstractmethod
    async def preprocess_transcript(self, raw_transcript: str) -> str:
        """
        Clean and format the raw transcript for LLM processing.
        
        Args:
            raw_transcript: Raw transcription output
            
        Returns:
            Cleaned and formatted transcript
        """
        ...

    async def transcribe_chunk(self, chunk_path: Path) -> str:
        """
        Transcribe a single audio chunk.

        Non-abstract default delegates to :meth:`transcribe`.
        ``WhisperTranscriber`` overrides this with chunk-specific logic.
        ``MockTranscriber`` inherits the default so it stays untouched.
        """
        return await self.transcribe(chunk_path)


class LLMExtractor(ABC):
    """Abstract interface for semantic artifact extraction using LLMs."""
    
    @abstractmethod
    async def extract_artifacts(self, transcript: str) -> MeetingArtifacts:
        """
        Extract structured Agile artifacts from a meeting transcript.
        
        Args:
            transcript: The meeting transcript text
            
        Returns:
            MeetingArtifacts with all extracted information
            
        Raises:
            RuntimeError: If extraction fails
        """
        ...
    
    @property
    @abstractmethod
    def is_mock(self) -> bool:
        """
        Indicates if this is a mock implementation.
        
        Returns:
            True if this is a mock/test implementation
        """
        ...
