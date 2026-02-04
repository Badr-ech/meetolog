"""
Abstract base classes (Interfaces) for Meetolog services.

These interfaces define the contracts for storage and service layers,
enabling dependency injection, test mocking, and future implementations
(e.g., Redis for JobStore, different LLM providers).

Design Pattern: Strategy Pattern + Dependency Injection
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from uuid import UUID

from .models import JobResponse, MeetingArtifacts


# =============================================================================
# Job Store Interface
# =============================================================================

class JobStore(ABC):
    """
    Abstract interface for job state persistence.
    
    Implementations:
    - LocalJobStore: In-memory dict + local JSON file backup (MVP)
    - RedisJobStore: Redis-backed storage (Version 2)
    """
    
    @abstractmethod
    async def save(self, job_id: UUID, job: JobResponse) -> None:
        """
        Save or create a new job.
        
        Args:
            job_id: Unique identifier for the job
            job: The job response object to persist
        """
        ...
    
    @abstractmethod
    async def load(self, job_id: UUID) -> JobResponse | None:
        """
        Load a job by its ID.
        
        Args:
            job_id: The job identifier to look up
            
        Returns:
            The JobResponse if found, None otherwise
        """
        ...
    
    @abstractmethod
    async def update(self, job_id: UUID, **kwargs: Any) -> JobResponse | None:
        """
        Update specific fields of an existing job.
        
        Args:
            job_id: The job identifier to update
            **kwargs: Fields to update (e.g., status, progress, message)
            
        Returns:
            The updated JobResponse if found, None otherwise
        """
        ...
    
    @abstractmethod
    async def exists(self, job_id: UUID) -> bool:
        """
        Check if a job exists.
        
        Args:
            job_id: The job identifier to check
            
        Returns:
            True if the job exists, False otherwise
        """
        ...
    
    @abstractmethod
    async def delete(self, job_id: UUID) -> bool:
        """
        Delete a job from the store.
        
        Args:
            job_id: The job identifier to delete
            
        Returns:
            True if deleted, False if not found
        """
        ...


# =============================================================================
# Transcriber Interface
# =============================================================================

class Transcriber(ABC):
    """
    Abstract interface for Speech-to-Text transcription.
    
    Implementations:
    - WhisperTranscriber: Local OpenAI Whisper model (production)
    - MockTranscriber: Returns hardcoded text for testing (CI mode)
    - DeepgramTranscriber: Cloud-based Deepgram API (Version 2)
    """
    
    @abstractmethod
    async def transcribe(self, audio_path: Path) -> str:
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


# =============================================================================
# LLM Extractor Interface
# =============================================================================

class LLMExtractor(ABC):
    """
    Abstract interface for semantic artifact extraction using LLMs.
    
    Implementations:
    - GeminiExtractor: Google Gemini API (production)
    - MockExtractor: Returns deterministic JSON for testing (CI mode)
    - OpenAIExtractor: OpenAI GPT API (Version 2)
    """
    
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
