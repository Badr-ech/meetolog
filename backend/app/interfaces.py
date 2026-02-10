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
    async def exists(self, job_id: UUID) -> bool:
        ...
    
    @abstractmethod
    async def delete(self, job_id: UUID) -> bool:
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
