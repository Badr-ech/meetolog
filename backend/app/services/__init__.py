"""
Services layer for business logic.

This module exports both abstract interfaces and concrete implementations.
Use the factory functions in dependencies.py for proper instantiation.

Note: Imports are lazy to allow TEST_MODE to work even if some 
production dependencies (e.g., reportlab, whisper) are not installed.
"""

# These are always available (no external deps)
from .mock_services import MockTranscriber, MockExtractor
from .job_store import LocalJobStore

# Production services - imported on demand to allow graceful degradation


def __getattr__(name: str):
    """Lazy import of services that require external dependencies."""
    if name == "WhisperTranscriber":
        from .transcription import WhisperTranscriber
        return WhisperTranscriber
    elif name == "GeminiExtractor":
        from .llm_extraction import GeminiExtractor
        return GeminiExtractor
    elif name == "PDFGeneratorService":
        from .pdf_generator import PDFGeneratorService
        return PDFGeneratorService
    elif name == "TranscriptionService":
        # Legacy alias
        from .transcription import WhisperTranscriber
        return WhisperTranscriber
    elif name == "LLMExtractionService":
        # Legacy alias
        from .llm_extraction import GeminiExtractor
        return GeminiExtractor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Production services
    "WhisperTranscriber",
    "GeminiExtractor",
    "PDFGeneratorService",
    "LocalJobStore",
    # Mock services
    "MockTranscriber",
    "MockExtractor",
    # Legacy aliases
    "TranscriptionService",
    "LLMExtractionService",
]
