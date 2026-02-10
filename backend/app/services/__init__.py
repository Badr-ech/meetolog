"""
Services layer for business logic (v2).

This module exports both abstract interfaces and concrete implementations.
Use the factory functions in dependencies.py for proper instantiation.

Note: Imports are lazy to allow TEST_MODE to work even if some 
production dependencies (e.g., reportlab, whisper) are not installed.

v2 Additions:
- LLMProvider abstraction layer (llm_engine.py)
- GeminiProvider, OpenAIProvider implementations
- get_llm_provider() factory function
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
    # v2 LLM abstraction layer
    elif name == "LLMProvider":
        from .llm_engine import LLMProvider
        return LLMProvider
    elif name == "GeminiProvider":
        from .llm_engine import GeminiProvider
        return GeminiProvider
    elif name == "OpenAIProvider":
        from .llm_engine import OpenAIProvider
        return OpenAIProvider
    elif name == "get_llm_provider":
        from .llm_engine import get_llm_provider
        return get_llm_provider
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
    # v2 LLM abstraction
    "LLMProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "get_llm_provider",
    # Legacy aliases
    "TranscriptionService",
    "LLMExtractionService",
]
