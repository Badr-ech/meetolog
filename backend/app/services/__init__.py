"""
Services layer — business logic implementations.

Imports are lazy so TEST_MODE works even when production
dependencies (reportlab, whisper) are not installed.
"""

# Always available (no external deps)
from .mock_services import MockTranscriber, MockExtractor


def __getattr__(name: str):
    """Lazy import of services that require external dependencies."""
    if name == "WhisperTranscriber":
        from .transcription import WhisperTranscriber
        return WhisperTranscriber
    elif name == "HierarchicalExtractor":
        from .llm_extraction import HierarchicalExtractor
        return HierarchicalExtractor
    elif name == "PDFGeneratorService":
        from .pdf_generator import PDFGeneratorService
        return PDFGeneratorService
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
    elif name == "ContextCompressor":
        from .compression import ContextCompressor
        return ContextCompressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "WhisperTranscriber",
    "HierarchicalExtractor",
    "PDFGeneratorService",
    "MockTranscriber",
    "MockExtractor",
    "LLMProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "get_llm_provider",
    "ContextCompressor",
]
