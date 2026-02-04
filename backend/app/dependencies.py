"""
Service Factory and Dependency Injection for Meetolog.

This module implements the Factory Pattern to create service instances
based on configuration. It handles:

1. TEST_MODE detection - Use mocks for CI/CD and testing
2. API key validation - Graceful degradation if keys are missing
3. Singleton patterns - Reuse service instances efficiently

Usage in FastAPI:
    from .dependencies import get_transcriber, get_extractor, get_job_store
    
    @app.post("/upload")
    async def upload(
        transcriber: Annotated[Transcriber, Depends(get_transcriber)],
        extractor: Annotated[LLMExtractor, Depends(get_extractor)],
        job_store: Annotated[JobStore, Depends(get_job_store)],
    ):
        ...
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from .config import get_settings, Settings
from .interfaces import JobStore, Transcriber, LLMExtractor

logger = logging.getLogger(__name__)

# =============================================================================
# Singleton Service Instances
# =============================================================================

_job_store: JobStore | None = None
_transcriber: Transcriber | None = None
_extractor: LLMExtractor | None = None


# =============================================================================
# Factory Functions
# =============================================================================

def create_job_store(settings: Settings) -> JobStore:
    """
    Create and return a JobStore instance.
    
    Currently only LocalJobStore is implemented.
    Redis support can be added by checking settings.redis_url.
    """
    from .services.job_store import LocalJobStore
    
    logger.info("Creating LocalJobStore instance")
    return LocalJobStore(
        persist_to_file=True,
        storage_dir=Path(settings.output_dir),
        jobs_filename="jobs_state.json",
    )


def create_transcriber(settings: Settings) -> Transcriber:
    """
    Create and return a Transcriber instance based on configuration.
    
    Logic:
    1. If TEST_MODE=true → MockTranscriber
    2. Otherwise → WhisperTranscriber (production)
    """
    if settings.test_mode:
        from .services.mock_services import MockTranscriber
        
        logger.info("TEST_MODE enabled: Using MockTranscriber")
        return MockTranscriber(simulated_delay=0.5)
    
    # Production mode - use real Whisper
    try:
        from .services.transcription import WhisperTranscriber
        
        logger.info("Using WhisperTranscriber (production mode)")
        return WhisperTranscriber(model_name=settings.whisper_model)
        
    except Exception as e:
        logger.error(f"Failed to initialize WhisperTranscriber: {e}")
        
        # Graceful degradation to mock
        from .services.mock_services import MockTranscriber
        
        logger.warning("Falling back to MockTranscriber due to initialization failure")
        return MockTranscriber(simulated_delay=0.5)


def create_extractor(settings: Settings) -> LLMExtractor:
    """
    Create and return an LLMExtractor instance based on configuration.
    
    Logic:
    1. If TEST_MODE=true → MockExtractor
    2. If GEMINI_API_KEY is missing → MockExtractor with warning
    3. Otherwise → GeminiExtractor (production)
    """
    if settings.test_mode:
        from .services.mock_services import MockExtractor
        
        logger.info("TEST_MODE enabled: Using MockExtractor")
        return MockExtractor(simulated_delay=0.3)
    
    # Check for API key
    if not settings.gemini_api_key:
        from .services.mock_services import MockExtractor
        
        logger.warning(
            "⚠️  GEMINI_API_KEY is not set! "
            "The application will use MockExtractor (deterministic test data). "
            "To use real LLM extraction, set the GEMINI_API_KEY environment variable."
        )
        return MockExtractor(simulated_delay=0.3)
    
    # Production mode - use real Gemini
    try:
        from .services.llm_extraction import GeminiExtractor
        
        logger.info("Using GeminiExtractor (production mode)")
        return GeminiExtractor(api_key=settings.gemini_api_key)
        
    except Exception as e:
        logger.error(f"Failed to initialize GeminiExtractor: {e}")
        
        # Graceful degradation to mock
        from .services.mock_services import MockExtractor
        
        logger.warning("Falling back to MockExtractor due to initialization failure")
        return MockExtractor(simulated_delay=0.3)


# =============================================================================
# Dependency Injection Functions (for FastAPI Depends)
# =============================================================================

def get_job_store() -> JobStore:
    """
    Get the singleton JobStore instance.
    
    FastAPI Dependency that returns the configured JobStore.
    """
    global _job_store
    
    if _job_store is None:
        settings = get_settings()
        _job_store = create_job_store(settings)
    
    return _job_store


def get_transcriber() -> Transcriber:
    """
    Get the singleton Transcriber instance.
    
    FastAPI Dependency that returns the configured Transcriber.
    """
    global _transcriber
    
    if _transcriber is None:
        settings = get_settings()
        _transcriber = create_transcriber(settings)
    
    return _transcriber


def get_extractor() -> LLMExtractor:
    """
    Get the singleton LLMExtractor instance.
    
    FastAPI Dependency that returns the configured LLMExtractor.
    """
    global _extractor
    
    if _extractor is None:
        settings = get_settings()
        _extractor = create_extractor(settings)
    
    return _extractor


# =============================================================================
# Type Aliases for FastAPI Depends
# =============================================================================

# Use these in endpoint function signatures for cleaner code:
# async def endpoint(job_store: JobStoreDep, transcriber: TranscriberDep):

JobStoreDep = Annotated[JobStore, Depends(get_job_store)]
TranscriberDep = Annotated[Transcriber, Depends(get_transcriber)]
LLMExtractorDep = Annotated[LLMExtractor, Depends(get_extractor)]


# =============================================================================
# Initialization & Cleanup
# =============================================================================

async def initialize_services() -> None:
    """
    Initialize all services on application startup.
    
    Called from FastAPI lifespan event.
    """
    settings = get_settings()
    
    logger.info("=" * 60)
    logger.info("Initializing Meetolog services...")
    logger.info(f"  TEST_MODE: {settings.test_mode}")
    logger.info(f"  GEMINI_API_KEY: {'[SET]' if settings.gemini_api_key else '[NOT SET]'}")
    logger.info("=" * 60)
    
    # Initialize job store and load persisted state
    job_store = get_job_store()
    
    # Import LocalJobStore type for type checking
    from .services.job_store import LocalJobStore
    if isinstance(job_store, LocalJobStore):
        loaded = await job_store.load_from_disk()
        logger.info(f"Loaded {loaded} persisted jobs from disk")
    
    # Pre-initialize other services
    transcriber = get_transcriber()
    extractor = get_extractor()
    
    logger.info(f"  Transcriber: {type(transcriber).__name__}")
    logger.info(f"  Extractor: {type(extractor).__name__} (mock={extractor.is_mock})")
    logger.info("Services initialized successfully!")


def reset_services() -> None:
    """
    Reset all singleton services.
    
    Useful for testing to ensure clean state between tests.
    """
    global _job_store, _transcriber, _extractor
    
    _job_store = None
    _transcriber = None
    _extractor = None
    
    logger.info("All services have been reset")
