"""
Service factory and dependency injection.

Implements the Factory Pattern to create service instances
based on configuration (TEST_MODE, API key availability).
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from .config import get_settings, Settings
from .interfaces import JobStore, Transcriber, LLMExtractor

logger = logging.getLogger(__name__)

_job_store: JobStore | None = None
_transcriber: Transcriber | None = None
_extractor: LLMExtractor | None = None


def create_job_store(settings: Settings) -> JobStore:
    from .services.job_store import LocalJobStore
    
    logger.info("Creating LocalJobStore instance")
    return LocalJobStore(
        persist_to_file=True,
        storage_dir=Path(settings.output_dir),
        jobs_filename="jobs_state.json",
    )


def create_transcriber(settings: Settings) -> Transcriber:
    if settings.test_mode:
        from .services.mock_services import MockTranscriber
        
        logger.info("TEST_MODE enabled: Using MockTranscriber")
        return MockTranscriber(simulated_delay=0.5)
    
    try:
        from .services.transcription import WhisperTranscriber
        
        logger.info("Using WhisperTranscriber (production mode)")
        return WhisperTranscriber(model_name=settings.whisper_model)
        
    except Exception as e:
        logger.error(f"Failed to initialize WhisperTranscriber: {e}")
        
        from .services.mock_services import MockTranscriber
        
        logger.warning("Falling back to MockTranscriber due to initialization failure")
        return MockTranscriber(simulated_delay=0.5)


def create_extractor(settings: Settings) -> LLMExtractor:
    """Create an LLMExtractor: MockExtractor in test mode or when API key is missing, else GeminiExtractor."""
    if settings.test_mode:
        from .services.mock_services import MockExtractor
        
        logger.info("TEST_MODE enabled: Using MockExtractor")
        return MockExtractor(simulated_delay=0.3)
    
    # Check for API key
    if not settings.gemini_api_key:
        from .services.mock_services import MockExtractor
        
        logger.warning(
            "GEMINI_API_KEY is not set. "
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


def get_job_store() -> JobStore:
    global _job_store
    
    if _job_store is None:
        settings = get_settings()
        _job_store = create_job_store(settings)
    
    return _job_store


def get_transcriber() -> Transcriber:
    global _transcriber
    
    if _transcriber is None:
        settings = get_settings()
        _transcriber = create_transcriber(settings)
    
    return _transcriber


def get_extractor() -> LLMExtractor:
    global _extractor
    
    if _extractor is None:
        settings = get_settings()
        _extractor = create_extractor(settings)
    
    return _extractor


JobStoreDep = Annotated[JobStore, Depends(get_job_store)]
TranscriberDep = Annotated[Transcriber, Depends(get_transcriber)]
LLMExtractorDep = Annotated[LLMExtractor, Depends(get_extractor)]


async def initialize_services() -> None:
    settings = get_settings()
    
    logger.info("=" * 60)
    logger.info("Initializing Meetolog services...")
    logger.info(f"  TEST_MODE: {settings.test_mode}")
    logger.info(f"  LLM_PROVIDER: {settings.llm_provider}")
    logger.info(f"  GEMINI_API_KEY: {'[SET]' if settings.gemini_api_key else '[NOT SET]'}")
    logger.info(f"  OPENAI_API_KEY: {'[SET]' if settings.openai_api_key else '[NOT SET]'}")
    logger.info(f"  REDIS_URL: {settings.redis_url}")
    logger.info("=" * 60)
    
    if settings.test_mode:
        transcriber = get_transcriber()
        extractor = get_extractor()
        
        logger.info(f"  Transcriber: {type(transcriber).__name__}")
        logger.info(f"  Extractor: {type(extractor).__name__} (mock={extractor.is_mock})")
    
    logger.info("Services initialized successfully!")


def reset_services() -> None:
    global _job_store, _transcriber, _extractor
    
    _job_store = None
    _transcriber = None
    _extractor = None
    
    logger.info("All services have been reset")
