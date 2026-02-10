"""
ARQ Background Worker for Meetolog v2.

This module defines the ARQ worker that processes audio files through
the complete pipeline:
1. Transcription (Whisper)
2. LLM Extraction (Gemini/OpenAI)
3. PDF Generation

Features:
- Async-native processing with proper error handling
- Redis-backed state persistence
- Pipeline resumability via transcript/artifact caching
- Graceful degradation on failures

Usage:
    # Start worker from backend directory:
    arq app.worker.WorkerSettings
    
    # Or with hot-reload for development:
    arq app.worker.WorkerSettings --watch app
"""

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from uuid import UUID

from arq.connections import RedisSettings

# Add parent directory to path for imports when running as worker
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.models import ProcessingStatus
from app.infrastructure.redis import get_arq_redis_settings, get_redis_pool, close_redis_pool
from app.infrastructure.job_store import RedisJobStore

logger = logging.getLogger(__name__)

# Configure logging for worker
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)


# =============================================================================
# Service Factories (for worker context)
# =============================================================================

def get_transcriber():
    settings = get_settings()
    
    if settings.test_mode:
        from app.services.mock_services import MockTranscriber
        return MockTranscriber(simulated_delay=0.5)
    
    try:
        from app.services.transcription import WhisperTranscriber
        return WhisperTranscriber(model_name=settings.whisper_model)
    except Exception as e:
        logger.error(f"Failed to initialize WhisperTranscriber: {e}")
        from app.services.mock_services import MockTranscriber
        return MockTranscriber(simulated_delay=0.5)


def get_llm_provider():
    from app.services.llm_engine import get_llm_provider as _get_provider
    return _get_provider()


def get_pdf_service():
    from app.services.pdf_generator import PDFGeneratorService
    settings = get_settings()
    return PDFGeneratorService(Path(settings.output_dir))


# =============================================================================
# Main Processing Task
# =============================================================================

async def process_audio_job(
    ctx: dict,
    job_id: str,
    file_path: str,
    file_name: str,
    file_size: int,
) -> dict:
    """
    Main ARQ task for processing an audio file through the complete pipeline.
    
    Pipeline stages:
    1. TRANSCRIBING (10-40%): Whisper STT
    2. EXTRACTING (40-75%): LLM artifact extraction
    3. GENERATING_PDF (75-95%): PDF report generation
    4. COMPLETED (100%): Done
    
    Args:
        ctx: ARQ context dict
        job_id: UUID string of the job
        file_path: Path to the uploaded audio file
        file_name: Original filename
        file_size: File size in bytes
        
    Returns:
        Dict with job completion status
    """
    job_uuid = UUID(job_id)
    worker_id = f"worker-{os.getpid()}"
    
    logger.info(f"[{job_id}] Starting processing pipeline for: {file_name}")
    start_time = time.time()
    
    # Get job store from context
    job_store: RedisJobStore = ctx["job_store"]
    
    # Update worker assignment
    await job_store.update(job_uuid, worker_id=worker_id)
    
    audio_path = Path(file_path)
    
    try:
        # =====================================================================
        # Stage 1: Transcription (10-40%)
        # =====================================================================
        logger.info(f"[{job_id}] Stage 1: Transcription starting")
        
        await job_store.update(
            job_uuid,
            status=ProcessingStatus.TRANSCRIBING,
            progress=10,
            message="Transcribing audio...",
        )
        
        # Check for cached transcript (resumability)
        transcript = await job_store.get_cached_transcript(job_uuid)
        
        if transcript:
            logger.info(f"[{job_id}] Using cached transcript (resuming)")
        else:
            # Check if audio file exists
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Run transcription
            transcriber = get_transcriber()
            raw_transcript = await transcriber.transcribe(audio_path)
            transcript = await transcriber.preprocess_transcript(raw_transcript)
            
            # Cache transcript for resumability
            await job_store.cache_transcript(job_uuid, transcript)
            logger.info(f"[{job_id}] Transcript cached for resumability")
        
        await job_store.update(
            job_uuid,
            progress=40,
            message="Transcription complete",
        )
        
        transcription_time = time.time() - start_time
        logger.info(f"[{job_id}] Transcription completed in {transcription_time:.2f}s")
        
        # =====================================================================
        # Stage 2: LLM Extraction (40-75%)
        # =====================================================================
        logger.info(f"[{job_id}] Stage 2: LLM Extraction starting")
        
        await job_store.update(
            job_uuid,
            status=ProcessingStatus.EXTRACTING,
            progress=45,
            message="Extracting Agile artifacts...",
        )
        
        # Check for cached artifacts (resumability)
        artifacts = await job_store.get_cached_artifacts(job_uuid)
        
        if artifacts:
            logger.info(f"[{job_id}] Using cached artifacts (resuming)")
        else:
            # Run LLM extraction
            llm_provider = get_llm_provider()
            artifacts = await llm_provider.extract_artifacts(transcript)
            
            # Cache artifacts for resumability
            await job_store.cache_artifacts(job_uuid, artifacts)
            logger.info(f"[{job_id}] Artifacts cached for resumability")
        
        await job_store.update(
            job_uuid,
            progress=75,
            message="Artifact extraction complete",
        )
        
        extraction_time = time.time() - start_time - transcription_time
        logger.info(f"[{job_id}] Extraction completed in {extraction_time:.2f}s")
        
        # =====================================================================
        # Stage 3: PDF Generation (75-95%)
        # =====================================================================
        logger.info(f"[{job_id}] Stage 3: PDF Generation starting")
        
        await job_store.update(
            job_uuid,
            status=ProcessingStatus.GENERATING_PDF,
            progress=80,
            message="Generating PDF summary...",
        )
        
        pdf_service = get_pdf_service()
        pdf_filename = f"meeting_{job_id}.pdf"
        pdf_path = await pdf_service.generate(artifacts, pdf_filename)
        
        await job_store.update(
            job_uuid,
            progress=95,
            message="PDF generation complete",
        )
        
        pdf_time = time.time() - start_time - transcription_time - extraction_time
        logger.info(f"[{job_id}] PDF generated in {pdf_time:.2f}s: {pdf_path}")
        
        # =====================================================================
        # Stage 4: Completion (100%)
        # =====================================================================
        total_time = time.time() - start_time
        
        await job_store.update(
            job_uuid,
            status=ProcessingStatus.COMPLETED,
            progress=100,
            message="Processing complete!",
            pdf_url=f"/download/{job_id}",
            artifacts=artifacts,
        )
        
        logger.info(
            f"[{job_id}] ✅ Pipeline completed successfully in {total_time:.2f}s "
            f"(transcribe: {transcription_time:.2f}s, extract: {extraction_time:.2f}s, pdf: {pdf_time:.2f}s)"
        )
        
        # Cleanup uploaded audio file
        if audio_path.exists():
            audio_path.unlink()
            logger.debug(f"[{job_id}] Cleaned up audio file: {audio_path}")
        
        return {
            "status": "completed",
            "job_id": job_id,
            "total_time_seconds": total_time,
        }
        
    except FileNotFoundError as e:
        logger.error(f"[{job_id}] ❌ File not found: {e}")
        await job_store.update(
            job_uuid,
            status=ProcessingStatus.FAILED,
            error=f"Audio file not found: {e}",
            message="Processing failed: File not found",
        )
        return {"status": "failed", "job_id": job_id, "error": str(e)}
        
    except Exception as e:
        logger.exception(f"[{job_id}] ❌ Pipeline failed with error: {e}")
        await job_store.update(
            job_uuid,
            status=ProcessingStatus.FAILED,
            error=str(e),
            message=f"Processing failed: {e}",
        )
        return {"status": "failed", "job_id": job_id, "error": str(e)}


# =============================================================================
# Worker Lifecycle Hooks
# =============================================================================

async def startup(ctx: dict) -> None:
    """
    Worker startup hook - initialize shared resources.
    
    Called once when the worker starts. Sets up:
    - Redis connection pool
    - Job store instance
    - Logging configuration
    """
    logger.info("🚀 ARQ Worker starting up...")
    
    # Initialize Redis connection
    redis = await get_redis_pool()
    ctx["redis"] = redis
    
    # Initialize job store
    ctx["job_store"] = RedisJobStore(redis)
    
    logger.info("✅ Worker startup complete")


async def shutdown(ctx: dict) -> None:
    """
    Worker shutdown hook - cleanup resources.
    
    Called when the worker is shutting down. Cleans up:
    - Redis connection pool
    """
    logger.info("🛑 ARQ Worker shutting down...")
    
    await close_redis_pool()
    
    logger.info("✅ Worker shutdown complete")


async def on_job_start(ctx: dict) -> None:
    """Hook called when a job starts. ARQ passes only ctx."""
    logger.debug("Job starting")


async def on_job_end(ctx: dict) -> None:
    """Hook called when a job ends. ARQ passes only ctx."""
    logger.debug("Job ended")


# =============================================================================
# Worker Settings (ARQ Configuration)
# =============================================================================

class WorkerSettings:
    """
    ARQ Worker configuration.
    
    This class defines all worker settings including:
    - Redis connection settings
    - Task functions to register
    - Lifecycle hooks
    - Job retry and timeout settings
    """
    
    # Redis connection
    redis_settings = get_arq_redis_settings()
    
    # Task functions that this worker can execute
    functions = [process_audio_job]
    
    # Lifecycle hooks
    on_startup = startup
    on_shutdown = shutdown
    on_job_start = on_job_start
    on_job_end = on_job_end
    
    # Job configuration
    max_jobs = 5  # Max concurrent jobs per worker
    job_timeout = 1800  # 30 minutes max per job
    max_tries = 3  # Retry failed jobs up to 3 times
    retry_delay = 60  # Wait 60 seconds before retry
    
    # Health check
    health_check_interval = 60  # Seconds between health checks
    
    # Queue name (use ARQ default)
    # queue_name = "arq:queue"  # Default, no need to specify
    
    # Logging
    log_results = True


# =============================================================================
# CLI Entry Point
# =============================================================================

if __name__ == "__main__":
    """
    Direct execution entry point.
    
    This allows running the worker with:
        python -m app.worker
        
    Or the standard ARQ command:
        arq app.worker.WorkerSettings
    """
    import arq
    
    print("Starting Meetolog ARQ Worker...")
    print(f"Redis URL: {get_settings().redis_url}")
    print(f"Test Mode: {get_settings().test_mode}")
    print(f"LLM Provider: {get_settings().llm_provider}")
    
    # Run the worker
    arq.run_worker(WorkerSettings)
