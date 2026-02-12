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
import shutil
import sys
import tempfile
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
        # Stage 1: Transcription (10-40%) — chunked for restart resilience
        # =====================================================================
        logger.info(f"[{job_id}] Stage 1: Transcription starting")
        
        await job_store.update(
            job_uuid,
            status=ProcessingStatus.TRANSCRIBING,
            progress=10,
            message="Transcribing audio...",
        )
        
        # Check for cached full transcript (resumability from prior run)
        transcript = await job_store.get_cached_transcript(job_uuid)
        
        if transcript:
            logger.info(f"[{job_id}] Using cached transcript (resuming)")
        else:
            # -----------------------------------------------------------------
            # Resolve audio source: local file or Redis-stored compressed audio
            # -----------------------------------------------------------------
            work_dir = Path(tempfile.mkdtemp(prefix=f"meetolog_{job_id}_"))
            
            try:
                if audio_path.exists():
                    # Fresh job — audio file is on disk
                    effective_audio = audio_path
                    
                    # Compress & store in Redis for restart resilience
                    try:
                        from app.services.transcription import compress_audio_for_storage
                        
                        logger.info(f"[{job_id}] Compressing audio for Redis backup...")
                        compressed = await asyncio.to_thread(
                            compress_audio_for_storage, audio_path
                        )
                        stored = await job_store.store_audio(job_uuid, compressed)
                        if stored:
                            logger.info(f"[{job_id}] Audio backed up to Redis")
                        else:
                            logger.warning(f"[{job_id}] Audio too large for Redis backup")
                    except Exception as e:
                        logger.warning(f"[{job_id}] Failed to backup audio to Redis: {e}")
                else:
                    # Resumed job — try to restore from Redis
                    stored_audio = await job_store.get_stored_audio(job_uuid)
                    if stored_audio is None:
                        raise FileNotFoundError(
                            f"Audio file not found on disk or in Redis: {audio_path}"
                        )
                    
                    logger.info(f"[{job_id}] Restoring audio from Redis backup...")
                    from app.services.transcription import decompress_audio
                    
                    restored_path = work_dir / "restored_audio.wav"
                    await asyncio.to_thread(
                        decompress_audio, stored_audio, restored_path
                    )
                    effective_audio = restored_path
                    logger.info(f"[{job_id}] Audio restored from Redis")
                
                # -----------------------------------------------------------------
                # Split into chunks and transcribe incrementally
                # -----------------------------------------------------------------
                from app.services.transcription import (
                    split_audio_into_chunks,
                    get_audio_duration,
                )
                
                chunk_dir = work_dir / "chunks"
                chunks = await asyncio.to_thread(
                    split_audio_into_chunks, effective_audio, chunk_dir
                )
                total_chunks = len(chunks)
                logger.info(f"[{job_id}] Audio split into {total_chunks} chunks")
                
                # Check which chunks were already transcribed (prior partial run)
                completed_indices, _ = await job_store.get_completed_chunk_indices(job_uuid)
                
                if completed_indices:
                    logger.info(
                        f"[{job_id}] Resuming: {len(completed_indices)}/{total_chunks} "
                        f"chunks already done"
                    )
                
                # Transcribe remaining chunks
                transcriber = get_transcriber()
                
                for i, chunk_path in enumerate(chunks):
                    if i in completed_indices:
                        continue
                    
                    # Progress: spread 10-38% across chunks
                    chunk_progress = 10 + int(28 * (i / total_chunks))
                    await job_store.update(
                        job_uuid,
                        progress=chunk_progress,
                        message=f"Transcribing chunk {i + 1}/{total_chunks}...",
                    )
                    
                    chunk_text = await transcriber.transcribe_chunk(chunk_path)
                    chunk_text = await transcriber.preprocess_transcript(chunk_text)

                    # Cache immediately — survives restarts
                    await job_store.save_chunk_transcript(
                        job_uuid, i, chunk_text, total_chunks
                    )
                    logger.info(
                        f"[{job_id}] Chunk {i + 1}/{total_chunks} transcribed and cached"
                    )
                
                # Assemble full transcript from all chunks
                transcript = await job_store.assemble_transcript_from_chunks(job_uuid)
                if not transcript:
                    raise RuntimeError("Failed to assemble transcript from chunks")
                
                # Cache the full transcript and clean up chunk/audio data
                await job_store.cache_transcript(job_uuid, transcript)
                await job_store.delete_chunk_data(job_uuid)
                logger.info(f"[{job_id}] Full transcript cached, chunk data cleaned up")
                
            finally:
                # Clean up temp directory
                shutil.rmtree(work_dir, ignore_errors=True)
        
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
    - Zombie job recovery with resumability
    - Logging configuration
    """
    logger.info("🚀 ARQ Worker starting up...")
    
    # Initialize Redis connection
    redis = await get_redis_pool()
    ctx["redis"] = redis
    
    # Initialize job store
    job_store = RedisJobStore(redis)
    ctx["job_store"] = job_store
    
    # Recover zombie jobs (jobs interrupted by restart)
    try:
        from arq import create_pool
        
        stale_jobs = await job_store.find_stale_jobs()
        
        if not stale_jobs:
            logger.info("✅ No stale jobs found")
        else:
            logger.warning(f"🧟 Found {len(stale_jobs)} stale job(s), attempting recovery...")
            
            # Get ARQ pool for re-queuing
            arq_pool = await create_pool(get_arq_redis_settings())
            
            resumed = 0
            failed = 0
            
            for job_id, metadata in stale_jobs:
                # A job is resumable if it has:
                # 1. A full cached transcript or artifacts (existing logic), OR
                # 2. Audio stored in Redis (can re-transcribe from chunks)
                has_progress = (
                    metadata["has_transcript"] == "1" or 
                    metadata["has_artifacts"] == "1"
                )
                has_audio = metadata.get("has_audio") == "1"
                
                can_resume = has_progress or has_audio
                
                if can_resume:
                    # Re-queue the job to resume processing
                    try:
                        await arq_pool.enqueue_job(
                            "process_audio_job",
                            job_id,
                            metadata["file_path"],
                            metadata["file_name"],
                            metadata["file_size"],
                        )
                        resumed += 1
                        resume_reason = []
                        if has_progress:
                            resume_reason.append("cached progress")
                        if has_audio:
                            resume_reason.append("stored audio")
                        logger.info(
                            f"✅ Re-queued job {job_id} for resumption "
                            f"({', '.join(resume_reason)})"
                        )
                    except Exception as e:
                        logger.error(f"Failed to re-queue job {job_id}: {e}")
                        await job_store.mark_job_failed(
                            UUID(job_id),
                            f"Failed to resume after restart: {e}"
                        )
                        failed += 1
                else:
                    # No cached progress and no stored audio — unrecoverable
                    await job_store.mark_job_failed(
                        UUID(job_id),
                        "Job interrupted by server restart. Please re-upload your file."
                    )
                    failed += 1
                    logger.warning(f"❌ Marked job {job_id} as failed (no cached progress or audio)")
            
            await arq_pool.close()
            
            logger.info(
                f"🔄 Recovery complete: {resumed} resumed, {failed} failed"
            )
    except Exception as e:
        logger.error(f"Failed to recover zombie jobs: {e}")
    
    # Pre-warm the Whisper model into memory so the first job
    # doesn't waste time loading it (saves ~2-3s per cold start)
    settings = get_settings()
    if not settings.test_mode:
        try:
            logger.info(f"🔊 Pre-warming Whisper model: {settings.whisper_model}")
            from app.services.transcription import _get_cached_model
            import asyncio
            await asyncio.to_thread(_get_cached_model, settings.whisper_model)
            logger.info("🔊 Whisper model ready")
        except Exception as e:
            logger.warning(f"Failed to pre-warm Whisper model (will load on first job): {e}")
    
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
