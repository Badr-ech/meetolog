"""
Meetolog Backend — FastAPI application entry point.

Defines all HTTP endpoints for audio upload, job status polling,
artifact retrieval/editing, PDF download, and Jira export.
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from uuid import UUID, uuid4

import aiofiles
from arq import create_pool
from arq.connections import ArqRedis
from fastapi import FastAPI, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from fastapi.responses import JSONResponse

from .config import get_settings
from .models import JobResponse, ProcessingStatus, MeetingArtifacts
from .infrastructure.redis import (
    get_redis_pool,
    close_redis_pool,
    get_arq_redis_settings,
    check_redis_health,
)
from .infrastructure.job_store import RedisJobStore
from .services.jira_mapper import map_artifacts_to_jira

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# ARQ connection pool (singleton)
_arq_pool: ArqRedis | None = None


async def get_job_store() -> RedisJobStore:
    redis = await get_redis_pool()
    return RedisJobStore(redis)


async def get_arq_pool() -> ArqRedis:
    global _arq_pool
    
    if _arq_pool is None:
        _arq_pool = await create_pool(get_arq_redis_settings())
        logger.info("ARQ connection pool created")
    
    return _arq_pool


# Type alias for cleaner endpoint signatures
JobStoreDep = Annotated[RedisJobStore, Depends(get_job_store)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _arq_pool
    
    # Startup
    logger.info("Starting Meetolog API...")
    
    # Initialize Redis
    try:
        redis = await get_redis_pool()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        logger.warning("API starting without Redis - endpoints may fail")
    
    # Initialize ARQ pool
    try:
        _arq_pool = await create_pool(get_arq_redis_settings())
        logger.info("ARQ connection pool created")
    except Exception as e:
        logger.error(f"Failed to create ARQ pool: {e}")
        logger.warning("API starting without ARQ - job enqueueing will fail")
    
    # Ensure directories exist
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Upload directory: {settings.upload_dir}")
    logger.info(f"Output directory: {settings.output_dir}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Test Mode: {settings.test_mode}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Meetolog API...")
    
    if _arq_pool is not None:
        await _arq_pool.close()
        _arq_pool = None
    
    await close_redis_pool()
    
    logger.info("Shutdown complete")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Meetolog API",
    description="Transform meeting recordings into structured Agile artifacts",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend access
allowed_origins = [origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()]
logger.info(f"CORS allowed origins: {allowed_origins}")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_upload_dir() -> Path:
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


@app.get("/")
async def root():
    redis_health = await check_redis_health()
    
    return {
        "service": "Meetolog API",
        "version": "2.0.0",
        "status": "healthy" if redis_health["status"] == "healthy" else "degraded",
        "test_mode": settings.test_mode,
        "llm_provider": settings.llm_provider,
        "redis": redis_health,
    }


@app.post("/upload", response_model=JobResponse)
async def upload_audio(
    file: UploadFile,
    job_store: JobStoreDep,
):
    """
    Upload an audio file for processing.
    
    Accepts audio files (mp3, wav, m4a, ogg, webm) and enqueues
    a background job for processing.
    
    Returns a job ID to track processing status.
    
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_audio_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. "
                   f"Allowed: {settings.allowed_audio_extensions}",
        )
    
    # Validate file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    size_mb = file_size / (1024 * 1024)
    file.file.seek(0)  # Reset
    
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB. Max: {settings.max_upload_size_mb}MB",
        )
    
    # Generate job ID and save file
    job_id = uuid4()
    upload_dir = get_upload_dir()
    audio_path = upload_dir / f"{job_id}{file_ext}"
    
    # Save file asynchronously
    content = await file.read()
    async with aiofiles.open(audio_path, "wb") as f:
        await f.write(content)
    
    logger.info(f"File uploaded: {file.filename} -> {audio_path} ({size_mb:.2f} MB)")
    
    # Create job record in Redis with UPLOADING status
    job = JobResponse(
        job_id=job_id,
        status=ProcessingStatus.UPLOADING,
        message="Job queued for processing",
        progress=0,
    )
    
    await job_store.save(
        job_id,
        job,
        file_path=str(audio_path),
        file_name=file.filename,
        file_size=file_size,
    )
    
    # Enqueue job to ARQ worker
    try:
        arq_pool = await get_arq_pool()
        
        arq_job = await arq_pool.enqueue_job(
            "process_audio_job",
            job_id=str(job_id),
            file_path=str(audio_path),
            file_name=file.filename,
            file_size=file_size,
            _job_id=str(job_id),  # Use our job_id as ARQ job_id
        )
        
        logger.info(f"Job {job_id} enqueued to ARQ (arq_job_id: {arq_job.job_id})")
        
    except Exception as e:
        logger.error(f"Failed to enqueue job {job_id}: {e}")
        
        # Update job status to failed
        await job_store.update(
            job_id,
            status=ProcessingStatus.FAILED,
            error=f"Failed to enqueue job: {e}",
            message="Job enqueueing failed",
        )
        
        # Clean up uploaded file
        if audio_path.exists():
            audio_path.unlink()
        
        raise HTTPException(
            status_code=503,
            detail="Job queue unavailable. Please try again later.",
        )
    
    return job


@app.get("/status/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: UUID, job_store: JobStoreDep):
    """
    Get the current status of a processing job.
    
    Returns progress, status, and artifacts when complete.
    
    """
    job = await job_store.load(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job


@app.get("/download/{job_id}")
async def download_pdf(job_id: UUID, job_store: JobStoreDep):
    """
    Download the generated PDF summary for a completed job.
    
    """
    job = await job_store.load(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Current status: {job.status.value}",
        )
    
    pdf_path = Path(settings.output_dir) / f"meeting_{job_id}.pdf"
    
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF file not found")
    
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"meeting_summary_{job_id}.pdf",
    )


@app.put("/artifacts/{job_id}", response_model=JobResponse)
async def update_artifacts(
    job_id: UUID,
    artifacts: MeetingArtifacts,
    job_store: JobStoreDep,
):
    """
    Replace all artifacts for a completed job.

    Accepts the full MeetingArtifacts payload (PUT semantics) so that
    Pydantic validates the entire schema on every save.  Partial updates
    are intentionally unsupported to prevent schema drift.

    - 404 if the job does not exist.
    - 400 if the job is not in COMPLETED state.
    - 422 (automatic) if the body violates the MeetingArtifacts schema.
    """
    job = await job_store.load(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot edit artifacts: job status is '{job.status.value}'. "
                   "Only completed jobs may be edited.",
        )

    updated_job = await job_store.update_artifacts(job_id, artifacts)
    return updated_job


@app.get("/artifacts/{job_id}", response_model=MeetingArtifacts)
async def get_artifacts(job_id: UUID, job_store: JobStoreDep):
    """
    Get the extracted artifacts as JSON for a completed job.
    
    """
    job = await job_store.load(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Current status: {job.status.value}",
        )
    
    if not job.artifacts:
        # Try loading from cache directly
        artifacts = await job_store.get_cached_artifacts(job_id)
        if not artifacts:
            raise HTTPException(status_code=404, detail="Artifacts not found")
        return artifacts
    
    return job.artifacts


@app.get("/export/jira/{job_id}")
async def export_jira(job_id: UUID, job_store: JobStoreDep):
    """
    Export artifacts as a Jira-compatible bulk-import JSON file.

    Returns a downloadable JSON file matching the Jira bulk import format:
    ``{ "projects": [{ "key": "MEET", "issues": [...] }] }``

    - 404 if the job does not exist or has no artifacts.
    - 400 if the job is not in COMPLETED state.
    """
    job = await job_store.load(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Current status: {job.status.value}",
        )

    artifacts = job.artifacts
    if not artifacts:
        artifacts = await job_store.get_cached_artifacts(job_id)
    if not artifacts:
        raise HTTPException(status_code=404, detail="Artifacts not found")

    payload = map_artifacts_to_jira(artifacts)

    return JSONResponse(
        content=payload.model_dump(),
        headers={
            "Content-Disposition": f'attachment; filename="meetolog_jira_export_{job_id}.json"',
        },
    )


@app.get("/health")
async def health_check():
    """
    Detailed health check for monitoring systems.
    
    """
    redis_health = await check_redis_health()
    
    # Check if ARQ pool is connected
    arq_status = "unknown"
    try:
        arq_pool = await get_arq_pool()
        # Check queue length (ARQ default queue key)
        redis = await get_redis_pool()
        queue_len = await redis.llen("arq:queue")
        arq_status = "healthy"
    except Exception as e:
        arq_status = f"unhealthy: {e}"
        queue_len = -1
    
    overall_status = "healthy"
    if redis_health["status"] != "healthy" or arq_status != "healthy":
        overall_status = "degraded"
    
    return {
        "status": overall_status,
        "components": {
            "redis": redis_health,
            "arq": {
                "status": arq_status,
                "queue_length": queue_len,
            },
        },
        "config": {
            "test_mode": settings.test_mode,
            "llm_provider": settings.llm_provider,
            "redis_url": settings.redis_url.split("@")[-1] if "@" in settings.redis_url else settings.redis_url,  # Hide password
        },
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # Use 1 for development, increase for production
        reload=settings.debug,
        log_level="info",
    )
