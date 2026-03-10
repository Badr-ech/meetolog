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

from fastapi import FastAPI, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from .config import get_settings
from .models import JobResponse, ProcessingStatus, MeetingArtifacts
from .models.metadata import FileMetadata
from .models.db_models import JobRecord
from .infrastructure.db import get_async_session, init_db, close_db
from .infrastructure.postgres_job_store import PostgresJobStore
from .infrastructure.postgres_queue import PostgresJobQueue
from .services.jira_mapper import map_artifacts_to_jira
from .services.storage import S3StorageService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Initialize settings
settings = get_settings()

# S3 storage service (singleton)
_s3_service: S3StorageService | None = None


def get_s3_service() -> S3StorageService:
    """Return the global S3StorageService, creating it on first call."""
    global _s3_service
    if _s3_service is None:
        _s3_service = S3StorageService()
    return _s3_service


async def get_job_store(
    session: AsyncSession = Depends(get_async_session),
) -> PostgresJobStore:
    return PostgresJobStore(session)


async def get_job_queue(
    session: AsyncSession = Depends(get_async_session),
) -> PostgresJobQueue:
    return PostgresJobQueue(session)


# Type aliases for cleaner endpoint signatures
JobStoreDep = Annotated[PostgresJobStore, Depends(get_job_store)]
JobQueueDep = Annotated[PostgresJobQueue, Depends(get_job_queue)]
DBSessionDep = Annotated[AsyncSession, Depends(get_async_session)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Meetolog API...")

    # Initialize PostgreSQL and create tables
    if settings.database_url:
        try:
            await init_db()
            logger.info("PostgreSQL database initialised")
        except Exception as e:
            logger.error(f"Failed to initialise database: {e}")
            logger.warning("API starting without PostgreSQL")
    else:
        logger.warning("DATABASE_URL not set — job persistence unavailable")

    logger.info(f"S3 bucket: {settings.aws_s3_bucket or '(not configured)'}")
    logger.info(f"LLM Provider: {settings.llm_provider}")
    logger.info(f"Test Mode: {settings.test_mode}")

    yield

    # Shutdown
    logger.info("Shutting down Meetolog API...")
    await close_db()
    logger.info("Shutdown complete")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Meetolog API",
    description="Transform meeting recordings into structured Agile artifacts",
    version="3.0.0",
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


# ---------------------------------------------------------------------------
# Request / response schemas for the presigned upload flow
# ---------------------------------------------------------------------------


class PresignRequest(BaseModel):
    filename: str
    file_type: str
    file_size: int = Field(..., gt=0, description="Declared file size in bytes")


class PresignResponse(BaseModel):
    url: str
    fields: dict[str, str]
    s3_key: str


class EnqueueRequest(BaseModel):
    s3_key: str = Field(..., description="S3 object key returned by /upload/presign")
    file_name: str = Field(..., description="Original filename for metadata storage")
    file_size: int = Field(..., gt=0, description="File size in bytes")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/")
async def root():
    return {
        "service": "Meetolog API",
        "version": "3.0.0",
        "status": "healthy",
        "test_mode": settings.test_mode,
        "llm_provider": settings.llm_provider,
    }


@app.post("/upload/presign", response_model=PresignResponse)
async def presign_upload(body: PresignRequest):
    """
    Generate a presigned S3 POST URL for direct browser-to-S3 audio upload.

    The client should:
    1. POST file metadata here to receive ``url``, ``fields``, and ``s3_key``.
    2. Build a ``multipart/form-data`` body with the returned fields appended
       first, then the ``file`` field last, and POST it to ``url``.
    3. On HTTP 200/204 from S3, call ``POST /jobs/enqueue`` with ``s3_key`` to
       trigger the transcription pipeline.

    - 400 if the MIME type is not an accepted audio format.
    - 400 if the declared file size is outside the permitted range.
    - 503 if the S3 service is unreachable or credentials are invalid.
    """
    if not settings.aws_s3_bucket:
        raise HTTPException(
            status_code=503,
            detail="S3 storage is not configured on this server.",
        )

    s3 = get_s3_service()
    try:
        result = await s3.generate_upload_presigned_post(
            filename=body.filename,
            file_type=body.file_type,
            file_size=body.file_size,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error("Presign generation failed: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Failed to generate upload URL. Please try again later.",
        )

    return PresignResponse(**result)


@app.post("/jobs/enqueue", response_model=JobResponse)
async def enqueue_job(
    body: EnqueueRequest,
    queue: JobQueueDep,
    db: DBSessionDep,
):
    """
    Enqueue a transcription job for an audio file already uploaded to S3.

    Inserts a row into the ``job_records`` table with ``pending`` status.
    A background worker polling the table will pick it up.

    - 400 if ``s3_key`` does not begin with the expected ``uploads/`` prefix.
    """
    if not body.s3_key.startswith("uploads/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid s3_key: must reference an object in the uploads/ prefix.",
        )

    safe_file_name = Path(body.file_name).name
    job_id = uuid4()

    # Persist file metadata in PostgreSQL
    if settings.database_url:
        metadata_record = FileMetadata(
            job_id=str(job_id),
            s3_key=body.s3_key,
            original_filename=safe_file_name,
            file_size_bytes=body.file_size,
        )
        db.add(metadata_record)
        await db.commit()

    # Insert job into the persistent Postgres queue
    record = await queue.enqueue_job(
        job_id=job_id,
        s3_key=body.s3_key,
        file_name=safe_file_name,
        file_size=body.file_size,
    )
    logger.info("Job %s enqueued to Postgres queue", job_id)

    return JobResponse(
        job_id=record.id,
        status=ProcessingStatus.UPLOADING,
        message=record.message,
        progress=record.progress,
    )


@app.post("/upload", response_model=JobResponse)
async def upload_audio(
    file: UploadFile,
    queue: JobQueueDep,
    db: DBSessionDep,
):
    """
    Upload an audio file for processing.

    Streams the file directly to S3 (no local storage), persists
    metadata in PostgreSQL, and inserts a job into the Postgres queue.
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

    # Generate job ID and S3 key
    job_id = uuid4()
    s3_key = f"uploads/{job_id}{file_ext}"

    # Stream upload to S3
    s3 = get_s3_service()
    try:
        await s3.upload_stream(file.file, s3_key)
    except Exception as e:
        logger.error(f"S3 upload failed for job {job_id}: {e}")
        raise HTTPException(
            status_code=502,
            detail="Failed to upload file to storage. Please try again later.",
        )

    logger.info(f"File uploaded to S3: {file.filename} -> {s3_key} ({size_mb:.2f} MB)")

    # Persist file metadata in PostgreSQL
    metadata_record = FileMetadata(
        job_id=str(job_id),
        s3_key=s3_key,
        original_filename=file.filename,
        file_size_bytes=file_size,
    )
    db.add(metadata_record)
    await db.commit()

    # Insert job into the persistent Postgres queue
    record = await queue.enqueue_job(
        job_id=job_id,
        s3_key=s3_key,
        file_name=file.filename,
        file_size=file_size,
    )
    logger.info("Job %s enqueued to Postgres queue", job_id)

    return JobResponse(
        job_id=record.id,
        status=ProcessingStatus.UPLOADING,
        message=record.message,
        progress=record.progress,
    )


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
async def download_pdf(
    job_id: UUID,
    job_store: JobStoreDep,
    session: Annotated[AsyncSession, Depends(get_async_session)],
):
    """
    Download the generated PDF summary for a completed job.

    Redirects the client to a short-lived presigned S3 URL for the PDF.
    """
    job = await job_store.load(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Current status: {job.status.value}",
        )

    # Prefer S3 when the worker has stored a pdf_s3_key
    result = await session.execute(
        select(JobRecord.pdf_s3_key).where(JobRecord.id == job_id)
    )
    pdf_s3_key = result.scalar_one_or_none()

    if pdf_s3_key:
        s3 = get_s3_service()
        presigned_url = await s3.generate_presigned_get_url(pdf_s3_key, expires_in=3600)
        return RedirectResponse(url=presigned_url, status_code=307)

    # No S3 key means the PDF was never uploaded
    raise HTTPException(status_code=404, detail="PDF file not found")


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
async def health_check(
    db: DBSessionDep,
):
    """
    Detailed health check for monitoring systems.

    Reports database connectivity and the number of pending/processing jobs.
    """
    db_status = "healthy"
    pending = 0
    processing = 0
    try:
        row = (
            await db.execute(
                select(
                    func.count()
                    .filter(JobRecord.status == "pending")
                    .label("pending"),
                    func.count()
                    .filter(JobRecord.status == "processing")
                    .label("processing"),
                )
            )
        ).one()
        pending, processing = row.pending, row.processing
    except Exception as exc:
        db_status = f"unhealthy: {exc}"

    overall = "healthy" if db_status == "healthy" else "degraded"

    return {
        "status": overall,
        "components": {
            "database": {
                "status": db_status,
                "pending_jobs": pending,
                "processing_jobs": processing,
            },
        },
        "config": {
            "test_mode": settings.test_mode,
            "llm_provider": settings.llm_provider,
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
