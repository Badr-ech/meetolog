"""FastAPI application entry point for the Meetolog backend.

Exposes endpoints for audio upload (direct or presigned), job status
polling, artifact retrieval/editing, PDF download, and Jira export.
"""

import asyncio
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
from .core.logger import configure_logging, get_logger
from .models import JobResponse, ProcessingStatus, MeetingArtifacts, parse_processing_status
from .models.metadata import FileMetadata
from .models.db_models import JobRecord
from .infrastructure.db import get_async_session, init_db, close_db
from .infrastructure.postgres_job_store import PostgresJobStore
from .infrastructure.ecs_scaler import scale_worker_up
from .infrastructure.postgres_queue import PostgresJobQueue
from .services.jira_mapper import map_artifacts_to_jira
from .services.storage import S3StorageService

settings = get_settings()
configure_logging(json_output=not settings.debug, log_level="INFO")
logger = get_logger(component="api")

# Presigned-GET URL for the generated PDF is valid for one hour.
_PDF_DOWNLOAD_URL_TTL_SECONDS = 3600

_s3_service: S3StorageService | None = None


def get_s3_service() -> S3StorageService:
    """Return the process-wide S3StorageService, creating it on first call."""
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


JobStoreDep = Annotated[PostgresJobStore, Depends(get_job_store)]
JobQueueDep = Annotated[PostgresJobQueue, Depends(get_job_queue)]
DBSessionDep = Annotated[AsyncSession, Depends(get_async_session)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("api_starting")

    if settings.database_url:
        try:
            await init_db()
            logger.info("database_initialised")
        except Exception as exc:
            logger.error("database_init_failed", error=str(exc))
            logger.warning("api_starting_without_database")
    else:
        logger.warning("database_url_not_set")

    logger.info(
        "api_ready",
        s3_bucket=settings.aws_s3_bucket or "(not configured)",
        llm_provider=settings.llm_provider,
        test_mode=settings.test_mode,
    )

    yield

    logger.info("api_shutting_down")
    await close_db()
    logger.info("api_shutdown_complete")


app = FastAPI(
    title="Meetolog API",
    description="Transform meeting recordings into structured Agile artifacts",
    version="3.0.0",
    lifespan=lifespan,
)

allowed_origins = [
    origin.strip() for origin in settings.cors_origins.split(",") if origin.strip()
]
logger.info("cors_configured", allowed_origins=allowed_origins)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / response schemas
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
        logger.error("presign_generation_failed", error=str(exc))
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

    if settings.database_url:
        db.add(FileMetadata(
            job_id=str(job_id),
            s3_key=body.s3_key,
            original_filename=safe_file_name,
            file_size_bytes=body.file_size,
        ))
        await db.commit()

    record = await queue.enqueue_job(
        job_id=job_id,
        s3_key=body.s3_key,
        file_name=safe_file_name,
        file_size=body.file_size,
    )
    logger.info("job_enqueued", job_id=str(job_id), source="presigned_upload")

    # Wake the worker if it is scaled to 0. Fire-and-forget — a failure
    # here must never block the response; the job is already in the DB and
    # will be picked up once the worker is running.
    asyncio.create_task(scale_worker_up(
        cluster=settings.ecs_cluster,
        service=settings.ecs_worker_service,
        region=settings.aws_region,
    ))

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
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in settings.allowed_audio_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: {file_ext}. "
                f"Allowed: {settings.allowed_audio_extensions}"
            ),
        )

    # Determine the file size from the stream without buffering it in memory.
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    size_mb = file_size / (1024 * 1024)

    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f} MB. Max: {settings.max_upload_size_mb} MB",
        )

    job_id = uuid4()
    s3_key = f"uploads/{job_id}{file_ext}"

    s3 = get_s3_service()
    try:
        await s3.upload_stream(file.file, s3_key)
    except Exception as exc:
        logger.error("s3_upload_failed", job_id=str(job_id), error=str(exc))
        raise HTTPException(
            status_code=502,
            detail="Failed to upload file to storage. Please try again later.",
        )

    logger.info(
        "file_uploaded_to_s3",
        job_id=str(job_id),
        filename=file.filename,
        s3_key=s3_key,
        size_mb=round(size_mb, 2),
    )

    db.add(FileMetadata(
        job_id=str(job_id),
        s3_key=s3_key,
        original_filename=file.filename,
        file_size_bytes=file_size,
    ))
    await db.commit()

    record = await queue.enqueue_job(
        job_id=job_id,
        s3_key=s3_key,
        file_name=file.filename,
        file_size=file_size,
    )
    logger.info("job_enqueued", job_id=str(job_id), source="direct_upload")

    # Wake the worker if it is scaled to 0.
    asyncio.create_task(scale_worker_up(
        cluster=settings.ecs_cluster,
        service=settings.ecs_worker_service,
        region=settings.aws_region,
    ))

    return JobResponse(
        job_id=record.id,
        status=ProcessingStatus.UPLOADING,
        message=record.message,
        progress=record.progress,
    )


@app.get("/status/{job_id}", response_model=JobResponse)
async def get_job_status(job_id: UUID, job_store: JobStoreDep):
    """Return progress, status, and (when complete) artifacts for a job."""
    job = await job_store.load(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/download/{job_id}")
async def download_pdf(
    job_id: UUID,
    job_store: JobStoreDep,
    session: DBSessionDep,
):
    """Redirect the client to a short-lived presigned S3 URL for the PDF."""
    job = await job_store.load(job_id)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Current status: {job.status.value}",
        )

    result = await session.execute(
        select(JobRecord.pdf_s3_key).where(JobRecord.id == job_id)
    )
    pdf_s3_key = result.scalar_one_or_none()
    if not pdf_s3_key:
        raise HTTPException(status_code=404, detail="PDF file not found")

    presigned_url = await get_s3_service().generate_presigned_get_url(
        pdf_s3_key, expires_in=_PDF_DOWNLOAD_URL_TTL_SECONDS,
    )
    return RedirectResponse(url=presigned_url, status_code=307)


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
    """Return the extracted artifacts for a completed job."""
    job = await job_store.load(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != ProcessingStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not complete. Current status: {job.status.value}",
        )

    artifacts = job.artifacts or await job_store.get_cached_artifacts(job_id)
    if not artifacts:
        raise HTTPException(status_code=404, detail="Artifacts not found")
    return artifacts


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


@app.post("/jobs/{job_id}/cancel", response_model=JobResponse)
async def cancel_job(
    job_id: UUID,
    queue: JobQueueDep,
    job_store: JobStoreDep,
):
    """
    Cancel a queued or in-progress transcription job.

    The endpoint is safe to call from a browser ``beforeunload`` handler
    via ``navigator.sendBeacon`` (POST with an empty body) as well as
    from an explicit user action.  It is idempotent: calling it on a job
    that is already ``cancelled`` returns HTTP 200 without side effects.

    State machine
    -------------
    * ``pending`` / any active stage → ``cancelled`` (atomic UPDATE).
    * ``cancelled`` → 200 (already cancelled, no change).
    * ``completed`` / ``failed`` → 409 Conflict.
    * Not found → 404.

    The worker polls the database status at every chunk boundary.  When
    it reads ``cancelled`` it aborts processing, purges its ``/tmp``
    working directory, and exits the job cleanly without marking it as
    ``failed``.
    """
    try:
        previous_status = await queue.cancel_job(job_id)
    except ValueError as exc:
        # Job is in a terminal state (completed or failed).
        raise HTTPException(status_code=409, detail=str(exc))

    if previous_status is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Return the current job record so the caller can update its local state
    # immediately without waiting for the next polling cycle.
    job = await job_store.load(job_id)
    if job is None:
        # Extremely unlikely: the row vanished between cancel and load.
        raise HTTPException(status_code=404, detail="Job not found after cancellation")

    logger.info(
        "job_cancel_requested",
        job_id=str(job_id),
        previous_status=previous_status,
    )
    return job


@app.get("/health")
async def health_check(db: DBSessionDep):
    """Report database connectivity and the number of pending/processing jobs."""
    db_status = "healthy"
    pending = 0
    processing = 0
    try:
        row = (
            await db.execute(
                select(
                    func.count().filter(JobRecord.status == "pending").label("pending"),
                    func.count().filter(JobRecord.status == "processing").label("processing"),
                )
            )
        ).one()
        pending, processing = row.pending, row.processing
    except Exception as exc:
        db_status = f"unhealthy: {exc}"

    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
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
        reload=settings.debug,
        log_level="info",
    )
