"""
Meetolog Backend - FastAPI Application
Main entry point with API endpoints for audio processing.

IMPORTANT - Deployment Constraints:
- This application MUST run as a single instance (workers=1)
- Uses local file storage for job state (no horizontal scaling)
- Requires ffmpeg installed on the host system
"""

import logging
import shutil
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import get_settings
from .dependencies import (
    get_job_store,
    get_transcriber,
    get_extractor,
    initialize_services,
    JobStoreDep,
    TranscriberDep,
    LLMExtractorDep,
)
from .models import JobResponse, ProcessingStatus, MeetingArtifacts


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# PDF service singleton (lazy initialized)
_pdf_service = None


def get_pdf_service():
    """Get or create the PDF generator service."""
    global _pdf_service
    if _pdf_service is None:
        from .services import PDFGeneratorService
        settings = get_settings()
        _pdf_service = PDFGeneratorService(Path(settings.output_dir))
    return _pdf_service


# =============================================================================
# Application Lifespan
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan event handler.
    
    Startup: Initialize services, load persisted job state
    Shutdown: Clean up resources
    """
    # Startup
    logger.info("Starting Meetolog API...")
    await initialize_services()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Meetolog API...")


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Meetolog API",
    description="Transform meeting recordings into structured Agile artifacts",
    version="1.0.0",
    lifespan=lifespan,
)

# Configure CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize settings
settings = get_settings()


def get_upload_dir() -> Path:
    """Get and ensure upload directory exists."""
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    return upload_dir


async def process_audio(
    job_id: UUID,
    audio_path: Path,
    job_store: JobStoreDep,
    transcriber: TranscriberDep,
    extractor: LLMExtractorDep,
):
    """
    Background task to process an audio file through the full pipeline.
    
    Pipeline:
    1. Transcribe audio to text
    2. Preprocess the transcript
    3. Extract Agile artifacts using LLM
    4. Generate PDF summary
    
    Args:
        job_id: Unique identifier for the job
        audio_path: Path to the uploaded audio file
        job_store: Job storage instance (injected)
        transcriber: Transcription service (injected)
        extractor: LLM extraction service (injected)
    """
    try:
        # Step 1: Transcription
        await job_store.update(
            job_id,
            status=ProcessingStatus.TRANSCRIBING,
            progress=20,
            message="Transcribing audio..."
        )
        
        raw_transcript = await transcriber.transcribe(audio_path)
        transcript = await transcriber.preprocess_transcript(raw_transcript)
        
        # Step 2: LLM Extraction
        await job_store.update(
            job_id,
            status=ProcessingStatus.EXTRACTING,
            progress=50,
            message="Extracting Agile artifacts..."
        )
        
        artifacts = await extractor.extract_artifacts(transcript)
        
        # Step 3: PDF Generation
        await job_store.update(
            job_id,
            status=ProcessingStatus.GENERATING_PDF,
            progress=80,
            message="Generating PDF summary..."
        )
        
        pdf_service = get_pdf_service()
        pdf_filename = f"meeting_{job_id}.pdf"
        pdf_path = await pdf_service.generate(artifacts, pdf_filename)
        
        # Complete
        await job_store.update(
            job_id,
            status=ProcessingStatus.COMPLETED,
            progress=100,
            message="Processing complete!",
            artifacts=artifacts,
            pdf_url=f"/download/{job_id}"
        )
        
        logger.info(f"Job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        await job_store.update(
            job_id,
            status=ProcessingStatus.FAILED,
            error=str(e),
            message=f"Processing failed: {e}"
        )
    
    finally:
        # Cleanup uploaded file
        if audio_path.exists():
            audio_path.unlink()
            logger.debug(f"Cleaned up uploaded file: {audio_path}")


@app.get("/")
async def root():
    """Health check endpoint with service status."""
    settings = get_settings()
    extractor = get_extractor()
    
    return {
        "service": "Meetolog API",
        "status": "healthy",
        "version": "1.0.0",
        "test_mode": settings.test_mode,
        "using_mock_llm": extractor.is_mock,
    }


@app.post("/upload", response_model=JobResponse)
async def upload_audio(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    job_store: JobStoreDep,
    transcriber: TranscriberDep,
    extractor: LLMExtractorDep,
):
    """
    Upload an audio file for processing.
    
    Accepts audio files (mp3, wav, m4a, ogg, webm) and triggers
    background processing to extract meeting artifacts.
    
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
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)  # Reset
    
    if size_mb > settings.max_upload_size_mb:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB. Max: {settings.max_upload_size_mb}MB",
        )
    
    # Save uploaded file
    job_id = uuid4()
    upload_dir = get_upload_dir()
    audio_path = upload_dir / f"{job_id}{file_ext}"
    
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Create job record
    job = JobResponse(
        job_id=job_id,
        status=ProcessingStatus.PENDING,
        message="File uploaded, processing starting...",
        progress=0,
    )
    await job_store.save(job_id, job)
    
    # Trigger background processing
    background_tasks.add_task(
        process_audio,
        job_id,
        audio_path,
        job_store,
        transcriber,
        extractor,
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
        raise HTTPException(status_code=404, detail="Artifacts not found")
    
    return job.artifacts


# =============================================================================
# Entry Point for Direct Execution
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # CRITICAL: Run with workers=1 to avoid race conditions
    # This application uses local file/memory storage for job state.
    # Multiple workers would cause data loss and inconsistent state.
    # For horizontal scaling, implement RedisJobStore (Version 2).
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        workers=1,  # MUST be 1 for local storage
        reload=settings.debug,
        log_level="info",
    )
