"""
Meetolog background worker — three operating modes, one image.

The ``SERVICE_TYPE`` environment variable selects which role this process
plays when the container starts:

``worker`` / ``splitter`` (the ECS service, desired-count 0 → 1)
    Polls ``job_records`` for pending jobs.  For each job it downloads the
    audio, splits it into 5-minute chunks, uploads each chunk to S3, creates
    ``job_chunks`` rows, detects the recording language, marks the job
    ``transcribing``, and fires off up to ``max_parallel_chunks`` ephemeral
    chunk-worker tasks via ``ecs:RunTask``.  When the queue drains it calls
    ``scale_worker_down`` and exits (idle-shutdown behaviour preserved).

``chunk_worker`` (ephemeral RunTask, 0.5 vCPU / 2 GB)
    Reads ``JOB_ID`` from the environment, then loops: claim the next
    pending chunk for that job → download from S3 → transcribe with Whisper
    → mark complete.  Repeats until no chunks remain.  The last worker to
    exhaust the queue calls ``try_transition_to_assembling``; only the one
    whose atomic UPDATE succeeds launches the assembler RunTask.

``assembler`` (ephemeral RunTask, 1 vCPU / 4 GB)
    Reads ``JOB_ID`` from the environment, reassembles the full transcript
    from the ordered chunk records, then runs the LLM extraction → PDF →
    S3 upload pipeline — identical to the old monolithic worker's final
    stages.

Cancellation is checked at every major boundary in all three modes.  When
a job is cancelled mid-flight the workers log, clean up their temp files,
and exit without marking the job ``failed``.
"""

import asyncio
import gc
import os
import signal
import sys
import tempfile
import time
import traceback
from pathlib import Path
from uuid import UUID

import structlog

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.core.logger import configure_logging, get_logger
from app.models import MeetingArtifacts, ProcessingStatus
from app.infrastructure.db import close_db, get_session_factory, init_db
from app.infrastructure.chunk_queue import ChunkQueue
from app.infrastructure.ecs_scaler import (
    run_assembler,
    run_chunk_workers,
    scale_worker_down,
)
from app.infrastructure.postgres_queue import PostgresJobQueue
from app.services.storage import S3StorageService

configure_logging(json_output=True, log_level="INFO")

logger = get_logger(component="worker")

POLL_INTERVAL_SECONDS = 5


# ---------------------------------------------------------------------------
# Cancellation sentinel
# ---------------------------------------------------------------------------

class _JobCancelledError(Exception):
    """Raised at any checkpoint when the DB signals cancellation."""


async def _check_cancellation(
    job_id: UUID,
    queue: PostgresJobQueue,
    job_logger,
) -> None:
    status = await queue.fetch_job_status(job_id)
    if status == "cancelled":
        job_logger.info("cancellation_detected", job_id=str(job_id))
        raise _JobCancelledError(f"Job {job_id} cancelled")


# ---------------------------------------------------------------------------
# Shared service helpers
# ---------------------------------------------------------------------------

def _get_transcriber():
    settings = get_settings()
    if settings.test_mode:
        from app.services.mock_services import MockTranscriber
        return MockTranscriber(simulated_delay=0.5)
    try:
        from app.services.transcription import WhisperTranscriber
        return WhisperTranscriber(model_name=settings.whisper_model)
    except Exception as exc:
        logger.error("Failed to initialize WhisperTranscriber: %s", exc)
        from app.services.mock_services import MockTranscriber
        return MockTranscriber(simulated_delay=0.5)


def _get_llm_provider():
    from app.services.llm_engine import get_llm_provider as _get_provider
    from app.services.llm_extraction import HierarchicalExtractor
    return HierarchicalExtractor(_get_provider())


def _get_pdf_service(output_dir: Path):
    from app.services.pdf_generator import PDFGeneratorService
    return PDFGeneratorService(output_dir)


# ---------------------------------------------------------------------------
# Splitter — splits audio and fires chunk workers
# ---------------------------------------------------------------------------

async def _run_splitter_job(
    job_id: UUID,
    s3_key: str,
    file_name: str,
    queue: PostgresJobQueue,
    s3_service: S3StorageService,
    worker_id: str,
) -> None:
    """Download audio, split into chunks, upload chunks, launch workers."""
    settings = get_settings()
    job_logger = get_logger(component="splitter", job_id=str(job_id), worker_id=worker_id)
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(job_id=str(job_id), worker_id=worker_id)

    start_time = time.monotonic()
    job_logger.info("split_job_started", s3_key=s3_key)

    try:
        with tempfile.TemporaryDirectory(prefix=f"meetolog_split_{job_id}_") as work_dir:
            work_path = Path(work_dir)
            file_ext = Path(file_name).suffix.lower() or ".wav"
            audio_path = work_path / f"{job_id}{file_ext}"

            await queue.update_job_progress(
                job_id,
                status=ProcessingStatus.SPLITTING.value,
                progress=5,
                message="Splitting audio into chunks…",
            )

            job_logger.info("s3_download_start")
            await s3_service.download_to_file(s3_key, str(audio_path))

            await _check_cancellation(job_id, queue, job_logger)

            # Split into fixed-duration chunk WAV files.
            from app.utils.audio import split_audio_into_chunks
            chunk_dir = work_path / "chunks"
            chunks = await asyncio.to_thread(
                split_audio_into_chunks,
                audio_path,
                chunk_dir,
                300,  # 5-minute chunks
            )
            total_chunks = len(chunks)
            job_logger.info("audio_split", total_chunks=total_chunks)

            await _check_cancellation(job_id, queue, job_logger)

            # Detect language from the first chunk so all workers share it.
            detected_language: str | None = None
            if chunks and not settings.test_mode:
                try:
                    from app.services.transcription import WhisperTranscriber
                    transcriber = WhisperTranscriber(model_name=settings.whisper_model)
                    _, detected_language = await transcriber.transcribe_single_chunk(
                        chunks[0], language=None
                    )
                    # We only needed the language; discard the transcript here —
                    # the chunk worker will re-transcribe chunk 0 from S3.
                    del transcriber
                    gc.collect()
                    job_logger.info("language_detected", language=detected_language)
                except Exception as exc:
                    job_logger.warning(
                        "language_detection_failed",
                        error=str(exc),
                    )

            # Upload each chunk to S3 and collect the keys.
            chunk_s3_keys: list[str] = []
            for i, chunk_path in enumerate(chunks):
                chunk_key = f"chunks/{job_id}/{i:03d}.wav"
                await s3_service.upload_file(str(chunk_path), chunk_key)
                chunk_s3_keys.append(chunk_key)
                job_logger.info(
                    "chunk_uploaded",
                    chunk_index=i,
                    total=total_chunks,
                    s3_key=chunk_key,
                )

            await _check_cancellation(job_id, queue, job_logger)

        # Temp dir is gone; chunk WAVs now live only in S3.

        # Create DB records for each chunk.
        session_factory = get_session_factory()
        async with session_factory() as session:
            chunk_queue = ChunkQueue(session)
            await chunk_queue.create_chunks(job_id, chunk_s3_keys)

        # Persist the detected language on the job record.
        if detected_language:
            from sqlalchemy import update
            from app.models.db_models import JobRecord
            async with session_factory() as session:
                await session.execute(
                    update(JobRecord)
                    .where(JobRecord.id == job_id)
                    .values(detected_language=detected_language)
                )
                await session.commit()

        # Transition job to transcribing.
        await queue.update_job_progress(
            job_id,
            status=ProcessingStatus.TRANSCRIBING.value,
            progress=10,
            message=f"Transcribing {total_chunks} chunk(s) in parallel…",
        )

        # Launch parallel chunk workers (up to max_parallel_chunks).
        num_workers = min(total_chunks, settings.max_parallel_chunks)
        await run_chunk_workers(
            cluster=settings.ecs_cluster,
            service=settings.ecs_worker_service,
            task_definition=settings.ecs_worker_task_definition,
            job_id=job_id,
            num_workers=num_workers,
            detected_language=detected_language,
            region=settings.aws_region,
        )

        duration = round(time.monotonic() - start_time, 2)
        job_logger.info(
            "split_job_complete",
            total_chunks=total_chunks,
            num_workers=num_workers,
            duration_seconds=duration,
        )

    except _JobCancelledError:
        job_logger.info("split_job_cancelled", job_id=str(job_id))

    except Exception:
        job_logger.exception("split_job_error")
        try:
            await queue.mark_job_failed(job_id, traceback.format_exc()[-1000:])
        except Exception:
            job_logger.exception("mark_failed_error")

    finally:
        structlog.contextvars.clear_contextvars()


# ---------------------------------------------------------------------------
# Splitter loop (the persistent ECS service)
# ---------------------------------------------------------------------------

_shutdown_event = asyncio.Event()


def _request_shutdown() -> None:
    logger.info("shutdown_requested")
    _shutdown_event.set()


async def splitter_loop(worker_id: str | None = None) -> None:
    """Poll for pending jobs, split audio, and launch chunk workers.

    Preserves the idle auto-scale-down behaviour: after
    ``settings.worker_idle_shutdown_polls`` consecutive empty polls the
    service scales itself to 0 and exits.
    """
    if worker_id is None:
        worker_id = f"splitter-{os.getpid()}"

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _request_shutdown())

    settings = get_settings()
    await init_db()
    session_factory = get_session_factory()
    s3_service = S3StorageService()

    logger.info(
        "splitter_started",
        worker_id=worker_id,
        poll_interval=POLL_INTERVAL_SECONDS,
        idle_shutdown_polls=settings.worker_idle_shutdown_polls,
        max_parallel_chunks=settings.max_parallel_chunks,
    )

    idle_polls = 0

    try:
        while not _shutdown_event.is_set():
            async with session_factory() as session:
                queue = PostgresJobQueue(session)
                record = await queue.claim_next_job(worker_id)

            if record is None:
                idle_polls += 1
                logger.debug(
                    "queue_empty",
                    idle_polls=idle_polls,
                    shutdown_after=settings.worker_idle_shutdown_polls,
                )

                if idle_polls >= settings.worker_idle_shutdown_polls:
                    logger.info(
                        "worker_idle_scaling_down",
                        idle_polls=idle_polls,
                        worker_id=worker_id,
                    )
                    await scale_worker_down(
                        cluster=settings.ecs_cluster,
                        service=settings.ecs_worker_service,
                        region=settings.aws_region,
                    )
                    _shutdown_event.set()
                    break

                try:
                    await asyncio.wait_for(
                        _shutdown_event.wait(),
                        timeout=POLL_INTERVAL_SECONDS,
                    )
                except asyncio.TimeoutError:
                    pass
                continue

            idle_polls = 0
            logger.info("job_claimed", job_id=str(record.id), worker_id=worker_id)

            async with session_factory() as job_session:
                job_queue = PostgresJobQueue(job_session)
                await _run_splitter_job(
                    job_id=record.id,
                    s3_key=record.s3_key,
                    file_name=record.file_name,
                    queue=job_queue,
                    s3_service=s3_service,
                    worker_id=worker_id,
                )
    finally:
        await close_db()
        logger.info("splitter_stopped", worker_id=worker_id)


# ---------------------------------------------------------------------------
# Chunk worker — transcribes one job's chunks in a loop
# ---------------------------------------------------------------------------

async def chunk_worker_main() -> None:
    """Claim and transcribe chunks for the job given by JOB_ID env var.

    Loops until no chunks remain for the job, then checks whether all
    chunks are done and tries to transition the job to ``assembling``.
    Only the worker whose atomic CAS succeeds launches the assembler.
    """
    job_id_str = os.environ.get("JOB_ID", "")
    if not job_id_str:
        logger.error("chunk_worker_missing_job_id")
        sys.exit(1)

    job_id = UUID(job_id_str)
    # Language hint from the splitter (may be absent for older deployments).
    detected_language: str | None = os.environ.get("DETECTED_LANGUAGE") or None

    worker_id = f"chunk-{os.getpid()}"
    settings = get_settings()

    job_logger = get_logger(
        component="chunk_worker",
        job_id=job_id_str,
        worker_id=worker_id,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(job_id=job_id_str, worker_id=worker_id)

    await init_db()
    session_factory = get_session_factory()
    s3_service = S3StorageService()
    transcriber = _get_transcriber()

    job_logger.info(
        "chunk_worker_started",
        detected_language=detected_language,
    )

    chunks_processed = 0

    try:
        while True:
            # Check for cancellation before claiming the next chunk.
            async with session_factory() as session:
                queue = PostgresJobQueue(session)
                status = await queue.fetch_job_status(job_id)

            if status == "cancelled":
                job_logger.info("chunk_worker_job_cancelled")
                break

            if status not in ("transcribing",):
                # Job moved to a terminal or unexpected state.
                job_logger.info("chunk_worker_job_not_transcribing", status=status)
                break

            # Claim the next chunk.
            async with session_factory() as session:
                chunk_q = ChunkQueue(session)
                chunk = await chunk_q.claim_next_chunk(job_id, worker_id)

            if chunk is None:
                job_logger.info(
                    "chunk_worker_no_more_chunks",
                    chunks_processed=chunks_processed,
                )
                break

            chunk_id = chunk.id
            chunk_index = chunk.chunk_index
            chunk_s3_key = chunk.audio_s3_key

            job_logger.info(
                "chunk_claimed",
                chunk_index=chunk_index,
                s3_key=chunk_s3_key,
            )

            start = time.monotonic()

            try:
                with tempfile.TemporaryDirectory(
                    prefix=f"meetolog_chunk_{job_id}_{chunk_index}_"
                ) as work_dir:
                    chunk_path = Path(work_dir) / f"chunk_{chunk_index:03d}.wav"
                    await s3_service.download_to_file(chunk_s3_key, str(chunk_path))

                    # Use the language hint when available; fall back to auto.
                    # If this is a mock (test mode), transcribe_single_chunk
                    # is not defined — fall back to transcribe_chunk.
                    if hasattr(transcriber, "transcribe_single_chunk"):
                        text, lang = await transcriber.transcribe_single_chunk(
                            chunk_path, language=detected_language
                        )
                    else:
                        text = await transcriber.transcribe_chunk(chunk_path)
                        lang = detected_language or "unknown"

                async with session_factory() as session:
                    chunk_q = ChunkQueue(session)
                    await chunk_q.mark_chunk_completed(chunk_id, text, lang)

                elapsed = round(time.monotonic() - start, 2)
                job_logger.info(
                    "chunk_transcribed",
                    chunk_index=chunk_index,
                    chars=len(text),
                    language=lang,
                    duration_seconds=elapsed,
                )
                chunks_processed += 1
                del text
                gc.collect()

            except Exception as exc:
                job_logger.error(
                    "chunk_transcription_failed",
                    chunk_index=chunk_index,
                    error=str(exc),
                )
                async with session_factory() as session:
                    chunk_q = ChunkQueue(session)
                    await chunk_q.mark_chunk_failed(chunk_id, str(exc))
                # Continue to the next chunk rather than crashing the worker.

        # All chunks claimed.  Try to become the one worker that triggers assembly.
        async with session_factory() as session:
            chunk_q = ChunkQueue(session)
            won = await chunk_q.try_transition_to_assembling(job_id)

        if won:
            job_logger.info("triggering_assembler", job_id=job_id_str)
            await run_assembler(
                cluster=settings.ecs_cluster,
                service=settings.ecs_worker_service,
                task_definition=settings.ecs_worker_task_definition,
                job_id=job_id,
                region=settings.aws_region,
            )
        else:
            job_logger.info(
                "assembler_already_triggered",
                chunks_processed=chunks_processed,
            )

    except Exception:
        job_logger.exception("chunk_worker_fatal_error")
        async with session_factory() as session:
            queue = PostgresJobQueue(session)
            await queue.mark_job_failed(
                job_id, f"Chunk worker fatal error: {traceback.format_exc()[-500:]}"
            )
    finally:
        await close_db()
        structlog.contextvars.clear_contextvars()
        job_logger.info(
            "chunk_worker_done",
            chunks_processed=chunks_processed,
        )


# ---------------------------------------------------------------------------
# Assembler — reassembles transcript and runs extraction + PDF
# ---------------------------------------------------------------------------

async def assembler_main() -> None:
    """Reassemble all chunk transcripts and run the extraction pipeline.

    Picks up the job from ``assembling`` status, reads chunk transcripts
    in order, joins them, then runs the same LLM extraction → PDF → S3
    upload pipeline as the old monolithic worker.
    """
    job_id_str = os.environ.get("JOB_ID", "")
    if not job_id_str:
        logger.error("assembler_missing_job_id")
        sys.exit(1)

    job_id = UUID(job_id_str)
    worker_id = f"assembler-{os.getpid()}"
    settings = get_settings()

    job_logger = get_logger(
        component="assembler",
        job_id=job_id_str,
        worker_id=worker_id,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(job_id=job_id_str, worker_id=worker_id)

    await init_db()
    session_factory = get_session_factory()
    s3_service = S3StorageService()

    start_time = time.monotonic()
    final_status = "failed"

    job_logger.info("assembler_started")

    try:
        # Check job is actually in assembling state (safety guard).
        async with session_factory() as session:
            queue = PostgresJobQueue(session)
            status = await queue.fetch_job_status(job_id)

        if status == "cancelled":
            job_logger.info("assembler_job_already_cancelled")
            return

        if status != "assembling":
            job_logger.warning(
                "assembler_unexpected_status",
                status=status,
            )
            return

        # --- Reassemble transcript from chunk records ---
        async with session_factory() as session:
            chunk_q = ChunkQueue(session)
            chunk_transcripts = await chunk_q.get_completed_transcripts(job_id)

        if not chunk_transcripts:
            raise RuntimeError("Assembler found no completed chunk transcripts")

        transcript = " ".join(t for t in chunk_transcripts if t)
        if not transcript.strip():
            raise RuntimeError("Assembled transcript is empty")

        job_logger.info(
            "transcript_assembled",
            num_chunks=len(chunk_transcripts),
            total_chars=len(transcript),
        )

        # --- Stage: LLM Extraction ---
        async with session_factory() as session:
            queue = PostgresJobQueue(session)
            await queue.update_job_progress(
                job_id,
                status=ProcessingStatus.EXTRACTING.value,
                progress=50,
                message="Extracting Agile artifacts…",
            )

        async with session_factory() as session:
            queue = PostgresJobQueue(session)
            await _check_cancellation(job_id, queue, job_logger)

        llm_provider = _get_llm_provider()

        _STAGE_PCT = {
            "summarizing": 52,
            "compressing": 60,
            "retrieving": 63,
            "extracting": 66,
        }

        async def _on_extraction_stage(stage: str, detail: str) -> None:
            async with session_factory() as session:
                q = PostgresJobQueue(session)
                await _check_cancellation(job_id, q, job_logger)
                pct = _STAGE_PCT.get(stage, 50)
                await q.update_job_progress(job_id, progress=pct, message=detail)

        artifacts: MeetingArtifacts = await llm_provider.extract_artifacts(
            transcript, job_id=job_id, on_stage=_on_extraction_stage,
        )

        from app.services.heuristics import backfill_confidence_scores
        backfill_confidence_scores(artifacts)

        async with session_factory() as session:
            queue = PostgresJobQueue(session)
            await queue.update_job_progress(
                job_id, progress=72, message="Artifact extraction complete",
            )

        extraction_time = round(time.monotonic() - start_time, 2)
        job_logger.info("extraction_complete", duration_seconds=extraction_time)

        del llm_provider
        gc.collect()

        async with session_factory() as session:
            queue = PostgresJobQueue(session)
            await _check_cancellation(job_id, queue, job_logger)

        # --- Stage: PDF Generation ---
        async with session_factory() as session:
            queue = PostgresJobQueue(session)
            await queue.update_job_progress(
                job_id,
                status=ProcessingStatus.GENERATING_PDF.value,
                progress=75,
                message="Generating PDF summary…",
            )

        with tempfile.TemporaryDirectory(
            prefix=f"meetolog_pdf_{job_id}_"
        ) as pdf_dir:
            pdf_service = _get_pdf_service(Path(pdf_dir))
            pdf_filename = f"meeting_{job_id}.pdf"
            await pdf_service.generate(artifacts, pdf_filename)
            pdf_local_path = Path(pdf_dir) / pdf_filename

            job_logger.info("pdf_generated")

            async with session_factory() as session:
                queue = PostgresJobQueue(session)
                await _check_cancellation(job_id, queue, job_logger)

            # --- Stage: S3 Upload ---
            async with session_factory() as session:
                queue = PostgresJobQueue(session)
                await queue.update_job_progress(
                    job_id, progress=90, message="Uploading results to S3…"
                )

            job_id_str_for_s3 = str(job_id)
            artifacts_dict = artifacts.model_dump(mode="json")
            pdf_s3_key = await s3_service.upload_pdf(str(pdf_local_path), job_id_str_for_s3)
            artifacts_s3_key = await s3_service.upload_artifacts_json(
                artifacts_dict, job_id_str_for_s3
            )

        job_logger.info(
            "s3_upload_complete",
            pdf_s3_key=pdf_s3_key,
            artifacts_s3_key=artifacts_s3_key,
        )

        # --- Mark complete ---
        async with session_factory() as session:
            queue = PostgresJobQueue(session)
            await queue.mark_job_completed(
                job_id,
                artifacts=artifacts_dict,
                pdf_url=f"/download/{job_id}",
                pdf_s3_key=pdf_s3_key,
                artifacts_s3_key=artifacts_s3_key,
            )
        final_status = "completed"

    except _JobCancelledError:
        duration = round(time.monotonic() - start_time, 2)
        job_logger.info(
            "assembler_job_cancelled",
            job_id=job_id_str,
            duration_seconds=duration,
        )
        final_status = "cancelled"

    except Exception:
        job_logger.exception("assembler_error")
        try:
            async with session_factory() as session:
                queue = PostgresJobQueue(session)
                await queue.mark_job_failed(
                    job_id, traceback.format_exc()[-1000:]
                )
        except Exception:
            job_logger.exception("mark_failed_error")

    finally:
        duration = round(time.monotonic() - start_time, 2)
        job_logger.info(
            "assembler_done",
            status=final_status,
            duration_seconds=duration,
        )
        await close_db()
        structlog.contextvars.clear_contextvars()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    service_type = os.environ.get("SERVICE_TYPE", "worker").lower()

    if service_type == "chunk_worker":
        asyncio.run(chunk_worker_main())
    elif service_type == "assembler":
        asyncio.run(assembler_main())
    else:
        # "worker" and "splitter" both run the splitter loop.
        asyncio.run(splitter_loop())
