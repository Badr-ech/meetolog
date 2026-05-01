"""
PostgreSQL-backed background worker for Meetolog.

Polls the ``job_records`` table using ``SELECT … FOR UPDATE SKIP LOCKED``
to claim pending jobs, then runs the transcription → extraction → PDF
pipeline entirely within a temporary directory.  All outputs are uploaded
to S3 — the worker holds zero persistent local state, enabling safe
horizontal scaling across multiple replicas.
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

# Allow ``python -m app.worker`` from inside the backend/ directory.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import get_settings
from app.core.logger import configure_logging, get_logger
from app.models import MeetingArtifacts, ProcessingStatus
from app.infrastructure.db import get_session_factory, close_db, init_db
from app.infrastructure.postgres_queue import PostgresJobQueue
from app.services.storage import S3StorageService

configure_logging(json_output=True, log_level="INFO")

logger = get_logger(component="worker")

# How long to sleep between Postgres polls when the queue is empty.
POLL_INTERVAL_SECONDS = 5


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
    provider = _get_provider()
    return HierarchicalExtractor(provider)


def _get_pdf_service(output_dir: Path):
    """Return a PDFGeneratorService writing into the given temp directory."""
    from app.services.pdf_generator import PDFGeneratorService
    return PDFGeneratorService(output_dir)


async def process_job(
    job_id: UUID,
    s3_key: str,
    file_name: str,
    queue: PostgresJobQueue,
    s3_service: S3StorageService,
    worker_id: str = "unknown",
) -> None:
    """Run the full transcription -> extraction -> PDF -> S3 pipeline.

    All intermediate files live inside a ``TemporaryDirectory`` that is
    destroyed automatically when the context exits — whether the pipeline
    succeeds or crashes.  Final outputs (PDF, JSON) are persisted to S3
    before the temp dir is removed, keeping the worker fully stateless.

    The entire pipeline is wrapped in a catch-all exception handler so a
    single bad job can never crash the worker loop.
    """
    # Bind structured context for every log emitted during this job.
    job_logger = get_logger(
        component="worker",
        job_id=str(job_id),
        worker_id=worker_id,
        file_name=file_name,
    )
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(job_id=str(job_id), worker_id=worker_id)

    start_time = time.monotonic()
    final_status = "failed"
    job_logger.info("job_started", s3_key=s3_key)

    try:
        with tempfile.TemporaryDirectory(prefix=f"meetolog_{job_id}_") as work_dir:
            work_path = Path(work_dir)
            file_ext = Path(file_name).suffix.lower() or ".wav"
            audio_path = work_path / f"{job_id}{file_ext}"

            # --- Download audio from S3 ---
            job_logger.info("s3_download_start", s3_key=s3_key)
            await s3_service.download_to_file(s3_key, str(audio_path))

            # --- Stage 0 (optional): Speaker Diarization ---
            settings = get_settings()
            diarization_enabled = bool(settings.hf_token) and not settings.test_mode
            diarization_timeline = None

            if diarization_enabled:
                await queue.update_job_progress(
                    job_id,
                    status=ProcessingStatus.DIARIZING.value,
                    progress=5,
                    message="Identifying speakers…",
                )

                from app.services.diarization import SpeakerDiarizer
                from app.utils.audio import convert_to_mono_wav

                mono_wav_path = work_path / f"{job_id}_mono.wav"
                job_logger.info("converting_mono_wav")
                await asyncio.to_thread(convert_to_mono_wav, audio_path, mono_wav_path)

                diarizer = SpeakerDiarizer(settings.hf_token)
                diarization_timeline = await diarizer.diarize(mono_wav_path)

                # Free diarization resources before Whisper loads
                del diarizer
                mono_wav_path.unlink(missing_ok=True)
                gc.collect()

                await queue.update_job_progress(
                    job_id,
                    progress=18,
                    message=f"Speaker analysis complete — {len(set(s.speaker for s in diarization_timeline))} speakers detected",
                )
                job_logger.info(
                    "diarization_complete",
                    num_turns=len(diarization_timeline),
                    num_speakers=len(set(s.speaker for s in diarization_timeline)),
                )

            # --- Stage 1: Transcription ---
            await queue.update_job_progress(
                job_id,
                status=ProcessingStatus.TRANSCRIBING.value,
                progress=20 if diarization_enabled else 25,
                message="Transcribing audio...",
            )

            transcriber = _get_transcriber()

            if diarization_enabled and diarization_timeline is not None:
                # Transcribe with segment timestamps for speaker alignment
                base_pct = 20
                span_pct = 25

                async def _on_transcription_progress(chunk_idx: int, total: int) -> None:
                    pct = base_pct + int(span_pct * ((chunk_idx + 1) / total))
                    job_logger.info(
                        "transcription_progress",
                        chunk=chunk_idx + 1,
                        total_chunks=total,
                        progress_pct=pct,
                    )
                    await queue.update_job_progress(
                        job_id,
                        progress=pct,
                        message=f"Transcribing chunk {chunk_idx + 1}/{total}...",
                    )

                _, segments = await transcriber.transcribe_with_segments(
                    audio_path,
                    progress_callback=_on_transcription_progress,
                )

                from app.services.diarization import SpeakerDiarizer as _SD
                transcript = _SD.assign_speakers(segments, diarization_timeline)
                del segments, diarization_timeline
                gc.collect()
            else:
                # Original path — plain text transcription
                async def _on_transcription_progress(chunk_idx: int, total: int) -> None:
                    pct = 25 + int(23 * ((chunk_idx + 1) / total))
                    job_logger.info(
                        "transcription_progress",
                        chunk=chunk_idx + 1,
                        total_chunks=total,
                        progress_pct=pct,
                    )
                    await queue.update_job_progress(
                        job_id,
                        progress=pct,
                        message=f"Transcribing chunk {chunk_idx + 1}/{total}...",
                    )

                transcript = await transcriber.transcribe(
                    audio_path,
                    progress_callback=_on_transcription_progress,
                )

            await queue.update_job_progress(
                job_id, progress=48, message="Transcription complete",
            )
            transcription_time = time.monotonic() - start_time
            job_logger.info("transcription_complete", duration_seconds=round(transcription_time, 2))

            # --- Stage 2: LLM Extraction ---
            await queue.update_job_progress(
                job_id,
                status=ProcessingStatus.EXTRACTING.value,
                progress=50,
                message="Extracting Agile artifacts...",
            )

            llm_provider = _get_llm_provider()

            # Granular progress callback — the pipeline invokes this
            # at each stage so the user sees real-time updates.
            _STAGE_PCT = {
                "summarizing": 52,
                "compressing": 60,
                "retrieving": 63,
                "extracting": 66,
            }

            async def _on_extraction_stage(stage: str, detail: str) -> None:
                pct = _STAGE_PCT.get(stage, 50)
                await queue.update_job_progress(job_id, progress=pct, message=detail)

            artifacts: MeetingArtifacts = await llm_provider.extract_artifacts(
                transcript, job_id=job_id, on_stage=_on_extraction_stage,
            )

            from app.services.heuristics import backfill_confidence_scores
            backfill_confidence_scores(artifacts)

            await queue.update_job_progress(
                job_id, progress=72, message="Artifact extraction complete",
            )
            extraction_time = time.monotonic() - start_time - transcription_time
            job_logger.info("extraction_complete", duration_seconds=round(extraction_time, 2))

            # Free extraction objects before PDF generation.
            del llm_provider
            gc.collect()

            # --- Stage 3: PDF Generation (into temp dir) ---
            await queue.update_job_progress(
                job_id,
                status=ProcessingStatus.GENERATING_PDF.value,
                progress=75,
                message="Generating PDF summary...",
            )

            pdf_service = _get_pdf_service(work_path)
            pdf_filename = f"meeting_{job_id}.pdf"
            await pdf_service.generate(artifacts, pdf_filename)
            pdf_local_path = work_path / pdf_filename

            pdf_time = time.monotonic() - start_time - transcription_time - extraction_time
            job_logger.info("pdf_generated", duration_seconds=round(pdf_time, 2))

            # --- Stage 4: Upload results to S3 ---
            await queue.update_job_progress(
                job_id, progress=90, message="Uploading results to S3...",
            )

            job_id_str = str(job_id)
            artifacts_dict = artifacts.model_dump(mode="json")
            pdf_s3_key = await s3_service.upload_pdf(str(pdf_local_path), job_id_str)
            artifacts_s3_key = await s3_service.upload_artifacts_json(artifacts_dict, job_id_str)

            job_logger.info("s3_upload_complete", pdf_s3_key=pdf_s3_key, artifacts_s3_key=artifacts_s3_key)

            # --- Stage 5: Mark complete ---
            await queue.mark_job_completed(
                job_id,
                artifacts=artifacts_dict,
                pdf_url=f"/download/{job_id}",
                pdf_s3_key=pdf_s3_key,
                artifacts_s3_key=artifacts_s3_key,
            )
            final_status = "completed"

    except FileNotFoundError as exc:
        job_logger.error("job_error", error=f"Audio file not found: {exc}")
        try:
            await queue.mark_job_failed(job_id, f"Audio file not found: {exc}")
        except Exception:
            job_logger.exception("mark_failed_error")

    except Exception:
        job_logger.exception("job_error")
        try:
            await queue.mark_job_failed(job_id, traceback.format_exc()[-1000:])
        except Exception:
            job_logger.exception("mark_failed_error")

    finally:
        duration = round(time.monotonic() - start_time, 2)
        job_logger.info(
            "job_finished",
            status=final_status,
            duration_seconds=duration,
        )
        structlog.contextvars.clear_contextvars()


# ------------------------------------------------------------------
# Async worker loop
# ------------------------------------------------------------------

_shutdown_event = asyncio.Event()


def _request_shutdown() -> None:
    """Signal handler — request a graceful loop exit."""
    logger.info("shutdown_requested")
    _shutdown_event.set()


async def worker_loop(worker_id: str | None = None) -> None:
    """Continuously poll the Postgres job queue and process jobs.

    The loop exits cleanly on SIGINT / SIGTERM.  If no job is available
    it sleeps for ``POLL_INTERVAL_SECONDS`` to avoid wasteful queries.
    """
    if worker_id is None:
        worker_id = f"worker-{os.getpid()}"

    # Register signal handlers for graceful shutdown. ``add_signal_handler``
    # is unsupported on Windows, so we fall back to ``signal.signal`` there.
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_shutdown)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _request_shutdown())

    await init_db()
    session_factory = get_session_factory()
    s3_service = S3StorageService()

    # Whisper is loaded on demand, not at worker startup. When diarization is
    # active the pyannote model needs the full RAM budget; loading Whisper
    # eagerly would push peak usage beyond the 2 GB Fargate Spot budget.

    logger.info("worker_started", worker_id=worker_id, poll_interval=POLL_INTERVAL_SECONDS)

    try:
        while not _shutdown_event.is_set():
            async with session_factory() as session:
                queue = PostgresJobQueue(session)
                record = await queue.claim_next_job(worker_id)

            if record is None:
                try:
                    await asyncio.wait_for(
                        _shutdown_event.wait(),
                        timeout=POLL_INTERVAL_SECONDS,
                    )
                except asyncio.TimeoutError:
                    pass
                continue

            logger.info("job_claimed", job_id=str(record.id), worker_id=worker_id)

            # Each job gets its own session so a failure cannot poison
            # the poll loop's connection state.
            async with session_factory() as job_session:
                job_queue = PostgresJobQueue(job_session)
                await process_job(
                    job_id=record.id,
                    s3_key=record.s3_key,
                    file_name=record.file_name,
                    queue=job_queue,
                    s3_service=s3_service,
                    worker_id=worker_id,
                )
    finally:
        await close_db()
        logger.info("worker_stopped", worker_id=worker_id)


if __name__ == "__main__":
    asyncio.run(worker_loop())
