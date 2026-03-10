"""Transcription service for converting audio to text.

Uses OpenAI Whisper (local model) for Speech-to-Text with a
memory-safe chunked pipeline:

- Audio is split into fixed-duration chunks via ``ffmpeg``
  (see :mod:`app.utils.audio`)
- The Whisper model is loaded **once** per process and cached
- Each chunk is transcribed **sequentially** (no parallelism)
- ``gc.collect()`` is called after every chunk to reclaim memory
- Chunk files are deleted from disk immediately after transcription
- A caller-supplied ``progress_callback`` enables granular progress
  tracking in PostgreSQL

Implements the :class:`Transcriber` interface for dependency injection.
"""

import asyncio
import gc
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Awaitable, Callable

from ..interfaces import Transcriber
from ..utils.audio import AudioSplitError, get_audio_duration, split_audio_into_chunks

logger = logging.getLogger(__name__)

__all__ = [
    "WhisperTranscriber",
    "ProgressCallback",
    "_get_cached_model",
]

# 2-arg callback: (chunk_index, total_chunks)
ProgressCallback = Callable[[int, int], Awaitable[None]]

# Lazy import whisper to allow mock mode without whisper installed
_whisper = None
# Singleton model cache: keeps the loaded model in memory across jobs
# so we don't reload from disk (~2-3s) on every transcription request
_model_cache: dict[str, object] = {}


def _get_whisper():
    global _whisper
    if _whisper is None:
        try:
            import whisper
            _whisper = whisper
        except ImportError as e:
            logger.error(f"Failed to import whisper: {e}")
            raise RuntimeError(
                "openai-whisper is not installed. Install it with: pip install openai-whisper\n"
                "Or set TEST_MODE=true to use mock transcription."
            ) from e
    return _whisper


def _get_cached_model(model_name: str):
    """Load and cache a Whisper model. Subsequent calls return the cached instance."""
    if model_name not in _model_cache:
        whisper = _get_whisper()
        logger.info(f"Loading Whisper model: {model_name}")
        _model_cache[model_name] = whisper.load_model(model_name)
        logger.info(f"Whisper model '{model_name}' loaded and cached")
    return _model_cache[model_name]


class WhisperTranscriber(Transcriber):
    """
    Chunked Whisper transcriber with memory-safe processing.

    Splits audio into fixed-duration chunks, transcribes each sequentially,
    and merges the results.  A ``gc.collect()`` call after every chunk
    prevents memory bloat during long recordings.
    """

    def __init__(
        self,
        model_name: str = "base",
        chunk_duration_sec: int = 300,
    ):
        self._model_name = model_name
        self._chunk_duration_sec = chunk_duration_sec
        logger.info(
            f"WhisperTranscriber initialized "
            f"(model={model_name}, chunk_duration={chunk_duration_sec}s)"
        )

    def _load_model(self):
        return _get_cached_model(self._model_name)

    def _transcribe_sync(self, audio_path: Path) -> str:
        model = self._load_model()

        logger.info(f"Transcribing audio file: {audio_path}")
        result = model.transcribe(
            str(audio_path),
            language=None,   # Auto-detect language
            verbose=False,
        )

        transcript = result.get("text", "").strip()
        detected_language = result.get("language", "unknown")
        logger.info(f"Transcription complete. Detected language: {detected_language}")

        return transcript

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio_path: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> str:
        """Transcribe an audio file using chunked processing.

        Pipeline:
            1. Split audio into ``chunk_duration_sec``-second WAV chunks
               via ffmpeg (disk I/O only, nothing held in RAM).
            2. Load the Whisper model **once** (cached across calls).
            3. For each chunk:
               a. Transcribe (offloaded to the thread pool).
               b. Delete the chunk file from disk immediately.
               c. ``del`` the result dict and call ``gc.collect()``.
               d. Invoke *progress_callback* if provided.
            4. Return the concatenated transcript.

        Args:
            audio_path: Path to the source audio file.
            progress_callback: Optional async callable ``(chunk_index, total_chunks)``
                invoked after each chunk completes.  Used by the worker to write
                per-chunk progress to PostgreSQL.

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
            RuntimeError: If transcription fails.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_model()

        chunk_dir = Path(tempfile.mkdtemp(prefix="meetolog_chunks_"))

        try:
            chunks = await asyncio.to_thread(
                split_audio_into_chunks,
                audio_path,
                chunk_dir,
                self._chunk_duration_sec,
            )
            total_chunks = len(chunks)

            logger.info(
                "Starting chunked transcription: %d chunk(s), model=%s",
                total_chunks, self._model_name,
            )

            transcripts: list[str] = []

            for i, chunk_path in enumerate(chunks):
                chunk_text = await asyncio.to_thread(
                    self._transcribe_sync, chunk_path,
                )
                chunk_text = self._clean_text(chunk_text)
                transcripts.append(chunk_text)

                # Free disk space for this chunk immediately
                try:
                    os.remove(chunk_path)
                except OSError:
                    pass

                logger.info(
                    "Chunk %d/%d transcribed (%d chars)",
                    i + 1, total_chunks, len(chunk_text),
                )

                # Notify via the 2-arg callback
                if progress_callback is not None:
                    await progress_callback(i, total_chunks)

                # Reclaim memory aggressively
                del chunk_text
                gc.collect()

            full_transcript = " ".join(t for t in transcripts if t)

            if not full_transcript.strip():
                raise RuntimeError("Transcription returned empty result")

            logger.info(
                "Chunked transcription complete: %d chunks, %d chars total",
                total_chunks, len(full_transcript),
            )
            return full_transcript

        except (AudioSplitError, RuntimeError):
            raise

        except Exception as exc:
            logger.error("Chunked transcription failed: %s", exc)
            raise RuntimeError(f"Failed to transcribe audio: {exc}") from exc

        finally:
            shutil.rmtree(chunk_dir, ignore_errors=True)
            logger.debug("Cleaned up chunk directory: %s", chunk_dir)

    async def transcribe_chunk(self, chunk_path: Path) -> str:
        """Transcribe a single audio chunk (kept for backward compatibility)."""
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")
        text = await asyncio.to_thread(self._transcribe_sync, chunk_path)
        return self._clean_text(text)

    async def preprocess_transcript(self, raw_transcript: str) -> str:
        return self._clean_text(raw_transcript)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Basic whitespace normalisation."""
        lines = text.strip().split("\n")
        cleaned: list[str] = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned.append(" ".join(line.split()))
        return "\n".join(cleaned)
