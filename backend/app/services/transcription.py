"""Transcription service for converting audio to text.

Uses OpenAI Whisper (local model) for Speech-to-Text with a
memory-safe chunked pipeline:

- Audio is split into fixed-duration chunks via ``ffmpeg``
  (see :mod:`app.utils.audio`)
- The Whisper model is loaded **once** per process and cached
- Language is detected on the first chunk and passed explicitly to all
  subsequent chunks.  This prevents Whisper from re-running its 30-second
  language probe on every chunk, which was the root cause of jobs stalling
  on later chunks when the audio contained silence or ambiguous audio.
- Each chunk is transcribed sequentially (single-process path) or one per
  process (parallel-worker path via :meth:`transcribe_single_chunk`)
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
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable

from ..interfaces import Transcriber
from ..utils.audio import AudioSplitError, get_audio_duration, split_audio_into_chunks

logger = logging.getLogger(__name__)

__all__ = [
    "WhisperTranscriber",
    "ProgressCallback",
    "TranscriptSegment",
]

# Callback signature: (chunk_index, total_chunks) -> awaitable
ProgressCallback = Callable[[int, int], Awaitable[None]]

# Lazily imported so TEST_MODE works without openai-whisper installed.
_whisper = None

# Process-wide model cache: avoids the ~2 s disk reload on every transcription.
_model_cache: dict[str, object] = {}


@dataclass(slots=True)
class TranscriptSegment:
    """A single timestamped span of transcribed text.

    Timestamps are in seconds relative to the *full* recording (global
    timeline), not the chunk that produced the segment.
    """
    start: float
    end: float
    text: str


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
    and merges the results.  Language is detected once on the first chunk
    and reused for all subsequent chunks.  A ``gc.collect()`` call after
    every chunk prevents memory bloat during long recordings.
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

    def _transcribe_sync(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> tuple[str, str]:
        """Transcribe *audio_path* and return ``(text, detected_language)``.

        When *language* is ``None`` Whisper probes the first 30 seconds to
        detect it automatically.  Pass the result of the first chunk back in
        for all subsequent chunks to skip re-detection and prevent stalls on
        audio segments that are ambiguous (silence, music, etc.).
        """
        model = self._load_model()

        result = model.transcribe(
            str(audio_path),
            language=language,
            verbose=False,
        )

        text = result.get("text", "").strip()
        detected = result.get("language", "unknown")
        logger.info(
            "Transcription complete. language=%s (requested=%s)",
            detected, language or "auto",
        )
        return text, detected

    def _transcribe_segments_sync(
        self,
        audio_path: Path,
        language: str | None = None,
    ) -> tuple[list[TranscriptSegment], str]:
        """Transcribe and return ``(segments, detected_language)`` (blocking)."""
        model = self._load_model()

        result = model.transcribe(
            str(audio_path),
            language=language,
            verbose=False,
        )

        segments: list[TranscriptSegment] = []
        for seg in result.get("segments", []):
            text = seg.get("text", "").strip()
            if text:
                segments.append(TranscriptSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=text,
                ))
        return segments, result.get("language", "unknown")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transcribe(
        self,
        audio_path: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> str:
        """Transcribe an audio file using chunked processing.

        Language is detected from the first chunk and reused for all
        subsequent chunks, avoiding per-chunk re-detection overhead and
        preventing stalls on ambiguous audio later in the recording.

        Args:
            audio_path: Path to the source audio file.
            progress_callback: Optional async callable ``(chunk_index, total_chunks)``
                invoked after each chunk completes.

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
            detected_language: str | None = None

            for i, chunk_path in enumerate(chunks):
                chunk_text, lang = await asyncio.to_thread(
                    self._transcribe_sync,
                    chunk_path,
                    detected_language,  # None on first chunk → auto-detect
                )
                # Lock in the detected language after the first chunk.
                if detected_language is None:
                    detected_language = lang
                    logger.info("Language locked to '%s' for remaining chunks", lang)

                chunk_text = self._clean_text(chunk_text)
                transcripts.append(chunk_text)

                try:
                    os.remove(chunk_path)
                except OSError:
                    pass

                logger.info(
                    "Chunk %d/%d transcribed (%d chars)",
                    i + 1, total_chunks, len(chunk_text),
                )

                if progress_callback is not None:
                    await progress_callback(i, total_chunks)

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

    async def detect_language_only(self, audio_path: Path) -> str:
        """Probe the first 30 seconds of audio to identify its language.

        Takes ~2 seconds instead of several minutes — uses Whisper's built-in
        ``detect_language()`` which processes only a single 30-second mel
        spectrogram rather than running a full transcription.

        Args:
            audio_path: Path to any audio file Whisper can read.

        Returns:
            ISO 639-1 language code (e.g. ``"en"``, ``"nl"``).

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        def _detect() -> str:
            whisper_mod = _get_whisper()
            model = _get_cached_model(self._model_name)
            audio = whisper_mod.load_audio(str(audio_path))
            audio = whisper_mod.pad_or_trim(audio)  # first 30 seconds only
            mel = whisper_mod.log_mel_spectrogram(audio).to(model.device)
            _, probs = model.detect_language(mel)
            lang = max(probs, key=probs.get)
            logger.info("Language detection complete: %s", lang)
            return lang

        return await asyncio.to_thread(_detect)

    async def transcribe_single_chunk(
        self,
        chunk_path: Path,
        language: str | None = None,
    ) -> tuple[str, str]:
        """Transcribe a single pre-split chunk file.

        Used by the parallel ``chunk_worker`` Fargate tasks, each of which
        handles one audio segment from S3.  Returns ``(text, detected_language)``
        so the worker can persist the detected language alongside the transcript.

        Args:
            chunk_path: Path to the chunk WAV file.
            language:   ISO 639-1 code to pass to Whisper.  ``None`` triggers
                        automatic detection (used when the splitter could not
                        probe the language beforehand).

        Raises:
            FileNotFoundError: If *chunk_path* does not exist.
            RuntimeError: If Whisper fails.
        """
        if not chunk_path.exists():
            raise FileNotFoundError(f"Chunk file not found: {chunk_path}")

        text, detected = await asyncio.to_thread(
            self._transcribe_sync, chunk_path, language
        )
        return self._clean_text(text), detected

    async def transcribe_with_segments(
        self,
        audio_path: Path,
        progress_callback: ProgressCallback | None = None,
    ) -> tuple[str, list[TranscriptSegment]]:
        """Transcribe audio and return both text and globally-timestamped segments.

        Language is detected on the first chunk and locked in for the rest,
        matching the behaviour of :meth:`transcribe`.

        Returns
        -------
        tuple[str, list[TranscriptSegment]]
            *(full_transcript, segments)* — concatenated text plus a
            chronological list of timestamped segments.
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
                "Starting segment-level transcription: %d chunk(s), model=%s",
                total_chunks, self._model_name,
            )

            all_segments: list[TranscriptSegment] = []
            transcripts: list[str] = []
            detected_language: str | None = None

            for i, chunk_path in enumerate(chunks):
                chunk_offset = i * self._chunk_duration_sec

                chunk_segments, lang = await asyncio.to_thread(
                    self._transcribe_segments_sync,
                    chunk_path,
                    detected_language,
                )

                if detected_language is None:
                    detected_language = lang
                    logger.info("Language locked to '%s' for remaining chunks", lang)

                for seg in chunk_segments:
                    seg.start += chunk_offset
                    seg.end += chunk_offset
                    all_segments.append(seg)
                    transcripts.append(self._clean_text(seg.text))

                try:
                    os.remove(chunk_path)
                except OSError:
                    pass

                logger.info(
                    "Chunk %d/%d transcribed (%d segments)",
                    i + 1, total_chunks, len(chunk_segments),
                )

                if progress_callback is not None:
                    await progress_callback(i, total_chunks)

                del chunk_segments
                gc.collect()

            full_transcript = " ".join(t for t in transcripts if t)

            if not full_transcript.strip():
                raise RuntimeError("Transcription returned empty result")

            logger.info(
                "Segment transcription complete: %d chunks, %d segments, %d chars",
                total_chunks, len(all_segments), len(full_transcript),
            )
            return full_transcript, all_segments

        except (AudioSplitError, RuntimeError):
            raise

        except Exception as exc:
            logger.error("Segment transcription failed: %s", exc)
            raise RuntimeError(f"Failed to transcribe audio: {exc}") from exc

        finally:
            shutil.rmtree(chunk_dir, ignore_errors=True)

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
