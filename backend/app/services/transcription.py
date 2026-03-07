"""
Transcription service for converting audio to text.
Uses OpenAI Whisper (local model) for Speech-to-Text.

Supports chunked transcription for memory safety on CPU-only machines:
- Audio is split into fixed-duration chunks via ``ffmpeg``
  (see :mod:`app.utils.audio`)
- Each chunk is transcribed sequentially (no parallelism)
- ``gc.collect()`` is called after every chunk to reclaim memory
- All temporary chunk files are cleaned up in a ``try / finally`` block

Implements the :class:`Transcriber` interface for dependency injection.
"""

import asyncio
import gc
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Awaitable, Callable

from ..interfaces import Transcriber
from ..utils.audio import get_audio_duration, split_audio_into_chunks

logger = logging.getLogger(__name__)

# Re-export audio utilities so that existing ``from app.services.transcription
# import split_audio_into_chunks`` statements keep working.
__all__ = [
    "WhisperTranscriber",
    "ChunkProgressCallback",
    "compress_audio_for_storage",
    "decompress_audio",
    "get_audio_duration",
    "split_audio_into_chunks",
    "_get_cached_model",
]

# Type alias for the per-chunk progress callback.
#   (chunk_index: int, total_chunks: int, chunk_text: str) -> Awaitable[None]
ChunkProgressCallback = Callable[[int, int, str], Awaitable[None]]

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


def compress_audio_for_storage(audio_path: Path) -> bytes:
    """
    Compress audio to a small format for Redis storage.
    
    Converts to 16kHz mono Opus at 32kbps. A 42-minute meeting
    compresses from ~42MB WAV to ~5-8MB.
    
    Args:
        audio_path: Path to the source audio file
        
    Returns:
        Compressed audio as bytes
    """
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        cmd = [
            "ffmpeg", "-y", "-i", str(audio_path),
            "-ar", "16000", "-ac", "1",
            "-c:a", "libopus", "-b:a", "32k",
            str(tmp_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg compression failed: {result.stderr[:500]}")
        
        compressed = tmp_path.read_bytes()
        logger.info(
            f"Audio compressed: {audio_path.stat().st_size / 1024 / 1024:.1f}MB → "
            f"{len(compressed) / 1024 / 1024:.1f}MB"
        )
        return compressed
    finally:
        tmp_path.unlink(missing_ok=True)


def decompress_audio(audio_bytes: bytes, output_path: Path) -> Path:
    """
    Decompress stored audio bytes back to a WAV file for Whisper.
    
    Args:
        audio_bytes: Compressed audio data (Opus/OGG)
        output_path: Where to write the decompressed WAV
        
    Returns:
        Path to the decompressed WAV file
    """
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    try:
        tmp_path.write_bytes(audio_bytes)
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "ffmpeg", "-y", "-i", str(tmp_path),
            "-ar", "16000", "-ac", "1",
            "-c:a", "pcm_s16le",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg decompression failed: {result.stderr[:500]}")
        
        logger.info(f"Audio decompressed to: {output_path}")
        return output_path
    finally:
        tmp_path.unlink(missing_ok=True)


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
        self.on_chunk_complete: ChunkProgressCallback | None = None
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

    async def transcribe(self, audio_path: Path) -> str:
        """
        Transcribe an audio file using chunked processing.

        1. Splits the audio into ``chunk_duration_sec``-second chunks via
           ``ffmpeg`` (all I/O goes to disk, not RAM).
        2. Loads the Whisper model **once** (cached across invocations).
        3. Transcribes each chunk **sequentially** – no parallelism.
        4. Calls ``gc.collect()`` after every chunk for memory safety.
        5. Merges chunk transcripts with a single-space separator.

        If :attr:`on_chunk_complete` is set, it is awaited with
        ``(chunk_index, total_chunks, chunk_text)`` after each chunk,
        enabling the worker to report per-chunk progress to Redis.

        All temporary chunk files are removed in a ``finally`` block even
        if transcription fails mid-way.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Pre-load the model so any import errors surface early
        self._load_model()

        chunk_dir = Path(tempfile.mkdtemp(prefix="meetolog_chunks_"))

        try:
            # --- Split audio into chunks on disk ---
            chunks = await asyncio.to_thread(
                split_audio_into_chunks,
                audio_path,
                chunk_dir,
                self._chunk_duration_sec,
            )
            total_chunks = len(chunks)

            if total_chunks == 0:
                raise RuntimeError("Audio split produced zero chunks")

            logger.info(
                f"Starting chunked transcription: {total_chunks} chunk(s), "
                f"model={self._model_name}"
            )

            transcripts: list[str] = []

            for i, chunk_path in enumerate(chunks):
                # Transcribe single chunk (blocking work in thread pool)
                chunk_text = await asyncio.to_thread(
                    self._transcribe_sync, chunk_path,
                )

                # Basic whitespace normalisation
                chunk_text = self._clean_text(chunk_text)
                transcripts.append(chunk_text)

                logger.info(
                    f"Chunk {i + 1}/{total_chunks} transcribed "
                    f"({len(chunk_text)} chars)"
                )

                # Notify the caller (e.g. worker progress updates)
                if self.on_chunk_complete is not None:
                    await self.on_chunk_complete(i, total_chunks, chunk_text)

                # Explicit memory reclamation after each chunk
                del chunk_text
                gc.collect()

            full_transcript = " ".join(t for t in transcripts if t)

            if not full_transcript.strip():
                raise RuntimeError("Transcription returned empty result")

            return full_transcript

        except Exception as e:
            logger.error(f"Chunked transcription failed: {e}")
            raise RuntimeError(f"Failed to transcribe audio: {e}") from e

        finally:
            # Always clean up chunk files, even on failure
            shutil.rmtree(chunk_dir, ignore_errors=True)
            logger.debug(f"Cleaned up chunk directory: {chunk_dir}")

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
