"""
Audio processing utilities for Meetolog.

Provides ffmpeg-based audio splitting and duration probing for chunked
transcription.  All operations write to disk (not RAM) to prevent OOM
on resource-constrained environments (e.g., Render free tier, CPU-only
machines).

Usage:
    from app.utils.audio import split_audio_into_chunks, get_audio_duration
"""

import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: Path) -> float:
    """Return audio duration in seconds using ``ffprobe``.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in seconds, or ``0.0`` if probing fails.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(audio_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not probe audio duration: {e}")
        return 0.0


def split_audio_into_chunks(
    audio_path: Path,
    chunk_dir: Path,
    chunk_duration_seconds: int = 300,
) -> list[Path]:
    """Split an audio file into fixed-duration chunks using ``ffmpeg``.

    Chunks are written to *chunk_dir* as 16 kHz mono WAV files (Whisper's
    native format).  No chunk data is held in memory — only file paths.

    Args:
        audio_path: Path to the source audio file.
        chunk_dir: Directory to write chunk files into (created if absent).
        chunk_duration_seconds: Duration of each chunk in seconds
            (default ``300`` = 5 minutes).

    Returns:
        List of chunk file paths, ordered chronologically.

    Raises:
        RuntimeError: If the ``ffmpeg`` command exits with a non-zero code.
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    chunk_pattern = str(chunk_dir / f"{stem}_chunk_%03d.wav")

    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(chunk_duration_seconds),
        "-ar", "16000",   # Whisper native sample rate
        "-ac", "1",       # mono
        "-c:a", "pcm_s16le",
        chunk_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg chunk split failed: {result.stderr[:500]}")

    chunks = sorted(chunk_dir.glob(f"{stem}_chunk_*.wav"))
    logger.info(
        f"Split audio into {len(chunks)} chunk(s) "
        f"({chunk_duration_seconds}s each)"
    )
    return chunks
