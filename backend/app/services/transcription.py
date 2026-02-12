"""
Transcription service for converting audio to text.
Uses OpenAI Whisper (local model) for Speech-to-Text.

Supports chunked transcription for resilience on resource-constrained
platforms (e.g., Render free tier):
- Audio is split into chunks via ffmpeg
- Each chunk is transcribed independently
- Partial results are cached after each chunk
- Lower peak memory usage = fewer OOM kills

Implements the Transcriber interface for dependency injection.
"""

import asyncio
import logging
import subprocess
import tempfile
from pathlib import Path

from ..interfaces import Transcriber

logger = logging.getLogger(__name__)

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


# =============================================================================
# Audio Utilities
# =============================================================================

def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration in seconds using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=30,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not get audio duration: {e}")
        return 0.0


def split_audio_into_chunks(
    audio_path: Path,
    chunk_dir: Path,
    chunk_duration_seconds: int = 300,
) -> list[Path]:
    """
    Split an audio file into fixed-duration chunks using ffmpeg.
    
    Args:
        audio_path: Path to the source audio file
        chunk_dir: Directory to write chunk files into
        chunk_duration_seconds: Duration of each chunk in seconds (default: 5 min)
        
    Returns:
        List of paths to chunk files, ordered by time
    """
    chunk_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem
    
    # Use ffmpeg segment muxer to split into chunks
    chunk_pattern = str(chunk_dir / f"{stem}_chunk_%03d.wav")
    
    cmd = [
        "ffmpeg", "-y", "-i", str(audio_path),
        "-f", "segment",
        "-segment_time", str(chunk_duration_seconds),
        "-ar", "16000",   # Whisper's native sample rate
        "-ac", "1",       # Mono (Whisper's native channels)
        "-c:a", "pcm_s16le",
        chunk_pattern,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg split failed: {result.stderr[:500]}")
    
    # Collect chunk files in order
    chunks = sorted(chunk_dir.glob(f"{stem}_chunk_*.wav"))
    logger.info(f"Split audio into {len(chunks)} chunks ({chunk_duration_seconds}s each)")
    return chunks


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
    def __init__(self, model_name: str = "base"):
        self._model_name = model_name
        logger.info(f"WhisperTranscriber initialized with model: {model_name}")
    
    def _load_model(self):
        return _get_cached_model(self._model_name)
    
    def _transcribe_sync(self, audio_path: Path) -> str:
        model = self._load_model()
        
        logger.info(f"Transcribing audio file: {audio_path}")
        result = model.transcribe(
            str(audio_path),
            language=None,  # Auto-detect language
            verbose=False,
        )
        
        transcript = result.get("text", "").strip()
        detected_language = result.get("language", "unknown")
        logger.info(f"Transcription complete. Detected language: {detected_language}")
        
        return transcript
    
    async def transcribe(self, audio_path: Path) -> str:
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        try:
            transcript = await asyncio.to_thread(self._transcribe_sync, audio_path)
            
            if not transcript:
                raise RuntimeError("Transcription returned empty result")
            
            return transcript
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise RuntimeError(f"Failed to transcribe audio: {e}") from e
    
    async def transcribe_chunk(self, chunk_path: Path) -> str:
        """Transcribe a single audio chunk. Same as transcribe() but with clearer intent."""
        return await self.transcribe(chunk_path)
    
    async def preprocess_transcript(self, raw_transcript: str) -> str:
        lines = raw_transcript.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                line = ' '.join(line.split())
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
