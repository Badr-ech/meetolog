"""
Transcription service for converting audio to text.
Uses OpenAI Whisper (local model) for Speech-to-Text.

Implements the Transcriber interface for dependency injection.
"""

import asyncio
import logging
from pathlib import Path

from ..interfaces import Transcriber

logger = logging.getLogger(__name__)

# Lazy import whisper to allow mock mode without whisper installed
_whisper = None


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


class WhisperTranscriber(Transcriber):
    def __init__(self, model_name: str = "base"):
        self._model_name = model_name
        self._model = None
        logger.info(f"WhisperTranscriber initialized with model: {model_name}")
    
    def _load_model(self):
        if self._model is None:
            whisper = _get_whisper()
            logger.info(f"Loading Whisper model: {self._model_name}")
            self._model = whisper.load_model(self._model_name)
            logger.info("Whisper model loaded successfully")
        return self._model
    
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
        """
        Transcribe an audio file to text using Whisper.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Transcribed text content
            
        Raises:
            FileNotFoundError: If the audio file doesn't exist
            RuntimeError: If transcription fails
        """
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
    
    async def preprocess_transcript(self, raw_transcript: str) -> str:
        lines = raw_transcript.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:
                line = ' '.join(line.split())
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
