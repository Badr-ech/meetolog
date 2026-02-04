"""
Configuration management using Pydantic Settings.
Loads environment variables securely with validation.

Key Configuration:
- TEST_MODE: Enable mock services for CI/CD and testing
- GEMINI_API_KEY: Google Gemini API key for LLM extraction
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    Environment Variables:
        TEST_MODE: Set to "true" to use mock services (no API calls)
        GEMINI_API_KEY: Google Gemini API key for production LLM extraction
        WHISPER_MODEL: Whisper model size (tiny, base, small, medium, large)
        DEBUG: Enable debug logging
        MAX_UPLOAD_SIZE_MB: Maximum audio file upload size
        UPLOAD_DIR: Directory for temporary uploaded files
        OUTPUT_DIR: Directory for generated outputs (PDFs, job state)
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # ==========================================================================
    # CI/CD and Testing
    # ==========================================================================
    
    test_mode: bool = Field(
        default=False,
        description="Enable test mode with mock services (no external API calls)"
    )
    
    # ==========================================================================
    # API Keys
    # ==========================================================================
    
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key for LLM extraction"
    )
    
    # ==========================================================================
    # Whisper Configuration
    # ==========================================================================
    
    whisper_model: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="base",
        description="Whisper model size. 'base' is recommended for MVP."
    )
    
    # ==========================================================================
    # Application Settings
    # ==========================================================================
    
    app_name: str = Field(default="Meetolog")
    debug: bool = Field(default=False)
    
    # ==========================================================================
    # File Upload Settings
    # ==========================================================================
    
    max_upload_size_mb: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum upload file size in MB"
    )
    
    allowed_audio_extensions: list[str] = Field(
        default=[".mp3", ".wav", ".m4a", ".ogg", ".webm"]
    )
    
    # ==========================================================================
    # Storage Directories
    # ==========================================================================
    
    upload_dir: str = Field(
        default="uploads",
        description="Directory for temporary uploaded files"
    )
    
    output_dir: str = Field(
        default="outputs",
        description="Directory for generated PDFs and job state"
    )
    
    # ==========================================================================
    # Validators
    # ==========================================================================
    
    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def strip_api_key(cls, v: str) -> str:
        """Strip whitespace from API key."""
        if isinstance(v, str):
            return v.strip()
        return v or ""


@lru_cache
def get_settings() -> Settings:
    """
    Cached settings instance to avoid reloading on every request.
    
    Returns:
        Settings instance with validated configuration
    """
    return Settings()
