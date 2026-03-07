"""
Configuration management using Pydantic Settings.
Loads environment variables securely with validation.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    test_mode: bool = Field(
        default=False,
        description="Enable test mode with mock services (no external API calls)"
    )
    
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL for job state and queue"
    )
    
    redis_job_ttl_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="TTL in days for job data in Redis"
    )
    
    llm_provider: Literal["gemini", "openai"] = Field(
        default="gemini",
        description="LLM provider to use for artifact extraction"
    )
    
    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key for LLM extraction"
    )
    
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key for alternative LLM provider"
    )
    
    whisper_model: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="tiny",
        description="Whisper model size. 'tiny' for free tier, 'base' for better accuracy."
    )
    
    app_name: str = Field(default="Meetolog")
    debug: bool = Field(default=False)
    
    max_upload_size_mb: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Maximum upload file size in MB"
    )
    
    allowed_audio_extensions: list[str] = Field(
        default=[".mp3", ".wav", ".m4a", ".ogg", ".webm"]
    )
    
    upload_dir: str = Field(
        default="uploads",
        description="Directory for temporary uploaded files"
    )
    
    output_dir: str = Field(
        default="outputs",
        description="Directory for generated PDFs and job state"
    )
    
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated list of allowed CORS origins"
    )
    
    @field_validator("upload_dir", "output_dir", mode="after")
    @classmethod
    def resolve_to_absolute(cls, v: str) -> str:
        """Resolve relative paths to absolute paths based on cwd."""
        from pathlib import Path
        return str(Path(v).resolve())
    
    @field_validator("gemini_api_key", "openai_api_key", mode="before")
    @classmethod
    def strip_api_key(cls, v: str) -> str:
        """Strip whitespace from API key."""
        if isinstance(v, str):
            return v.strip()
        return v or ""


@lru_cache
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
