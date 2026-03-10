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
    
    cors_origins: str = Field(
        default="http://localhost:3000,http://127.0.0.1:3000",
        description="Comma-separated list of allowed CORS origins"
    )
    
    # AWS S3 configuration
    aws_access_key_id: str = Field(
        default="",
        description="AWS access key ID for S3"
    )
    aws_secret_access_key: str = Field(
        default="",
        description="AWS secret access key for S3"
    )
    aws_region: str = Field(
        default="us-east-1",
        description="AWS region for S3 bucket"
    )
    aws_s3_bucket: str = Field(
        default="",
        description="S3 bucket name for audio file storage"
    )
    aws_endpoint_url: str | None = Field(
        default=None,
        description="Custom S3 endpoint URL (e.g. http://minio:9000 for local MinIO)"
    )
    aws_public_endpoint_url: str | None = Field(
        default=None,
        description="Public S3 endpoint URL for browser-facing presigned URLs (e.g. http://localhost:9000)"
    )
    
    # PostgreSQL configuration
    database_url: str = Field(
        default="",
        description="Async PostgreSQL connection URL (postgresql+asyncpg://...)"
    )

    @field_validator("database_url", mode="after")
    @classmethod
    def normalize_database_url(cls, v: str) -> str:
        if v.startswith("postgresql://"):
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v
    
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
