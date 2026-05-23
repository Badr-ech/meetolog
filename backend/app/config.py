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

    gemini_model: str = Field(
        default="gemini-2.5-flash-lite",
        description="Gemini model name used for extraction"
    )

    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name used for extraction"
    )

    whisper_model: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="tiny",
        description="Whisper model size. 'tiny' for free tier, 'base' for better accuracy."
    )

    hf_token: str = Field(
        default="",
        description="HuggingFace access token for pyannote speaker diarization. "
                    "Leave empty to disable diarization."
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

    # Hierarchical summarization
    hierarchical_token_threshold: int = Field(
        default=12000,
        ge=1000,
        description="Transcripts exceeding this token count trigger hierarchical summarization",
    )
    hierarchical_chunk_max_tokens: int = Field(
        default=6000,
        ge=500,
        description="Maximum tokens per chunk in the Map phase",
    )
    hierarchical_chunk_overlap_tokens: int = Field(
        default=200,
        ge=0,
        description="Overlap tokens between consecutive chunks",
    )
    hierarchical_max_summary_tokens: int = Field(
        default=12000,
        ge=1000,
        description="If merged summaries exceed this, an additional Reduce pass is applied",
    )
    hierarchical_concurrency_limit: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Maximum concurrent LLM calls during the Map phase",
    )

    # Context Compression
    compression_enabled: bool = Field(
        default=True,
        description="Enable context compression before final LLM extraction to reduce token usage",
    )
    compression_target_budget_tokens: int = Field(
        default=8000,
        ge=500,
        description="Target token budget for compressed context sent to the extraction prompt",
    )

    # RAG Transcript Retrieval
    rag_chunk_max_tokens: int = Field(
        default=1500,
        ge=100,
        description="Maximum tokens per chunk for RAG indexing (smaller than hierarchical chunks for retrieval precision)",
    )
    rag_chunk_overlap_tokens: int = Field(
        default=100,
        ge=0,
        description="Overlap tokens between consecutive RAG chunks",
    )
    rag_top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of top-K chunks to retrieve per artifact category",
    )
    rag_max_context_tokens: int = Field(
        default=3000,
        ge=500,
        description="Maximum token budget for the RAG context injected into the extraction prompt",
    )
    rag_embedding_batch_size: int = Field(
        default=64,
        ge=1,
        le=2048,
        description="Batch size for embedding API calls during RAG indexing",
    )
    rag_storage_backend: Literal["memory", "pgvector"] = Field(
        default="memory",
        description="Storage backend for the RAG vector index. "
                    "'memory' uses ephemeral NumPy arrays (destroyed when the worker finishes); "
                    "'pgvector' persists embeddings in PostgreSQL via the pgvector extension.",
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

    # ECS worker auto-scaling
    ecs_cluster: str = Field(
        default="meetolog-cluster",
        description="ECS cluster name used for worker auto-scaling",
    )
    ecs_worker_service: str = Field(
        default="meetolog-worker",
        description="ECS service name for the background worker / splitter",
    )
    ecs_worker_task_definition: str = Field(
        default="meetolog-worker",
        description=(
            "ECS task definition family name used when launching chunk-worker "
            "and assembler RunTask calls.  Usually the same as the worker service "
            "task definition."
        ),
    )
    max_parallel_chunks: int = Field(
        default=6,
        ge=1,
        le=24,
        description=(
            "Maximum number of chunk-worker Fargate tasks launched per job. "
            "Each uses 0.5 vCPU, so 6 workers consume 3 vCPUs — well within the "
            "default Fargate 6-vCPU account limit alongside the API and splitter."
        ),
    )
    worker_idle_shutdown_polls: int = Field(
        default=6,
        ge=1,
        description=(
            "Consecutive empty queue polls before the worker scales itself to 0. "
            "Each poll is 5 s apart, so the default of 6 gives a 30 s idle window."
        ),
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
        """Strip whitespace from API key values."""
        if isinstance(v, str):
            return v.strip()
        return v or ""


@lru_cache
def get_settings() -> Settings:
    """Return the cached settings instance."""
    return Settings()
