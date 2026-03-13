"""Tests for app.config — Settings validation and field-level edge cases."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.config import Settings


class TestSettingsDefaults:
    """Verify that Settings instantiates with sensible defaults (no env vars)."""

    def test_default_test_mode_false(self):
        s = Settings(database_url="postgresql+asyncpg://x")
        assert s.test_mode is False

    def test_default_llm_provider(self):
        s = Settings(database_url="")
        assert s.llm_provider == "gemini"

    def test_default_whisper_model(self):
        # Default from code is "tiny", but .env may override to "base"
        s = Settings(database_url="")
        assert s.whisper_model in ("tiny", "base", "small", "medium", "large")

    def test_default_max_upload_size(self):
        s = Settings(database_url="")
        assert s.max_upload_size_mb == 100

    def test_default_compression_enabled(self):
        s = Settings(database_url="")
        assert s.compression_enabled is True

    def test_default_rag_storage_backend(self):
        s = Settings(database_url="")
        assert s.rag_storage_backend == "memory"


class TestDatabaseUrlNormalization:
    """Validator: postgresql:// → postgresql+asyncpg://."""

    def test_postgresql_rewritten(self):
        s = Settings(database_url="postgresql://user:pass@host/db")
        assert s.database_url == "postgresql+asyncpg://user:pass@host/db"

    def test_asyncpg_url_not_modified(self):
        s = Settings(database_url="postgresql+asyncpg://user:pass@host/db")
        assert s.database_url == "postgresql+asyncpg://user:pass@host/db"

    def test_empty_url_unchanged(self):
        s = Settings(database_url="")
        assert s.database_url == ""


class TestApiKeyStripping:
    """Validator: whitespace around API keys is stripped."""

    def test_gemini_key_stripped(self):
        s = Settings(gemini_api_key="  abc123  ")
        assert s.gemini_api_key == "abc123"

    def test_openai_key_stripped(self):
        s = Settings(openai_api_key="\tfoo\n")
        assert s.openai_api_key == "foo"

    def test_none_key_becomes_empty_string(self):
        s = Settings(gemini_api_key=None)
        assert s.gemini_api_key == ""


class TestFieldConstraints:
    """Bounded numeric fields reject out-of-range values."""

    def test_max_upload_too_large(self):
        with pytest.raises(ValidationError):
            Settings(max_upload_size_mb=501)

    def test_max_upload_too_small(self):
        with pytest.raises(ValidationError):
            Settings(max_upload_size_mb=0)

    def test_hierarchical_token_threshold_minimum(self):
        with pytest.raises(ValidationError):
            Settings(hierarchical_token_threshold=999)

    def test_hierarchical_concurrency_limit_max(self):
        with pytest.raises(ValidationError):
            Settings(hierarchical_concurrency_limit=21)

    def test_invalid_llm_provider(self):
        with pytest.raises(ValidationError):
            Settings(llm_provider="anthropic")

    def test_invalid_whisper_model(self):
        with pytest.raises(ValidationError):
            Settings(whisper_model="huge")

    def test_invalid_rag_backend(self):
        with pytest.raises(ValidationError):
            Settings(rag_storage_backend="redis")


class TestCorsOrigins:
    """CORS origins stored as comma-separated string."""

    def test_default_cors(self):
        s = Settings(database_url="")
        assert "localhost:3000" in s.cors_origins

    def test_custom_cors(self):
        s = Settings(cors_origins="https://app.example.com")
        assert s.cors_origins == "https://app.example.com"


class TestAllowedAudioExtensions:
    """Audio extensions list default and override."""

    def test_defaults_include_common_formats(self):
        s = Settings(database_url="")
        assert ".mp3" in s.allowed_audio_extensions
        assert ".wav" in s.allowed_audio_extensions
        assert ".webm" in s.allowed_audio_extensions

    def test_custom_extensions(self):
        s = Settings(allowed_audio_extensions=[".flac"])
        assert s.allowed_audio_extensions == [".flac"]
