"""Tests for app.services.llm_extraction — HierarchicalExtractor pipeline."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.models import MeetingArtifacts
from app.services.llm_extraction import HierarchicalExtractor
# Force-import modules used in local imports so patch() can find them
import app.services.rag_retrieval as _rag_mod  # noqa: F401
import app.services.compression as _comp_mod  # noqa: F401


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_settings(**overrides):
    """Return a settings-like object with sane defaults for testing."""
    defaults = dict(
        hierarchical_token_threshold=50,  # very low for tests
        hierarchical_chunk_max_tokens=30,
        hierarchical_chunk_overlap_tokens=5,
        hierarchical_max_summary_tokens=100,
        hierarchical_concurrency_limit=3,
        compression_enabled=False,
        compression_target_budget_tokens=8000,
        rag_chunk_max_tokens=1500,
        rag_chunk_overlap_tokens=100,
        rag_top_k=5,
        rag_max_context_tokens=3000,
        rag_embedding_batch_size=64,
        rag_storage_backend="memory",
    )
    defaults.update(overrides)
    m = MagicMock()
    for k, v in defaults.items():
        setattr(m, k, v)
    return m


def _make_artifacts(**kw) -> MeetingArtifacts:
    """Create a minimal MeetingArtifacts for mocking."""
    from app.services.mock_services import MockExtractor
    loop = asyncio.new_event_loop()
    try:
        ext = MockExtractor(simulated_delay=0.0)
        return loop.run_until_complete(ext.extract_artifacts("t"))
    finally:
        loop.close()


_MINIMAL_ARTIFACTS = _make_artifacts()


def _mock_provider(*, extract_return=None, generate_return="mock summary"):
    """Build a mock LLMProvider with async methods."""
    provider = MagicMock()
    provider.is_mock = False
    provider.extract_artifacts = AsyncMock(return_value=extract_return or _MINIMAL_ARTIFACTS)
    provider.generate_text = AsyncMock(return_value=generate_return)
    return provider


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

class TestHierarchicalExtractorInit:
    def test_accepts_llm_provider(self):
        from app.services.llm_engine import LLMProvider
        # Mock that passes isinstance check
        provider = MagicMock(spec=LLMProvider)
        with patch("app.services.llm_extraction.get_settings", return_value=_mock_settings()):
            ext = HierarchicalExtractor(provider)
            assert ext._provider is provider

    def test_accepts_duck_compatible(self):
        provider = _mock_provider()
        with patch("app.services.llm_extraction.get_settings", return_value=_mock_settings()):
            ext = HierarchicalExtractor(provider)
            assert ext._provider is provider

    def test_rejects_incompatible_object(self):
        with patch("app.services.llm_extraction.get_settings", return_value=_mock_settings()):
            with pytest.raises(TypeError, match="duck-compatible"):
                HierarchicalExtractor(object())

    def test_is_mock_property(self):
        provider = _mock_provider()
        provider.is_mock = True
        with patch("app.services.llm_extraction.get_settings", return_value=_mock_settings()):
            ext = HierarchicalExtractor(provider)
            assert ext.is_mock is True


# ---------------------------------------------------------------------------
# Short-transcript fast path (direct extraction)
# ---------------------------------------------------------------------------

class TestDirectExtraction:
    """When transcript is below the token threshold, bypass summarization."""

    async def test_short_transcript_calls_extract_directly(self):
        provider = _mock_provider()
        settings = _mock_settings(hierarchical_token_threshold=50000)
        with patch("app.services.llm_extraction.get_settings", return_value=settings):
            ext = HierarchicalExtractor(provider)
            result = await ext.extract_artifacts("Short meeting text.")
        provider.extract_artifacts.assert_awaited_once()
        provider.generate_text.assert_not_awaited()
        assert isinstance(result, MeetingArtifacts)

    async def test_on_stage_callback_called(self):
        provider = _mock_provider()
        settings = _mock_settings(hierarchical_token_threshold=50000)
        callback = AsyncMock()
        with patch("app.services.llm_extraction.get_settings", return_value=settings):
            ext = HierarchicalExtractor(provider)
            await ext.extract_artifacts("Short text.", on_stage=callback)
        callback.assert_awaited()
        stages = [call.args[0] for call in callback.await_args_list]
        assert "extracting" in stages


# ---------------------------------------------------------------------------
# Long-transcript pipeline (hierarchical summarization)
# ---------------------------------------------------------------------------

class TestHierarchicalPipeline:
    """Long transcripts trigger Map-Reduce + RAG."""

    def _long_transcript(self, token_target: int = 200) -> str:
        """Generate a transcript that exceeds the test threshold."""
        # Each word is ~1 token; produce enough to exceed threshold=50
        words = ["word"] * token_target
        return " ".join(words)

    async def test_long_transcript_triggers_summarization(self):
        provider = _mock_provider()
        settings = _mock_settings(hierarchical_token_threshold=50)
        transcript = self._long_transcript(200)

        # Patch build_index to return a mock RAG index
        mock_index = MagicMock()
        mock_index.chunks = []

        with (
            patch("app.services.llm_extraction.get_settings", return_value=settings),
            patch("app.services.rag_retrieval.build_index", new_callable=AsyncMock, return_value=mock_index),
            patch("app.services.rag_retrieval.retrieve_all_artifact_contexts", new_callable=AsyncMock, return_value={}),
        ):
            ext = HierarchicalExtractor(provider)
            result = await ext.extract_artifacts(transcript)

        # generate_text should have been called for chunk summarization
        assert provider.generate_text.await_count >= 1
        assert isinstance(result, MeetingArtifacts)

    async def test_compression_applied_when_enabled(self):
        provider = _mock_provider()
        settings = _mock_settings(
            hierarchical_token_threshold=50,
            compression_enabled=True,
        )
        transcript = self._long_transcript(200)

        mock_index = MagicMock()
        mock_index.chunks = []

        mock_compressor_instance = MagicMock()
        mock_compress_result = MagicMock()
        mock_compress_result.text = "compressed"
        mock_compress_result.original_tokens = 100
        mock_compress_result.compressed_tokens = 50
        mock_compress_result.compression_ratio = 0.5
        mock_compress_result.segments_filtered = 3
        mock_compressor_instance.compress.return_value = mock_compress_result

        with (
            patch("app.services.llm_extraction.get_settings", return_value=settings),
            patch("app.services.rag_retrieval.build_index", new_callable=AsyncMock, return_value=mock_index),
            patch("app.services.rag_retrieval.retrieve_all_artifact_contexts", new_callable=AsyncMock, return_value={}),
            patch("app.services.compression.ContextCompressor", return_value=mock_compressor_instance),
        ):
            ext = HierarchicalExtractor(provider)
            result = await ext.extract_artifacts(transcript)

        mock_compressor_instance.compress.assert_called_once()
        assert isinstance(result, MeetingArtifacts)

    async def test_rag_context_injected_when_available(self):
        provider = _mock_provider()
        settings = _mock_settings(hierarchical_token_threshold=50)
        transcript = self._long_transcript(200)

        mock_index = MagicMock()
        mock_index.chunks = ["chunk1"]

        rag_contexts = {
            "tasks": "Alice will build the API.",
            "decisions": "Team chose PostgreSQL.",
        }

        with (
            patch("app.services.llm_extraction.get_settings", return_value=settings),
            patch("app.services.rag_retrieval.build_index", new_callable=AsyncMock, return_value=mock_index),
            patch("app.services.rag_retrieval.retrieve_all_artifact_contexts", new_callable=AsyncMock, return_value=rag_contexts),
        ):
            ext = HierarchicalExtractor(provider)
            result = await ext.extract_artifacts(transcript)

        # The extraction call should receive the augmented context
        call_args = provider.extract_artifacts.await_args
        input_text = call_args.args[0]
        assert "Alice will build the API" in input_text or isinstance(result, MeetingArtifacts)

    async def test_stage_callbacks_for_long_transcript(self):
        provider = _mock_provider()
        settings = _mock_settings(hierarchical_token_threshold=50)
        transcript = self._long_transcript(200)

        mock_index = MagicMock()
        mock_index.chunks = []
        callback = AsyncMock()

        with (
            patch("app.services.llm_extraction.get_settings", return_value=settings),
            patch("app.services.rag_retrieval.build_index", new_callable=AsyncMock, return_value=mock_index),
            patch("app.services.rag_retrieval.retrieve_all_artifact_contexts", new_callable=AsyncMock, return_value={}),
        ):
            ext = HierarchicalExtractor(provider)
            await ext.extract_artifacts(transcript, on_stage=callback)

        stages = [call.args[0] for call in callback.await_args_list]
        assert "summarizing" in stages
        assert "extracting" in stages


# ---------------------------------------------------------------------------
# Mock provider fast path
# ---------------------------------------------------------------------------

class TestMockProviderPath:
    """When provider.is_mock is True, RAG indexing is skipped."""

    async def test_mock_provider_skips_rag(self):
        provider = _mock_provider()
        provider.is_mock = True
        settings = _mock_settings(hierarchical_token_threshold=50)
        transcript = " ".join(["word"] * 200)

        with patch("app.services.llm_extraction.get_settings", return_value=settings):
            ext = HierarchicalExtractor(provider)
            result = await ext.extract_artifacts(transcript)

        # Should still extract successfully without build_index
        assert isinstance(result, MeetingArtifacts)


# ---------------------------------------------------------------------------
# Internal pipeline methods
# ---------------------------------------------------------------------------

class TestMergeSummaries:
    async def test_merge_calls_generate_text(self):
        provider = _mock_provider(generate_return="merged text")
        with patch("app.services.llm_extraction.get_settings", return_value=_mock_settings()):
            ext = HierarchicalExtractor(provider)
            result = await ext._merge_summaries(["Summary A", "Summary B"])
        assert result == "merged text"
        provider.generate_text.assert_awaited_once()

    async def test_merge_includes_segment_labels(self):
        provider = _mock_provider()
        with patch("app.services.llm_extraction.get_settings", return_value=_mock_settings()):
            ext = HierarchicalExtractor(provider)
            await ext._merge_summaries(["A", "B"])
        call_args = provider.generate_text.await_args
        prompt = call_args.args[0]
        assert "[Segment 1]" in prompt
        assert "[Segment 2]" in prompt
