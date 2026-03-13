"""
Unified intelligence pipeline — hierarchical summarization, RAG,
compression, and structured artifact extraction.

Orchestrates the full intelligence layer for long meeting transcripts:

1. **Conditional Routing** — short transcripts bypass summarization
   and go straight to the provider's extraction prompt.
2. **Concurrent Processing** — for long transcripts, hierarchical
   Map-Reduce summarization and RAG transcript indexing run in
   parallel via ``asyncio.gather``.
3. **Context Compression** — the condensed summary is filtered through
   semantic scoring to remove filler and prioritise high-density
   actionable segments.
4. **RAG-Augmented Extraction** — per-artifact-category top-K segments
   are retrieved from the transcript index and injected alongside the
   compressed summary into the extraction prompt.
5. **Validation & Fallback** — the underlying provider handles
   temperature-fallback retries and Pydantic schema enforcement.
"""

from __future__ import annotations

import asyncio
from typing import Awaitable, Callable
from uuid import UUID

import structlog

from ..config import get_settings
from ..core.prompts import (
    CHUNK_SUMMARIZATION_PROMPT,
    MERGE_SUMMARIZATION_PROMPT,
    RAG_AUGMENTED_EXTRACTION_CONTEXT,
)
from ..models import MeetingArtifacts
from ..utils.text_chunking import (
    chunk_transcript,
    count_tokens,
    needs_hierarchical_summarization,
)

logger = structlog.get_logger(__name__)

# Async callback type for reporting pipeline stage transitions.
# Signature: (stage_name: str, detail_message: str) -> None
StageCallback = Callable[[str, str], Awaitable[None]]


# ======================================================================
# Unified Intelligence Pipeline
# ======================================================================

class HierarchicalExtractor:
    """Map-Reduce extraction pipeline for long meeting transcripts.

    When a transcript exceeds ``hierarchical_token_threshold`` tokens the
    pipeline proceeds as:

    1. **Chunk** — split the transcript into token-bounded blocks with
       configurable overlap.
    2. **Map (Level 1)** — summarise each chunk concurrently using the
       underlying LLM provider, retaining all actionable artifacts.
    3. **Reduce (Level 2+)** — if the concatenated chunk summaries still
       exceed ``hierarchical_max_summary_tokens``, recursively merge them
       until the text fits.
    4. **Extract** — pass the condensed summary (or the original transcript
       when short enough) through the provider's structured artifact
       extraction prompt.

    The class wraps *any* ``LLMProvider`` (Gemini, OpenAI, Mock) and is
    fully compatible with the existing stateless worker pipeline.
    """

    def __init__(self, provider):
        """Initialise the orchestrator with an ``LLMProvider`` instance."""
        from .llm_engine import LLMProvider
        if not isinstance(provider, LLMProvider):
            # Duck-typing fallback for MockExtractor
            if not (hasattr(provider, "extract_artifacts") and hasattr(provider, "generate_text")):
                raise TypeError(
                    f"provider must be an LLMProvider or duck-compatible, got {type(provider).__name__}"
                )
        self._provider = provider
        self._settings = get_settings()
        self._logger = structlog.get_logger(
            __name__, component="hierarchical_extractor"
        )

    @property
    def is_mock(self) -> bool:
        return self._provider.is_mock

    def _get_session_factory(self):
        """Return the async session factory when pgvector backend is active."""
        if self._settings.rag_storage_backend != "pgvector":
            return None
        from ..infrastructure.db import get_session_factory
        return get_session_factory()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def extract_artifacts(
        self,
        transcript: str,
        *,
        job_id: UUID | None = None,
        on_stage: StageCallback | None = None,
    ) -> MeetingArtifacts:
        """Run the unified intelligence pipeline.

        Pipeline stages (long transcripts only):

        1. **summarizing** — Hierarchical Map-Reduce summarization
           and RAG index build run concurrently.
        2. **compressing** — Context compression filters filler and
           prioritises high-density actionable segments.
        3. **retrieving** — Per-artifact-category top-K segments are
           retrieved from the transcript index.
        4. **extracting** — The provider's structured extraction prompt
           receives the compressed summary + RAG segments.

        Short transcripts (below ``hierarchical_token_threshold``)
        bypass stages 1–3 and go directly to the provider.

        Parameters
        ----------
        transcript:
            Full meeting transcript (speaker-labelled when diarization
            is enabled).
        job_id:
            Job identifier — used to scope pgvector embeddings.
        on_stage:
            Optional async callback ``(stage_name, detail)`` invoked
            at each pipeline stage transition.  The worker uses this
            to write granular progress to PostgreSQL.
        """
        async def _notify(stage: str, detail: str) -> None:
            if on_stage is not None:
                await on_stage(stage, detail)

        token_count = count_tokens(transcript)
        self._logger.info(
            "extraction_start",
            transcript_tokens=token_count,
            threshold=self._settings.hierarchical_token_threshold,
        )

        # --- Short-transcript fast path ---
        if not needs_hierarchical_summarization(
            transcript, self._settings.hierarchical_token_threshold
        ):
            self._logger.info("direct_extraction", reason="transcript within threshold")
            await _notify("extracting", "Extracting artifacts (direct)…")
            return await self._provider.extract_artifacts(transcript)

        # --- Long-transcript intelligence pipeline ---
        compressor = None
        if self._settings.compression_enabled:
            from .compression import ContextCompressor
            compressor = ContextCompressor(settings=self._settings)

        from .rag_retrieval import build_index, retrieve_all_artifact_contexts

        # Branch A: Hierarchical summarization (Map-Reduce)
        # Branch B: RAG transcript indexing (chunk → embed → store)
        # Both run concurrently via asyncio.gather.
        await _notify("summarizing", "Summarizing transcript (Map-Reduce)…")
        condensed_task = self._hierarchical_summarize(transcript)

        if self._provider.is_mock:
            condensed = await condensed_task
            self._logger.info(
                "hierarchical_summarization_complete",
                original_tokens=token_count,
                condensed_tokens=count_tokens(condensed),
                rag="skipped_mock",
            )
            if compressor is not None:
                await _notify("compressing", "Compressing context…")
                result = compressor.compress(condensed)
                self._logger.info(
                    "context_compression_applied",
                    original_tokens=result.original_tokens,
                    compressed_tokens=result.compressed_tokens,
                    ratio=round(result.compression_ratio, 3),
                    segments_filtered=result.segments_filtered,
                )
                condensed = result.text
            await _notify("extracting", "Extracting structured artifacts…")
            return await self._provider.extract_artifacts(condensed)

        index_task = build_index(
            transcript,
            settings=self._settings,
            job_id=job_id,
            session_factory=self._get_session_factory(),
        )
        condensed, rag_index = await asyncio.gather(condensed_task, index_task)

        self._logger.info(
            "hierarchical_summarization_complete",
            original_tokens=token_count,
            condensed_tokens=count_tokens(condensed),
            rag_chunks=len(rag_index.chunks),
        )

        # Context compression.
        if compressor is not None:
            await _notify("compressing", "Compressing context…")
            result = compressor.compress(condensed)
            self._logger.info(
                "context_compression_applied",
                original_tokens=result.original_tokens,
                compressed_tokens=result.compressed_tokens,
                ratio=round(result.compression_ratio, 3),
                segments_filtered=result.segments_filtered,
            )
            condensed = result.text

        # Per-artifact RAG retrieval.
        await _notify("retrieving", "Retrieving relevant transcript segments…")
        artifact_contexts = await retrieve_all_artifact_contexts(
            rag_index, settings=self._settings,
        )
        rag_context_block = "\n\n".join(
            f"### {atype.replace('_', ' ').title()}\n{ctx}"
            for atype, ctx in artifact_contexts.items()
            if ctx
        )

        await _notify("extracting", "Extracting structured artifacts…")

        if rag_context_block:
            augmented_input = RAG_AUGMENTED_EXTRACTION_CONTEXT.format(
                rag_context=rag_context_block,
                condensed_summary=condensed,
            )
            self._logger.info(
                "rag_augmented_extraction",
                rag_context_tokens=count_tokens(rag_context_block),
                augmented_input_tokens=count_tokens(augmented_input),
            )
            return await self._provider.extract_artifacts(augmented_input)

        # Fallback: no RAG context retrieved (e.g. embedding failure).
        self._logger.warning("rag_no_context_retrieved", fallback="condensed_only")
        return await self._provider.extract_artifacts(condensed)

    # ------------------------------------------------------------------
    # Map phase
    # ------------------------------------------------------------------

    async def _summarize_chunk(
        self,
        chunk_text: str,
        chunk_index: int,
        total_chunks: int,
        semaphore: asyncio.Semaphore,
    ) -> str:
        """Summarise a single chunk behind a concurrency semaphore."""
        async with semaphore:
            prompt = CHUNK_SUMMARIZATION_PROMPT.format(
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                chunk_text=chunk_text,
            )
            self._logger.info(
                "chunk_summarization_start",
                chunk=chunk_index,
                total=total_chunks,
                tokens=count_tokens(chunk_text),
            )
            summary = await self._provider.generate_text(prompt)
            self._logger.info(
                "chunk_summarization_done",
                chunk=chunk_index,
                summary_tokens=count_tokens(summary),
            )
            return summary

    # ------------------------------------------------------------------
    # Reduce phase
    # ------------------------------------------------------------------

    async def _merge_summaries(self, summaries: list[str]) -> str:
        """Merge a list of summaries into a single condensed text."""
        combined = "\n\n---\n\n".join(
            f"[Segment {i + 1}]\n{s}" for i, s in enumerate(summaries)
        )
        prompt = MERGE_SUMMARIZATION_PROMPT.format(
            num_summaries=len(summaries),
            combined_summaries=combined,
        )
        self._logger.info(
            "merge_start",
            num_summaries=len(summaries),
            combined_tokens=count_tokens(combined),
        )
        merged = await self._provider.generate_text(prompt)
        self._logger.info(
            "merge_done", merged_tokens=count_tokens(merged)
        )
        return merged

    # ------------------------------------------------------------------
    # Full hierarchical pipeline
    # ------------------------------------------------------------------

    async def _hierarchical_summarize(self, transcript: str) -> str:
        """Run the complete Map → Reduce pipeline and return condensed text."""
        settings = self._settings

        # --- Map ---
        chunks = chunk_transcript(
            transcript,
            max_chunk_tokens=settings.hierarchical_chunk_max_tokens,
            overlap_tokens=settings.hierarchical_chunk_overlap_tokens,
        )
        self._logger.info("chunking_done", num_chunks=len(chunks))

        semaphore = asyncio.Semaphore(settings.hierarchical_concurrency_limit)
        tasks = [
            self._summarize_chunk(chunk, idx + 1, len(chunks), semaphore)
            for idx, chunk in enumerate(chunks)
        ]
        summaries: list[str] = await asyncio.gather(*tasks)

        # --- Reduce (recursive) ---
        merged_text = "\n\n".join(summaries)
        reduce_level = 1
        while count_tokens(merged_text) > settings.hierarchical_max_summary_tokens:
            self._logger.info(
                "reduce_pass",
                level=reduce_level,
                tokens=count_tokens(merged_text),
                threshold=settings.hierarchical_max_summary_tokens,
            )
            # Re-chunk the merged summaries and summarize again.
            reduction_chunks = chunk_transcript(
                merged_text,
                max_chunk_tokens=settings.hierarchical_chunk_max_tokens,
                overlap_tokens=settings.hierarchical_chunk_overlap_tokens,
            )
            if len(reduction_chunks) <= 1:
                # Cannot reduce further — single chunk already.
                break

            sem = asyncio.Semaphore(settings.hierarchical_concurrency_limit)
            reduce_tasks = [
                self._summarize_chunk(rc, idx + 1, len(reduction_chunks), sem)
                for idx, rc in enumerate(reduction_chunks)
            ]
            reduction_summaries = await asyncio.gather(*reduce_tasks)
            merged_text = await self._merge_summaries(reduction_summaries)
            reduce_level += 1

        return merged_text
