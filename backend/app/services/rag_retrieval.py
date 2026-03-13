"""
RAG Transcript Retrieval — semantic search over diarized transcripts.

Builds an ephemeral in-memory vector index from transcript chunks and
exposes a ``retrieve_context`` method that returns the top-K most
relevant segments for a given query.  The index uses NumPy cosine
similarity — no external vector database required — and is destroyed
automatically when the worker's ``TemporaryDirectory`` context exits.

Embedding Provider Selection
----------------------------
* ``LLM_PROVIDER=gemini``  → Google ``text-embedding-004``
* ``LLM_PROVIDER=openai``  → OpenAI ``text-embedding-3-small``

Both providers are accessed through the same async interface so the
rest of the pipeline is provider-agnostic.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
import structlog

from ..config import get_settings, Settings
from ..utils.text_chunking import chunk_transcript, count_tokens

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import async_sessionmaker, AsyncSession
    from .transcript_index import PgVectorIndex

logger = structlog.get_logger(__name__)

# ── Artifact-type retrieval queries ───────────────────────────────
# Each key corresponds to a field in MeetingArtifacts; the value is a
# search query designed to surface the most relevant transcript segments
# for that artifact category.
ARTIFACT_RETRIEVAL_QUERIES: dict[str, str] = {
    "user_stories": (
        "feature requests, user needs, requirements, acceptance criteria, "
        "story points, as a user I want"
    ),
    "tasks": (
        "task assignments, work items, to-dos, who is responsible, deadlines, "
        "action items assigned to specific people"
    ),
    "decisions": (
        "decisions made, agreements reached, choices, determinations, "
        "conclusions, rationale for decisions"
    ),
    "blockers": (
        "blockers, impediments, dependencies, things preventing progress, "
        "issues raised, resolution plans"
    ),
    "action_items": (
        "follow-up actions, next steps, things to do after the meeting"
    ),
    "execution_tasks": (
        "concrete execution tasks, engineering work, design work, devops tasks, "
        "inferred tasks from decisions and blockers, owner roles, dependencies"
    ),
    "ideas": (
        "ideas, suggestions, proposals, brainstorming, exploratory concepts, "
        "potential improvements, innovative approaches, things worth exploring"
    ),
}


# ── Embedding helpers ─────────────────────────────────────────────

async def _embed_openai(texts: list[str], api_key: str) -> np.ndarray:
    """Generate embeddings via OpenAI ``text-embedding-3-small``."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def _call() -> list[list[float]]:
        resp = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in resp.data]

    vectors = await asyncio.to_thread(_call)
    return np.array(vectors, dtype=np.float32)


async def _embed_gemini(texts: list[str], api_key: str) -> np.ndarray:
    """Generate embeddings via Google ``text-embedding-004``."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)

    def _call() -> list[list[float]]:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=texts,
            task_type="RETRIEVAL_DOCUMENT",
        )
        return result["embedding"]

    vectors = await asyncio.to_thread(_call)
    # Gemini returns list[list[float]] when given a list input.
    return np.array(vectors, dtype=np.float32)


async def _embed_texts(
    texts: list[str],
    settings: Settings,
    *,
    task_type: str = "document",
) -> np.ndarray:
    """Route to the configured embedding provider with batching.

    Parameters
    ----------
    texts:
        Strings to embed.
    settings:
        Application settings (provider, API keys).
    task_type:
        Hint for embedding models that distinguish retrieval roles.
        ``"document"`` for indexing, ``"query"`` for search queries.

    Returns
    -------
    np.ndarray
        Shape ``(len(texts), embedding_dim)``.
    """
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    provider = settings.llm_provider.lower()
    batch_size = settings.rag_embedding_batch_size

    all_vectors: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        if provider == "openai":
            vecs = await _embed_openai(batch, settings.openai_api_key)
        else:
            vecs = await _embed_gemini(batch, settings.gemini_api_key)
        all_vectors.append(vecs)

    return np.vstack(all_vectors)


# ── TranscriptIndex ───────────────────────────────────────────────

@dataclass
class TranscriptIndex:
    """Ephemeral in-memory vector index over transcript chunks.

    Holds the chunk texts alongside their embedding vectors and performs
    cosine-similarity retrieval.  Intended to be constructed via
    :func:`build_index` and discarded when the enclosing temp directory
    is garbage-collected.
    """

    chunks: list[str] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.empty((0, 0), dtype=np.float32))

    # ── retrieval ─────────────────────────────────────────────────

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        settings: Settings | None = None,
    ) -> list[str]:
        """Return the *top_k* most relevant chunks for *query*.

        Parameters
        ----------
        query:
            Natural-language search query.
        top_k:
            Number of results.  Defaults to ``settings.rag_top_k``.
        settings:
            Application settings (used for embedding + defaults).
        """
        if settings is None:
            settings = get_settings()
        if top_k is None:
            top_k = settings.rag_top_k
        if self.vectors.size == 0:
            return []

        query_vec = await _embed_texts([query], settings, task_type="query")
        # Cosine similarity: dot(a, b) / (||a|| * ||b||)
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # avoid division by zero
        normed = self.vectors / norms

        q_norm = np.linalg.norm(query_vec)
        if q_norm == 0:
            return []
        q_normed = query_vec / q_norm

        similarities = (normed @ q_normed.T).squeeze()
        if similarities.ndim == 0:
            similarities = similarities.reshape(1)

        k = min(top_k, len(self.chunks))
        top_indices = np.argpartition(-similarities, k)[:k]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        return [self.chunks[i] for i in top_indices]


async def build_index(
    transcript: str,
    settings: Settings | None = None,
    *,
    job_id: UUID | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> TranscriptIndex | PgVectorIndex:
    """Chunk *transcript*, embed all chunks, and return a queryable index.

    When ``settings.rag_storage_backend`` is ``"pgvector"`` and both
    *job_id* and *session_factory* are provided, embeddings are persisted
    to PostgreSQL via :class:`~.transcript_index.PgVectorIndex`.  Otherwise
    the default ephemeral NumPy-backed :class:`TranscriptIndex` is used.

    Parameters
    ----------
    transcript:
        Full meeting transcript.
    settings:
        Application settings (chunk sizes, embedding config, backend).
    job_id:
        Job identifier — required for pgvector scoping.
    session_factory:
        Async SQLAlchemy session factory — required for pgvector.
    """
    if settings is None:
        settings = get_settings()

    # ── pgvector persistent path ──────────────────────────────────
    if settings.rag_storage_backend == "pgvector":
        if job_id is not None and session_factory is not None:
            from .transcript_index import PgVectorIndex

            return await PgVectorIndex.build(
                transcript, job_id, session_factory, settings,
            )
        logger.warning(
            "pgvector_fallback_memory",
            reason="job_id or session_factory not provided",
        )

    # ── in-memory NumPy path (default) ────────────────────────────
    chunks = chunk_transcript(
        transcript,
        max_chunk_tokens=settings.rag_chunk_max_tokens,
        overlap_tokens=settings.rag_chunk_overlap_tokens,
    )
    if not chunks:
        logger.warning("rag_build_index_empty", reason="no chunks produced")
        return TranscriptIndex()

    logger.info(
        "rag_build_index_start",
        num_chunks=len(chunks),
        total_tokens=count_tokens(transcript),
        backend="memory",
    )
    vectors = await _embed_texts(chunks, settings, task_type="document")
    logger.info(
        "rag_build_index_done",
        num_chunks=len(chunks),
        embedding_dim=vectors.shape[1] if vectors.ndim == 2 else 0,
        backend="memory",
    )
    return TranscriptIndex(chunks=chunks, vectors=vectors)


# ── High-level retrieval ──────────────────────────────────────────

async def retrieve_context(
    index: TranscriptIndex,
    query: str,
    top_k: int | None = None,
    max_context_tokens: int | None = None,
    settings: Settings | None = None,
) -> str:
    """Retrieve and format relevant transcript segments for *query*.

    Concatenates retrieved chunks (separated by ``---``) and truncates
    the combined context to *max_context_tokens* to avoid exceeding the
    LLM's context window.

    Parameters
    ----------
    index:
        Pre-built :class:`TranscriptIndex`.
    query:
        Natural-language query describing the target artifact category.
    top_k:
        Number of chunks to retrieve.  Defaults to ``settings.rag_top_k``.
    max_context_tokens:
        Token budget for the returned context block.  Defaults to
        ``settings.rag_max_context_tokens``.
    settings:
        Application settings.

    Returns
    -------
    str
        Formatted retrieval context ready for prompt injection.
        Empty string if no relevant chunks are found.
    """
    if settings is None:
        settings = get_settings()
    if max_context_tokens is None:
        max_context_tokens = settings.rag_max_context_tokens

    segments = await index.retrieve(query, top_k=top_k, settings=settings)
    if not segments:
        return ""

    # Assemble and respect token budget.
    lines: list[str] = []
    running_tokens = 0
    for i, seg in enumerate(segments):
        seg_tokens = count_tokens(seg)
        if running_tokens + seg_tokens > max_context_tokens and lines:
            break
        lines.append(f"[Retrieved Segment {i + 1}]\n{seg}")
        running_tokens += seg_tokens

    return "\n\n---\n\n".join(lines)


async def retrieve_all_artifact_contexts(
    index: TranscriptIndex,
    settings: Settings | None = None,
) -> dict[str, str]:
    """Run retrieval for every artifact category in parallel.

    Returns a mapping from artifact type name to formatted context string.
    Categories that yield no relevant segments map to empty strings.
    """
    if settings is None:
        settings = get_settings()

    async def _fetch(artifact_type: str, query: str) -> tuple[str, str]:
        ctx = await retrieve_context(index, query, settings=settings)
        return artifact_type, ctx

    tasks = [
        _fetch(atype, query)
        for atype, query in ARTIFACT_RETRIEVAL_QUERIES.items()
    ]
    results = await asyncio.gather(*tasks)
    return dict(results)
