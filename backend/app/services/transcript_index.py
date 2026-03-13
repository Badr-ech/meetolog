"""
Persistent vector index backed by PostgreSQL + pgvector.

Stores transcript chunk embeddings in a PostgreSQL table using the
pgvector extension's ``vector`` type and ``<=>`` cosine distance
operator.  Unlike the ephemeral NumPy-based ``TranscriptIndex``, a
``PgVectorIndex`` persists across worker restarts and can be queried
from any process with database access.

Table Structure
---------------
The ``transcript_embeddings`` table is created lazily on first use::

    id          UUID PRIMARY KEY
    job_id      UUID NOT NULL
    chunk_index INTEGER NOT NULL
    chunk_text  TEXT NOT NULL
    embedding   vector NOT NULL
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()

Requires the ``vector`` extension on the PostgreSQL server::

    CREATE EXTENSION IF NOT EXISTS vector;
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from uuid import UUID

import numpy as np
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from ..config import Settings, get_settings
from ..utils.text_chunking import chunk_transcript, count_tokens

logger = structlog.get_logger(__name__)

# Module-level flag to avoid repeated DDL checks within a single process.
_TABLE_ENSURED = False


async def _ensure_pgvector_table(session: AsyncSession) -> None:
    """Create the pgvector extension and embeddings table if absent."""
    global _TABLE_ENSURED
    if _TABLE_ENSURED:
        return

    await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    await session.execute(text("""
        CREATE TABLE IF NOT EXISTS transcript_embeddings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            job_id UUID NOT NULL,
            chunk_index INTEGER NOT NULL,
            chunk_text TEXT NOT NULL,
            embedding vector NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """))
    await session.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_transcript_embeddings_job_id "
        "ON transcript_embeddings (job_id)"
    ))
    await session.commit()
    _TABLE_ENSURED = True
    logger.info("pgvector_table_ensured")


def _ndarray_to_pgvector(arr: np.ndarray) -> str:
    """Format a 1-D NumPy array as a pgvector literal ``[1.0,2.0,…]``."""
    return "[" + ",".join(f"{v:.8f}" for v in arr.flat) + "]"


@dataclass
class PgVectorIndex:
    """Persistent vector index backed by PostgreSQL + pgvector.

    Constructed via the :meth:`build` classmethod. The ``retrieve``
    method uses pgvector's native ``<=>`` cosine distance operator for
    ranked nearest-neighbour search, scoped to a single ``job_id``.
    """

    job_id: UUID
    chunks: list[str] = field(default_factory=list)
    _session_factory: async_sessionmaker[AsyncSession] | None = field(
        default=None, repr=False,
    )

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    async def build(
        cls,
        transcript: str,
        job_id: UUID,
        session_factory: async_sessionmaker[AsyncSession],
        settings: Settings | None = None,
    ) -> PgVectorIndex:
        """Chunk, embed, and persist the transcript vectors.

        Parameters
        ----------
        transcript:
            Full meeting transcript.
        job_id:
            Job identifier — used to scope the stored embeddings.
        session_factory:
            Async SQLAlchemy session factory for database access.
        settings:
            Application settings (chunk sizes, embedding config).

        Returns
        -------
        PgVectorIndex
            Ready-to-query index backed by the persisted embeddings.
        """
        if settings is None:
            settings = get_settings()

        chunks = chunk_transcript(
            transcript,
            max_chunk_tokens=settings.rag_chunk_max_tokens,
            overlap_tokens=settings.rag_chunk_overlap_tokens,
        )
        if not chunks:
            logger.warning("pgvector_build_empty", reason="no chunks produced")
            return cls(job_id=job_id, _session_factory=session_factory)

        # Import embedding function from rag_retrieval (single source of truth).
        from .rag_retrieval import _embed_texts

        logger.info(
            "pgvector_build_start",
            job_id=str(job_id),
            num_chunks=len(chunks),
        )

        vectors = await _embed_texts(chunks, settings, task_type="document")

        async with session_factory() as session:
            await _ensure_pgvector_table(session)

            # Delete existing embeddings for this job (idempotent rebuild).
            await session.execute(
                text("DELETE FROM transcript_embeddings WHERE job_id = :job_id"),
                {"job_id": str(job_id)},
            )

            # Batch insert chunks + embeddings.
            for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                await session.execute(
                    text("""
                        INSERT INTO transcript_embeddings
                            (id, job_id, chunk_index, chunk_text, embedding)
                        VALUES
                            (:id, :job_id, :idx, :chunk_text, :embedding::vector)
                    """),
                    {
                        "id": str(uuid.uuid4()),
                        "job_id": str(job_id),
                        "idx": i,
                        "chunk_text": chunk,
                        "embedding": _ndarray_to_pgvector(vec),
                    },
                )

            await session.commit()

        logger.info(
            "pgvector_build_done",
            job_id=str(job_id),
            num_chunks=len(chunks),
            embedding_dim=vectors.shape[1] if vectors.ndim == 2 else 0,
        )
        return cls(
            job_id=job_id,
            chunks=chunks,
            _session_factory=session_factory,
        )

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        settings: Settings | None = None,
    ) -> list[str]:
        """Return the *top_k* most relevant chunks via cosine distance.

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
        if not self.chunks or self._session_factory is None:
            return []

        from .rag_retrieval import _embed_texts

        query_vec = await _embed_texts([query], settings, task_type="query")
        if query_vec.size == 0:
            return []

        vec_str = _ndarray_to_pgvector(query_vec[0])

        async with self._session_factory() as session:
            result = await session.execute(
                text("""
                    SELECT chunk_text
                    FROM transcript_embeddings
                    WHERE job_id = :job_id
                    ORDER BY embedding <=> :query_vec::vector
                    LIMIT :top_k
                """),
                {
                    "job_id": str(self.job_id),
                    "query_vec": vec_str,
                    "top_k": top_k,
                },
            )
            rows = result.fetchall()

        return [row[0] for row in rows]
