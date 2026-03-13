"""
Token-aware transcript chunking for hierarchical summarization.

Splits diarized meeting transcripts into bounded chunks that respect
speaker-turn and sentence boundaries, with configurable overlap to
preserve cross-chunk context.
"""

from __future__ import annotations

import re
from typing import Sequence

import tiktoken

# Default tiktoken encoding — cl100k_base covers GPT-4 / GPT-3.5.
# Token counts are approximate for non-OpenAI models (Gemini), but
# accurate enough for chunking decisions.
_DEFAULT_ENCODING = "cl100k_base"

_encoder: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    """Return a cached tiktoken encoder instance."""
    global _encoder
    if _encoder is None:
        _encoder = tiktoken.get_encoding(_DEFAULT_ENCODING)
    return _encoder


def count_tokens(text: str) -> int:
    """Return the approximate token count for *text*."""
    return len(_get_encoder().encode(text))


def _split_into_blocks(transcript: str) -> list[str]:
    """Split a transcript into logical blocks (speaker turns / paragraphs).

    A block is defined as either:
    * A speaker turn line (``Speaker N: …`` or ``SPEAKER_XX: …``), or
    * A paragraph separated by one or more blank lines.

    Blocks are never split mid-sentence.
    """
    # Split on blank lines first.
    raw_blocks = re.split(r"\n{2,}", transcript.strip())
    blocks: list[str] = []
    for raw in raw_blocks:
        stripped = raw.strip()
        if stripped:
            blocks.append(stripped)

    if not blocks:
        # Fallback: split on single newlines.
        blocks = [ln.strip() for ln in transcript.strip().splitlines() if ln.strip()]

    return blocks


def chunk_transcript(
    transcript: str,
    max_chunk_tokens: int,
    overlap_tokens: int = 200,
) -> list[str]:
    """Split *transcript* into token-bounded chunks with overlap.

    The algorithm groups logical blocks (speaker turns / paragraphs) into
    chunks whose combined token count does not exceed *max_chunk_tokens*.
    An *overlap_tokens* tail from the previous chunk is prepended to the
    next chunk to maintain conversational context across boundaries.

    Parameters
    ----------
    transcript:
        Full meeting transcript (may include speaker labels).
    max_chunk_tokens:
        Maximum tokens per chunk (excluding overlap preamble).
    overlap_tokens:
        Number of trailing tokens from the previous chunk to repeat at
        the start of the next chunk.

    Returns
    -------
    list[str]
        Ordered list of text chunks covering the entire transcript.
    """
    blocks = _split_into_blocks(transcript)
    if not blocks:
        return [transcript] if transcript.strip() else []

    chunks: list[str] = []
    current_blocks: list[str] = []
    current_tokens = 0

    for block in blocks:
        block_tokens = count_tokens(block)

        # If a single block exceeds the limit, yield it as its own chunk.
        if block_tokens > max_chunk_tokens:
            if current_blocks:
                chunks.append("\n\n".join(current_blocks))
                current_blocks = []
                current_tokens = 0
            chunks.append(block)
            continue

        if current_tokens + block_tokens > max_chunk_tokens and current_blocks:
            chunks.append("\n\n".join(current_blocks))
            # Build overlap from the tail of the emitted chunk.
            overlap_text = _extract_overlap(current_blocks, overlap_tokens)
            current_blocks = [overlap_text] if overlap_text else []
            current_tokens = count_tokens(overlap_text) if overlap_text else 0

        current_blocks.append(block)
        current_tokens += block_tokens

    if current_blocks:
        chunks.append("\n\n".join(current_blocks))

    return chunks


def _extract_overlap(blocks: list[str], overlap_tokens: int) -> str:
    """Return the trailing portion of *blocks* totalling ≈ *overlap_tokens*."""
    if overlap_tokens <= 0:
        return ""

    collected: list[str] = []
    collected_tokens = 0

    for block in reversed(blocks):
        bt = count_tokens(block)
        if collected_tokens + bt > overlap_tokens and collected:
            break
        collected.insert(0, block)
        collected_tokens += bt

    return "\n\n".join(collected)


def needs_hierarchical_summarization(
    transcript: str,
    token_threshold: int,
) -> bool:
    """Return True if *transcript* exceeds the token threshold for direct extraction."""
    return count_tokens(transcript) > token_threshold
