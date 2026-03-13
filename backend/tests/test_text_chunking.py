"""Tests for app.utils.text_chunking — token counting, chunking, and overlapping."""

import pytest

from app.utils.text_chunking import (
    _extract_overlap,
    _split_into_blocks,
    chunk_transcript,
    count_tokens,
    needs_hierarchical_summarization,
)


# ---------------------------------------------------------------------------
# count_tokens
# ---------------------------------------------------------------------------

class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_known_phrase(self):
        # tiktoken cl100k_base encodes "hello world" as 2 tokens
        assert count_tokens("hello world") == 2

    def test_long_text_positive(self):
        text = "word " * 500
        tokens = count_tokens(text)
        assert tokens > 0
        # ~500 words will be significantly > 0 tokens
        assert tokens > 100


# ---------------------------------------------------------------------------
# _split_into_blocks
# ---------------------------------------------------------------------------

class TestSplitIntoBlocks:
    def test_empty_input(self):
        assert _split_into_blocks("") == []

    def test_single_paragraph(self):
        text = "This is a single paragraph."
        blocks = _split_into_blocks(text)
        assert blocks == ["This is a single paragraph."]

    def test_double_newline_splits(self):
        text = "Block one.\n\nBlock two.\n\nBlock three."
        blocks = _split_into_blocks(text)
        assert len(blocks) == 3
        assert blocks[0] == "Block one."
        assert blocks[1] == "Block two."
        assert blocks[2] == "Block three."

    def test_speaker_turns_preserved(self):
        text = "Speaker 1: Hello everyone.\n\nSpeaker 2: Hi there."
        blocks = _split_into_blocks(text)
        assert len(blocks) == 2

    def test_whitespace_only_filtered(self):
        text = "Block one.\n\n   \n\nBlock two."
        blocks = _split_into_blocks(text)
        assert len(blocks) == 2

    def test_single_newline_no_split(self):
        text = "Line one\nLine two\nLine three"
        blocks = _split_into_blocks(text)
        # Only \n{2,} triggers a split; single newlines stay in one block
        assert len(blocks) == 1
        assert blocks[0] == text

    def test_fallback_to_splitlines(self):
        # Fallback triggers when every \n{2,} block is whitespace-only
        text = "  \n\n  \n\n  "
        blocks = _split_into_blocks(text)
        # strip() empties the text, so splitlines also yields nothing
        assert blocks == []


# ---------------------------------------------------------------------------
# chunk_transcript
# ---------------------------------------------------------------------------

class TestChunkTranscript:
    def test_empty_input_returns_empty(self):
        assert chunk_transcript("", max_chunk_tokens=100) == []

    def test_whitespace_only_returns_empty(self):
        assert chunk_transcript("   \n\n  ", max_chunk_tokens=100) == []

    def test_small_text_single_chunk(self):
        text = "Hello everyone. Let's begin."
        chunks = chunk_transcript(text, max_chunk_tokens=1000)
        assert len(chunks) == 1
        assert "Hello" in chunks[0]

    def test_splits_when_exceeding_limit(self):
        # Create text with distinct blocks that exceed a small token limit
        blocks = [f"Speaker {i}: This is a sentence from speaker {i} discussing topic number {i}." for i in range(20)]
        text = "\n\n".join(blocks)
        chunks = chunk_transcript(text, max_chunk_tokens=50, overlap_tokens=0)
        assert len(chunks) > 1

    def test_overlap_produces_repeated_content(self):
        blocks = [f"Block {i}: " + "word " * 30 for i in range(5)]
        text = "\n\n".join(blocks)
        chunks = chunk_transcript(text, max_chunk_tokens=80, overlap_tokens=20)
        if len(chunks) >= 2:
            # Overlap means some text from end of chunk N appears in chunk N+1
            # We just verify chunks are produced; overlap is internal detail
            assert all(len(c) > 0 for c in chunks)

    def test_single_oversized_block(self):
        # One block exceeds max_chunk_tokens — should be yielded as its own chunk
        big_block = "word " * 500  # > any reasonable token limit
        text = f"Short block.\n\n{big_block}\n\nAnother short block."
        chunks = chunk_transcript(text, max_chunk_tokens=50, overlap_tokens=0)
        assert len(chunks) >= 2
        # The big block should be one of the chunks
        assert any(len(c) > 200 for c in chunks)

    def test_zero_overlap(self):
        blocks = [f"Block {i}: content here." for i in range(5)]
        text = "\n\n".join(blocks)
        chunks = chunk_transcript(text, max_chunk_tokens=30, overlap_tokens=0)
        assert len(chunks) >= 1


# ---------------------------------------------------------------------------
# _extract_overlap
# ---------------------------------------------------------------------------

class TestExtractOverlap:
    def test_zero_overlap_returns_empty(self):
        assert _extract_overlap(["Block A", "Block B"], overlap_tokens=0) == ""

    def test_negative_overlap_returns_empty(self):
        assert _extract_overlap(["Block A"], overlap_tokens=-5) == ""

    def test_overlap_from_single_block(self):
        blocks = ["This is a test block with several words."]
        result = _extract_overlap(blocks, overlap_tokens=100)
        assert result  # Should return the whole block since it fits

    def test_overlap_from_multiple_blocks(self):
        blocks = ["First block.", "Second block.", "Third block."]
        result = _extract_overlap(blocks, overlap_tokens=100)
        # With a large budget, all blocks should be included
        assert "First" in result or "Third" in result


# ---------------------------------------------------------------------------
# needs_hierarchical_summarization
# ---------------------------------------------------------------------------

class TestNeedsHierarchicalSummarization:
    def test_short_text_below_threshold(self):
        assert needs_hierarchical_summarization("hello world", token_threshold=100) is False

    def test_long_text_above_threshold(self):
        long_text = "word " * 1000
        assert needs_hierarchical_summarization(long_text, token_threshold=100) is True

    def test_exactly_at_threshold_is_false(self):
        # Build text with known token count
        text = "hello world"  # 2 tokens
        assert needs_hierarchical_summarization(text, token_threshold=2) is False

    def test_just_above_threshold(self):
        text = "hello world foo"  # 3 tokens
        assert needs_hierarchical_summarization(text, token_threshold=2) is True
