"""Tests for app.services.compression — filler detection, scoring, and budget selection."""

import pytest
from unittest.mock import patch, MagicMock

from app.services.compression import (
    CompressionResult,
    ContextCompressor,
    _FILLER_PATTERNS,
)


# ---------------------------------------------------------------------------
# Helper to create a compressor with a mock settings object
# ---------------------------------------------------------------------------

def _make_compressor(budget: int = 500) -> ContextCompressor:
    settings = MagicMock()
    settings.compression_target_budget_tokens = budget
    return ContextCompressor(settings=settings)


# ---------------------------------------------------------------------------
# CompressionResult
# ---------------------------------------------------------------------------

class TestCompressionResult:
    def test_ratio_computed(self):
        result = CompressionResult(
            text="out",
            original_tokens=100,
            compressed_tokens=40,
            segments_total=10,
            segments_kept=4,
            segments_filtered=6,
        )
        assert result.compression_ratio == pytest.approx(0.4)

    def test_ratio_when_original_zero(self):
        result = CompressionResult(
            text="",
            original_tokens=0,
            compressed_tokens=0,
            segments_total=0,
            segments_kept=0,
            segments_filtered=0,
        )
        assert result.compression_ratio == 1.0


# ---------------------------------------------------------------------------
# Filler Detection
# ---------------------------------------------------------------------------

class TestFillerDetection:
    @pytest.mark.parametrize(
        "segment",
        [
            "Hi everyone",
            "Good morning",
            "Thanks everyone",
            "Can everyone hear me?",
            "Let me share my screen",
            "um",
            "uh",
            "hmm",
            "Okay",
            "Sure",
            "Got it",
            "Sounds good",
            "Makes sense",
            "[laughs]",
            "[laughter]",
            "[inaudible]",
            "you're on mute",
            "sorry I was muted",
        ],
    )
    def test_filler_segments_detected(self, segment):
        compressor = _make_compressor()
        assert compressor._is_filler(segment), f"Expected filler: {segment!r}"

    @pytest.mark.parametrize(
        "segment",
        [
            "We decided to use PostgreSQL for the job queue.",
            "Mike will implement the auth API by Friday.",
            "The deadline is end of sprint.",
            "Blocked on the email service configuration.",
            "Deploy the service to staging by Monday.",
        ],
    )
    def test_meaningful_segments_not_filler(self, segment):
        compressor = _make_compressor()
        assert not compressor._is_filler(segment), f"Unexpected filler: {segment!r}"


# ---------------------------------------------------------------------------
# Segment Scoring
# ---------------------------------------------------------------------------

class TestSegmentScoring:
    def test_baseline_score_for_plain_text(self):
        # Non-filler segment with no feature matches should get baseline 0.1
        score = ContextCompressor._score_segment("This is a plain sentence.")
        assert score >= 0.1

    def test_decision_language_scores_higher(self):
        plain = ContextCompressor._score_segment("A plain sentence about things.")
        decision = ContextCompressor._score_segment("We decided to deploy to production.")
        assert decision > plain

    def test_action_verbs_add_score(self):
        score = ContextCompressor._score_segment("We need to implement and deploy the service.")
        # "implement" and "deploy" should each add weight
        assert score > 0.1 + 2.0  # baseline + at least one action verb

    def test_temporal_markers_add_score(self):
        score = ContextCompressor._score_segment("The deadline is by end of sprint.")
        assert score > 0.1

    def test_assignment_patterns_add_score(self):
        score = ContextCompressor._score_segment("Mike will handle the backend implementation.")
        assert score > 0.1

    def test_blocker_language_adds_score(self):
        score = ContextCompressor._score_segment("We are blocked on the dependency update.")
        assert score > 0.1

    def test_quantitative_data_adds_score(self):
        score = ContextCompressor._score_segment("The API handles 1000 requests/s with 50ms latency.")
        assert score > 0.1


# ---------------------------------------------------------------------------
# Compress — end-to-end
# ---------------------------------------------------------------------------

class TestCompress:
    def test_short_text_passes_through(self):
        compressor = _make_compressor(budget=5000)
        result = compressor.compress("Short text.")
        assert result.text == "Short text."
        assert result.original_tokens == result.compressed_tokens
        assert result.compression_ratio == 1.0

    def test_filler_segments_removed(self):
        # Budget must be *smaller* than input tokens to trigger compression;
        # compress() early-returns unchanged text when within budget.
        compressor = _make_compressor(budget=20)
        text = (
            "Hi everyone.\n\n"
            "Good morning.\n\n"
            "We decided to use PostgreSQL for the job queue.\n\n"
            "Sounds good.\n\n"
            "Mike will implement the API by Friday."
        )
        result = compressor.compress(text)
        assert result.segments_filtered > 0
        assert result.segments_kept > 0

    def test_budget_constrains_output(self):
        compressor = _make_compressor(budget=20)
        # Build a long text that will exceed 20 tokens
        segments = [f"Segment {i}: We decided to implement the feature by Friday." for i in range(20)]
        text = "\n\n".join(segments)
        result = compressor.compress(text)
        assert result.compressed_tokens <= result.original_tokens
        assert result.segments_kept < result.segments_total

    def test_empty_text(self):
        compressor = _make_compressor(budget=100)
        result = compressor.compress("")
        assert result.text == ""
        assert result.original_tokens == 0

    def test_high_value_segments_preferred(self):
        compressor = _make_compressor(budget=50)
        text = (
            "This is a mundane observation about the weather.\n\n"
            "We decided to deploy to production by Friday. Mike is the owner and the deadline is end of sprint.\n\n"
            "I had coffee this morning it was nice.\n\n"
            "The API is blocked on the dependency update from the infrastructure team."
        )
        result = compressor.compress(text)
        # Decision language and blocker language should be preferred
        assert "decided" in result.text or "blocked" in result.text


# ---------------------------------------------------------------------------
# Compress Chunks
# ---------------------------------------------------------------------------

class TestCompressChunks:
    def test_compress_chunks_joins_and_compresses(self):
        compressor = _make_compressor(budget=5000)
        chunks = ["Chunk one: we decided to use Postgres.", "Chunk two: Mike will handle deployment."]
        result = compressor.compress_chunks(chunks)
        assert "Chunk one" in result.text
        assert "Chunk two" in result.text


# ---------------------------------------------------------------------------
# Budget Selection
# ---------------------------------------------------------------------------

class TestSelectWithinBudget:
    def test_empty_segments(self):
        result = ContextCompressor._select_within_budget([], token_budget=100)
        assert result == []

    def test_preserves_original_order(self):
        from app.services.compression import _ScoredSegment
        segments = [
            _ScoredSegment(index=0, text="First", tokens=2, score=1.0),
            _ScoredSegment(index=1, text="Second", tokens=2, score=3.0),
            _ScoredSegment(index=2, text="Third", tokens=2, score=2.0),
        ]
        selected = ContextCompressor._select_within_budget(segments, token_budget=100)
        # All fit within budget, should be in original index order
        assert [s.index for s in selected] == [0, 1, 2]

    def test_budget_limits_selection(self):
        from app.services.compression import _ScoredSegment
        segments = [
            _ScoredSegment(index=0, text="Low", tokens=10, score=1.0),
            _ScoredSegment(index=1, text="High", tokens=10, score=5.0),
            _ScoredSegment(index=2, text="Med", tokens=10, score=3.0),
        ]
        # Budget for only 2 segments — should pick highest scored
        selected = ContextCompressor._select_within_budget(segments, token_budget=20)
        assert len(selected) == 2
        selected_texts = {s.text for s in selected}
        assert "High" in selected_texts
        assert "Med" in selected_texts
