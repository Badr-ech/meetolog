"""
Context Compression — semantic filtering and chunk prioritization.

Reduces transcript and summary token footprint before the final LLM
extraction pass, cutting API cost and latency while strictly preserving
high-density semantic information (decisions, tasks, blockers, named
entities, deadlines, and technical specifics).

Compression Pipeline
--------------------
1. **Segment Splitting** — input text is split into logical segments
   (sentences, speaker turns, bullet points).
2. **Filler Filtering** — segments matching conversational filler,
   pleasantries, and off-topic banter patterns are removed.
3. **Semantic Scoring** — each surviving segment is scored on a
   weighted feature set: actionable verbs, decision language,
   temporal markers, named-entity signals, quantitative data, and
   assignment patterns.
4. **Budget-Constrained Selection** — segments are ranked by score
   and selected in original order until the configurable token budget
   is filled.

No external models or additional LLM calls are required.  The entire
compression pass runs in-process on CPU in O(n) time relative to
segment count.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

from ..config import Settings, get_settings
from ..utils.text_chunking import count_tokens

logger = structlog.get_logger(__name__)

# ── Filler patterns (compiled once) ──────────────────────────────

_FILLER_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in (
        # Greetings / closings
        r"^\s*(hi|hello|hey|good morning|good afternoon|good evening|thanks everyone|thank you all|bye|goodbye|see you|take care)\b",
        # Meeting logistics filler
        r"^\s*(can everyone hear me|is everyone here|let me share my screen|one moment|hold on|sorry.{0,15}(muted|mic)|you['']?re on mute)",
        # Verbal fillers
        r"^\s*(um+|uh+|ah+|hmm+|so+|okay so|right so|yeah so|well)\s*[,.]?\s*$",
        # Pure acknowledgements (standalone)
        r"^\s*(okay|ok|sure|right|got it|sounds good|makes sense|exactly|absolutely|definitely|perfect|great|alright|yep|yeah|yes|no worries)\s*[.!]?\s*$",
        # Laughter / filler reactions
        r"^\s*\[?(laughs?|laughter|chuckles?|coughs?|inaudible|crosstalk)\]?\s*$",
    )
]

# ── Scoring features ─────────────────────────────────────────────

# Actionable verbs indicating concrete work items or decisions.
_ACTION_VERBS = re.compile(
    r"\b(implement|deploy|migrate|refactor|fix|resolve|assign|create|build|"
    r"design|configure|test|review|approve|ship|release|integrate|update|"
    r"schedule|prioriti[sz]e|investigate|escalate|document|merge|revert|"
    r"provision|launch|deprecate|remove|add|write|deliver)\b",
    re.IGNORECASE,
)

# Decision and agreement language.
_DECISION_LANGUAGE = re.compile(
    r"\b(decided|agreed|confirmed|approved|concluded|resolved|"
    r"we['']?ll go with|we['']?ll use|we will|the decision is|"
    r"final answer|consensus|signed off|rationale|chosen|selected|"
    r"ruling|determination|verdict|we chose|let['']?s go with)\b",
    re.IGNORECASE,
)

# Temporal markers — deadlines, dates, sprint references.
_TEMPORAL_MARKERS = re.compile(
    r"\b(deadline|due date|by (monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|tomorrow|end of (week|day|sprint|month|quarter))|"
    r"next (week|sprint|month|quarter)|this (week|sprint)|"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+\d{1,2}|"
    r"\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?|sprint \d+|q[1-4]\b|"
    r"eta|timeline|target date|ship date)\b",
    re.IGNORECASE,
)

# Assignment and ownership patterns.
_ASSIGNMENT_PATTERNS = re.compile(
    r"\b(assigned to|owner|responsible|"
    r"(I|you|he|she|they|we|[A-Z][a-z]+)\s+(will|should|can|must|need[s]? to)|"
    r"action item|follow[- ]?up|take[s]? (care|ownership|this)|"
    r"[A-Z][a-z]+['']?s (task|job|responsibility))\b",
    re.IGNORECASE,
)

# Blocker and impediment language.
_BLOCKER_LANGUAGE = re.compile(
    r"\b(block(ed|er|ing)?|impediment|depend(s|ency|encies)|"
    r"waiting (on|for)|stuck|can['']?t proceed|risk|"
    r"prevent(s|ing|ed)?|issue|problem|concern|delay(ed|ing)?)\b",
    re.IGNORECASE,
)

# Quantitative data — numbers, percentages, story points.
_QUANTITATIVE = re.compile(
    r"\b(\d+(\.\d+)?\s*(%|percent|points?|hours?|days?|GB|MB|ms|"
    r"requests?/s|tokens?|users?|sprint[s]?|story[- ]?points?))\b",
    re.IGNORECASE,
)

# Named-entity heuristic — capitalised multi-word tokens or @mentions.
_NAMED_ENTITY = re.compile(
    r"(?:(?<!\.\s)(?<![.!?]\s)\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b)|(?:@\w+)",
)

# Feature weights (sum-based scoring).
_FEATURE_WEIGHTS: dict[str, tuple[re.Pattern[str], float]] = {
    "action_verbs": (_ACTION_VERBS, 2.0),
    "decision_language": (_DECISION_LANGUAGE, 3.0),
    "temporal_markers": (_TEMPORAL_MARKERS, 2.5),
    "assignment_patterns": (_ASSIGNMENT_PATTERNS, 2.5),
    "blocker_language": (_BLOCKER_LANGUAGE, 2.5),
    "quantitative": (_QUANTITATIVE, 1.5),
    "named_entity": (_NAMED_ENTITY, 1.0),
}


# ── Data structures ──────────────────────────────────────────────

@dataclass
class _ScoredSegment:
    """Internal representation of a scored text segment."""
    index: int
    text: str
    tokens: int
    score: float


@dataclass
class CompressionResult:
    """Output of a compression pass with diagnostic metrics."""
    text: str
    original_tokens: int
    compressed_tokens: int
    segments_total: int
    segments_kept: int
    segments_filtered: int
    compression_ratio: float = field(init=False)

    def __post_init__(self) -> None:
        if self.original_tokens > 0:
            self.compression_ratio = self.compressed_tokens / self.original_tokens
        else:
            self.compression_ratio = 1.0


# ── Core compressor ──────────────────────────────────────────────

class ContextCompressor:
    """Semantic filtering and chunk prioritization compressor.

    Accepts raw text or a list of pre-chunked strings and returns
    a compressed string that fits within a configurable token budget,
    preserving high-value semantic content in original order.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    # ── public API ────────────────────────────────────────────────

    def compress(
        self,
        text: str,
        *,
        target_token_budget: int | None = None,
    ) -> CompressionResult:
        """Compress *text* to fit within *target_token_budget* tokens.

        Parameters
        ----------
        text:
            Raw transcript, condensed summary, or any text block.
        target_token_budget:
            Maximum token count for the output.  Defaults to
            ``settings.compression_target_budget_tokens``.

        Returns
        -------
        CompressionResult
            Compressed text with diagnostic metrics.
        """
        if target_token_budget is None:
            target_token_budget = self._settings.compression_target_budget_tokens

        original_tokens = count_tokens(text)

        if original_tokens <= target_token_budget:
            return CompressionResult(
                text=text,
                original_tokens=original_tokens,
                compressed_tokens=original_tokens,
                segments_total=0,
                segments_kept=0,
                segments_filtered=0,
            )

        segments = self._split_segments(text)
        scored, filtered_count = self._filter_and_score(segments)
        selected = self._select_within_budget(scored, target_token_budget)

        compressed_text = "\n".join(seg.text for seg in selected)
        compressed_tokens = count_tokens(compressed_text)

        return CompressionResult(
            text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            segments_total=len(segments),
            segments_kept=len(selected),
            segments_filtered=filtered_count,
        )

    def compress_chunks(
        self,
        chunks: list[str],
        *,
        target_token_budget: int | None = None,
    ) -> CompressionResult:
        """Compress a list of text chunks into a single compressed string.

        Convenience wrapper that joins chunks and delegates to
        :meth:`compress`.
        """
        combined = "\n\n".join(chunks)
        return self.compress(combined, target_token_budget=target_token_budget)

    # ── segmentation ──────────────────────────────────────────────

    @staticmethod
    def _split_segments(text: str) -> list[str]:
        """Split text into logical segments for scoring.

        Splits on:
        - Double newlines (paragraph/speaker-turn boundaries)
        - Single newlines when the line begins with a bullet, dash, or
          speaker label (``Speaker N:`` / ``SPEAKER_XX:``)

        Single-sentence lines within a paragraph are kept together to
        avoid breaking coherent thoughts.
        """
        # First split on double newlines.
        raw_blocks = re.split(r"\n{2,}", text.strip())

        segments: list[str] = []
        for block in raw_blocks:
            block = block.strip()
            if not block:
                continue
            # Within a block, split on lines that begin with a bullet,
            # numbered list item, or speaker label.
            sub_lines = re.split(
                r"\n(?=\s*[-•*]\s|^\s*\d+[.)]\s|^\s*(?:Speaker|SPEAKER)\s*\w+\s*:)",
                block,
                flags=re.MULTILINE,
            )
            for line in sub_lines:
                stripped = line.strip()
                if stripped:
                    segments.append(stripped)

        return segments

    # ── filtering & scoring ───────────────────────────────────────

    @staticmethod
    def _is_filler(segment: str) -> bool:
        """Return True if *segment* matches a conversational filler pattern."""
        return any(pat.search(segment) for pat in _FILLER_PATTERNS)

    @staticmethod
    def _score_segment(segment: str) -> float:
        """Compute a semantic density score for *segment*.

        The score is the weighted sum of feature-match counts.  A
        minimum baseline of 0.1 is added for every non-filler segment
        to avoid dropping content that is simply low on keywords but
        may still carry contextual value.
        """
        score = 0.1  # baseline for surviving filler filter
        for _name, (pattern, weight) in _FEATURE_WEIGHTS.items():
            matches = pattern.findall(segment)
            score += len(matches) * weight
        return score

    def _filter_and_score(
        self, segments: list[str],
    ) -> tuple[list[_ScoredSegment], int]:
        """Filter filler and score remaining segments.

        Returns
        -------
        tuple
            (scored segments in original order, count of filtered segments)
        """
        scored: list[_ScoredSegment] = []
        filtered_count = 0

        for idx, seg in enumerate(segments):
            if self._is_filler(seg):
                filtered_count += 1
                continue
            score = self._score_segment(seg)
            scored.append(_ScoredSegment(
                index=idx,
                text=seg,
                tokens=count_tokens(seg),
                score=score,
            ))

        return scored, filtered_count

    # ── budget-constrained selection ──────────────────────────────

    @staticmethod
    def _select_within_budget(
        scored: list[_ScoredSegment],
        token_budget: int,
    ) -> list[_ScoredSegment]:
        """Select highest-scoring segments that fit within *token_budget*.

        Segments are ranked by score (descending) for selection, then
        returned in their original order to preserve narrative flow.
        """
        if not scored:
            return []

        # Rank by score descending; use original index as tiebreaker
        # to prefer earlier content when scores are equal.
        ranked = sorted(scored, key=lambda s: (-s.score, s.index))

        selected_indices: set[int] = set()
        running_tokens = 0

        for seg in ranked:
            if running_tokens + seg.tokens > token_budget:
                continue
            selected_indices.add(seg.index)
            running_tokens += seg.tokens

        # Restore original order.
        return [s for s in scored if s.index in selected_indices]
