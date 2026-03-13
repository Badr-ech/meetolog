"""Tests for app.core.prompts — prompt builder functions."""

from __future__ import annotations

from app.core.prompts import (
    CHUNK_SUMMARIZATION_PROMPT,
    MERGE_SUMMARIZATION_PROMPT,
    RAG_AUGMENTED_EXTRACTION_CONTEXT,
    build_extraction_prompt,
    build_task_detection_prompt,
    build_decision_detection_prompt,
    build_summarization_prompt,
)


SAMPLE_TRANSCRIPT = "Alice: We need to migrate to PostgreSQL.\nBob: I'll handle it."


class TestChunkSummarizationPrompt:
    """CHUNK_SUMMARIZATION_PROMPT template formatting."""

    def test_placeholders_formatted(self):
        result = CHUNK_SUMMARIZATION_PROMPT.format(
            chunk_index=1, total_chunks=3, chunk_text="hello"
        )
        assert "segment 1 of 3" in result
        assert "hello" in result

    def test_preserves_instructions(self):
        result = CHUNK_SUMMARIZATION_PROMPT.format(
            chunk_index=1, total_chunks=1, chunk_text=""
        )
        assert "Task assignments" in result
        assert "plain text" in result


class TestMergeSummarizationPrompt:
    """MERGE_SUMMARIZATION_PROMPT template formatting."""

    def test_placeholders_formatted(self):
        result = MERGE_SUMMARIZATION_PROMPT.format(
            num_summaries=2, combined_summaries="Summary A\nSummary B"
        )
        assert "2 sequential summaries" in result
        assert "Summary A" in result

    def test_deduplication_instruction(self):
        result = MERGE_SUMMARIZATION_PROMPT.format(
            num_summaries=1, combined_summaries=""
        )
        assert "Deduplicate" in result


class TestRagAugmentedContext:
    """RAG_AUGMENTED_EXTRACTION_CONTEXT template formatting."""

    def test_placeholders_formatted(self):
        result = RAG_AUGMENTED_EXTRACTION_CONTEXT.format(
            rag_context="<rag chunks>", condensed_summary="<summary>"
        )
        assert "<rag chunks>" in result
        assert "<summary>" in result

    def test_contains_synthesis_instructions(self):
        result = RAG_AUGMENTED_EXTRACTION_CONTEXT.format(
            rag_context="", condensed_summary=""
        )
        assert "SYNTHESIS INSTRUCTIONS" in result


class TestBuildExtractionPrompt:
    """build_extraction_prompt assembles the master extraction prompt."""

    def test_includes_transcript(self):
        prompt = build_extraction_prompt(SAMPLE_TRANSCRIPT)
        assert "migrate to PostgreSQL" in prompt

    def test_includes_role(self):
        prompt = build_extraction_prompt("x")
        assert "Agile Business Analyst" in prompt

    def test_includes_few_shot(self):
        prompt = build_extraction_prompt("x")
        assert "FEW-SHOT EXAMPLE" in prompt

    def test_includes_schema(self):
        prompt = build_extraction_prompt("x")
        assert "meeting_title" in prompt
        assert "confidence_score" in prompt

    def test_includes_instructions(self):
        prompt = build_extraction_prompt("x")
        assert "MEETING OVERVIEW" in prompt
        assert "USER STORIES" in prompt
        assert "EXECUTION TASKS" in prompt

    def test_empty_transcript(self):
        prompt = build_extraction_prompt("")
        # Should still produce a valid prompt structure
        assert "TRANSCRIPT:" in prompt


class TestBuildTaskDetectionPrompt:
    """build_task_detection_prompt assembles the task detection prompt."""

    def test_includes_transcript(self):
        prompt = build_task_detection_prompt(SAMPLE_TRANSCRIPT)
        assert "migrate to PostgreSQL" in prompt

    def test_includes_task_role(self):
        prompt = build_task_detection_prompt("x")
        assert "Task Extraction Analyst" in prompt

    def test_includes_explicit_inferred_guidance(self):
        prompt = build_task_detection_prompt("x")
        assert "Explicit" in prompt
        assert "Inferred" in prompt

    def test_includes_few_shot(self):
        prompt = build_task_detection_prompt("x")
        assert "Kafka producer adapter" in prompt

    def test_includes_schema(self):
        prompt = build_task_detection_prompt("x")
        assert "task_source" in prompt
        assert "confidence_score" in prompt


class TestBuildDecisionDetectionPrompt:
    """build_decision_detection_prompt assembles the CoT decision prompt."""

    def test_includes_transcript(self):
        prompt = build_decision_detection_prompt(SAMPLE_TRANSCRIPT)
        assert "migrate to PostgreSQL" in prompt

    def test_includes_decision_role(self):
        prompt = build_decision_detection_prompt("x")
        assert "Decision Analyst" in prompt

    def test_includes_cot_steps(self):
        prompt = build_decision_detection_prompt("x")
        assert "Step 1" in prompt
        assert "Chain-of-Thought" in prompt

    def test_includes_classification_signals(self):
        prompt = build_decision_detection_prompt("x")
        assert "confirmed" in prompt
        assert "discarded" in prompt
        assert "open" in prompt

    def test_includes_non_decisions(self):
        prompt = build_decision_detection_prompt("x")
        assert "non_decisions" in prompt


class TestBuildSummarizationPrompt:
    """build_summarization_prompt assembles the meeting summary prompt."""

    def test_includes_transcript(self):
        prompt = build_summarization_prompt(SAMPLE_TRANSCRIPT)
        assert "migrate to PostgreSQL" in prompt

    def test_includes_summary_structure(self):
        prompt = build_summarization_prompt("x")
        assert "Purpose" in prompt
        assert "Decisions Made" in prompt
        assert "Action Items" in prompt
        assert "Open Items" in prompt

    def test_word_limit_mentioned(self):
        prompt = build_summarization_prompt("x")
        assert "500 words" in prompt

    def test_analyst_role(self):
        prompt = build_summarization_prompt("x")
        assert "senior technical meeting analyst" in prompt
