"""
Deterministic heuristic scoring for extracted artifacts.

When the LLM omits a confidence_score (or returns an unparseable value),
this module computes a fallback score based on artifact completeness,
linguistic signals, and field-specific weights.

Score range: 0.0 – 1.0 (clamped).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import MeetingArtifacts

# ---------------------------------------------------------------------------
# Strong action verbs that increase confidence
# ---------------------------------------------------------------------------
_ACTION_VERBS = re.compile(
    r"\b("
    r"implement|deploy|create|fix|build|design|configure|set up|"
    r"develop|test|migrate|update|integrate|refactor|provision|"
    r"document|schedule|review|approve|ship|release|launch|"
    r"write|optimise|optimize|resolve|deliver|install|automate"
    r")\b",
    re.IGNORECASE,
)

# Ambiguous / hedging phrases that decrease confidence
_AMBIGUOUS_PHRASES = re.compile(
    r"\b("
    r"maybe|might|probably|perhaps|someone|could|possibly|"
    r"not sure|unclear|TBD|to be determined|eventually|sometime"
    r")\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Per-type field expectations
# ---------------------------------------------------------------------------

# Each entry maps a field name to ``True`` if the field is an *owner-like*
# attribute (used to award the owner bonus).
_FIELD_SPECS: dict[str, dict[str, bool]] = {
    "user_story": {
        "title": False,
        "as_a": False,
        "i_want": False,
        "so_that": False,
        "acceptance_criteria": False,
        "priority": True,
    },
    "task": {
        "title": False,
        "description": False,
        "assignee": True,
        "priority": True,
    },
    "decision": {
        "title": False,
        "description": False,
        "made_by": True,
        "rationale": False,
    },
    "blocker": {
        "title": False,
        "description": False,
        "owner": True,
        "resolution_plan": False,
    },
    "action_item": {
        "description": False,
        "assignee": True,
    },
    "execution_task": {
        "title": False,
        "description": False,
        "owner_role": True,
        "priority": True,
        "dependencies": False,
    },
}


def _field_populated(value: object) -> bool:
    """Return True if a value is meaningfully present (non-None, non-empty)."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return len(value) > 0
    return True


def calculate_artifact_confidence(
    artifact_dict: dict,
    artifact_type: str,
) -> float:
    """
    Calculate a deterministic confidence score for an artifact dictionary.

    Parameters
    ----------
    artifact_dict:
        Raw dict representation of the artifact (pre-Pydantic validation).
    artifact_type:
        One of ``"user_story"``, ``"task"``, ``"decision"``, ``"blocker"``,
        ``"action_item"``, ``"execution_task"``.

    Returns
    -------
    float
        Confidence score clamped to [0.0, 1.0].
    """
    score = 0.2  # base

    spec = _FIELD_SPECS.get(artifact_type, {})

    # --- +0.2 owner / assignee present ---
    has_owner = any(
        _field_populated(artifact_dict.get(field))
        for field, is_owner in spec.items()
        if is_owner
    )
    if has_owner:
        score += 0.2

    # --- +0.2 priority explicitly set ---
    priority_val = artifact_dict.get("priority")
    if _field_populated(priority_val):
        score += 0.2

    # --- +0.2 strong action verb in title/description ---
    text_parts = " ".join(
        str(artifact_dict.get(f, ""))
        for f in ("title", "description", "i_want")
        if artifact_dict.get(f)
    )
    if _ACTION_VERBS.search(text_parts):
        score += 0.2

    # --- -0.2 ambiguous phrases ---
    if _AMBIGUOUS_PHRASES.search(text_parts):
        score -= 0.2

    # --- +0.2 all expected fields populated ---
    all_populated = all(
        _field_populated(artifact_dict.get(field))
        for field in spec
    )
    if all_populated:
        score += 0.2

    return max(0.0, min(1.0, round(score, 2)))


def backfill_confidence_scores(artifacts: MeetingArtifacts) -> None:
    """
    Mutate *artifacts* in-place, filling ``confidence_score`` via heuristic
    for every artifact where the field is ``None``.

    Safe to call on artifacts that already carry scores (they are preserved).
    """
    _sections: list[tuple[str, str]] = [
        ("user_stories", "user_story"),
        ("tasks", "task"),
        ("decisions", "decision"),
        ("blockers", "blocker"),
        ("action_items", "action_item"),
        ("execution_tasks", "execution_task"),
    ]

    for attr, art_type in _sections:
        for item in getattr(artifacts, attr, []):
            if item.confidence_score is None:
                item.confidence_score = calculate_artifact_confidence(
                    item.model_dump(), art_type
                )
