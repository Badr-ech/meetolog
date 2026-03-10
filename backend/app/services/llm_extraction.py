"""
LLM-based semantic extraction service using Google Gemini API.
Extracts structured Agile artifacts from meeting transcripts.

Implements the LLMExtractor interface for dependency injection.
"""

import asyncio
import json
from datetime import datetime

import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..interfaces import LLMExtractor
from ..models import (
    ActionableTask,
    MeetingArtifacts,
    UserStory,
    Task,
    Decision,
    Blocker,
    ActionItem,
    Priority,
)
from .heuristics import calculate_artifact_confidence

logger = structlog.get_logger(__name__)

# Timeout for a single LLM API call (seconds).
LLM_CALL_TIMEOUT_SECONDS = 60

# Transient errors worth retrying.
_LLM_RETRYABLE = (ConnectionError, TimeoutError, asyncio.TimeoutError)

# Lazy import google.generativeai to allow mock mode without API key
_genai = None


def _get_genai():
    global _genai
    if _genai is None:
        try:
            import google.generativeai as genai
            _genai = genai
        except ImportError as e:
            logger.error(f"Failed to import google.generativeai: {e}")
            raise RuntimeError(
                "google-generativeai is not installed. Install it with: pip install google-generativeai\n"
                "Or set TEST_MODE=true to use mock extraction."
            ) from e
    return _genai


class GeminiExtractor(LLMExtractor):
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for GeminiExtractor. "
                "Set TEST_MODE=true to use mock extraction instead."
            )
        
        self.api_key = api_key
        self._model = None
        logger.info("GeminiExtractor initialized")
    
    @property
    def is_mock(self) -> bool:
        """This is a real implementation."""
        return False
    
    def _get_model(self):
        if self._model is None:
            genai = _get_genai()
            logger.info("gemini_model_init")
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel("gemini-2.5-flash-lite")
            logger.info("gemini_model_ready")
        return self._model
    
    def _build_extraction_prompt(self, transcript: str) -> str:
        return f"""You are an expert Agile Project Manager assistant analyzing a meeting transcript.
Your task is to extract all relevant Agile artifacts from the transcript and return them as strictly valid JSON.

TRANSCRIPT:
---
{transcript}
---

INSTRUCTIONS:
1. Identify all USER STORIES mentioned (look for "As a...", feature requests, user needs, or requirements)
2. Identify all TASKS (specific work items, assignments, to-dos, action items assigned to people)
3. Identify all DECISIONS made (agreements, choices, determinations, conclusions reached)
4. Identify all BLOCKERS (impediments, dependencies, things preventing progress, issues raised)
5. Identify all ACTION ITEMS (follow-ups that don't fit other categories)
6. Extract participant names mentioned in the transcript
7. Estimate story points for user stories using Fibonacci sequence (1, 2, 3, 5, 8, 13)
8. Assign priorities based on context and urgency (low, medium, high, critical)
9. Generate a concise 2-3 sentence summary of the meeting
10. For each artifact, include a 'confidence_score' between 0.0 and 1.0 representing how explicitly this item was discussed in the meeting.

CRITICAL: Return ONLY valid JSON with no additional text, markdown formatting, or explanation.
The response must be parseable by json.loads() directly.

Required JSON structure:
{{
    "meeting_title": "string - infer an appropriate title from the meeting content",
    "summary": "string - 2-3 sentence summary of the meeting",
    "participants": ["list of participant names mentioned"],
    "user_stories": [
        {{
            "title": "string - brief descriptive title",
            "as_a": "user role",
            "i_want": "desired action or feature",
            "so_that": "benefit or value",
            "acceptance_criteria": ["list of acceptance criteria if mentioned"],
            "priority": "low|medium|high|critical",
            "story_points": null or number (1, 2, 3, 5, 8, 13),
            "confidence_score": 0.0 to 1.0 or null
        }}
    ],
    "tasks": [
        {{
            "title": "string - brief task description",
            "description": "string - detailed description",
            "assignee": "name or null if not assigned",
            "priority": "low|medium|high|critical",
            "due_date": "string or null if not mentioned",
            "confidence_score": 0.0 to 1.0 or null
        }}
    ],
    "decisions": [
        {{
            "title": "string - decision summary",
            "description": "string - full decision details",
            "made_by": "name or null",
            "rationale": "string - reason for the decision",
            "confidence_score": 0.0 to 1.0 or null
        }}
    ],
    "blockers": [
        {{
            "title": "string - blocker summary",
            "description": "string - details about the blocker",
            "affected_tasks": ["list of affected task titles"],
            "owner": "name responsible for resolving or null",
            "resolution_plan": "string - proposed solution if discussed",
            "confidence_score": 0.0 to 1.0 or null
        }}
    ],
    "action_items": [
        {{
            "description": "string - what needs to be done",
            "assignee": "name or null",
            "due_date": "string or null",
            "confidence_score": 0.0 to 1.0 or null
        }}
    ],
    "execution_tasks": [
        {{
            "title": "string - concise task title",
            "description": "string - detailed description of work required",
            "owner_role": "string - responsible role (Engineering, Design, Product, DevOps, QA) or specific name",
            "priority": "High|Medium|Low",
            "task_source": "Explicit|Inferred",
            "dependencies": ["list of other tasks or conditions this depends on"],
            "confidence_score": 0.0 to 1.0 or null
        }}
    ]
}}

### Task Extraction & Inference Protocol
You must analyze the transcript to generate a list of 'execution_tasks'.
1.  **Explicit Tasks:** Identify clearly stated action items (e.g., "John will update the API").
2.  **Inferred Tasks:** Deduce necessary work based on Decisions or Blockers.
    *   *Example:* If a decision is "Switch to Postgres," infer a task: "Provision Postgres instance" (Role: DevOps).
    *   *Example:* If a blocker is "Missing UI designs," infer a task: "Finalize UI Mocks" (Role: Design).
3.  **Schema Enforcement:**
    *   Assign a logical `owner_role` based on the task nature if a person isn't named.
    *   Set `task_source` to 'Explicit' or 'Inferred' accordingly.
    *   **Anti-Hallucination:** Do not invent tasks for features not discussed. Only infer steps necessary to achieve the meeting's stated goals.

If no items exist for a category, return an empty array [].
Now analyze the transcript and return the JSON:"""

    async def extract_artifacts(self, transcript: str) -> MeetingArtifacts:
        """
        Extract Agile artifacts from a meeting transcript using Gemini.
        
        Args:
            transcript: The meeting transcript text
            
        Returns:
            MeetingArtifacts with all extracted information
            
        Raises:
            RuntimeError: If the API call fails or response cannot be parsed
        """
        model = self._get_model()
        prompt = self._build_extraction_prompt(transcript)
        
        try:
            logger.info("llm_extraction_start")
            
            genai = _get_genai()
            
            response = await self._call_llm_with_retry(model, prompt, genai)
            
            if not response or not response.text:
                raise RuntimeError("Gemini API returned empty response")
            
            # Parse the JSON response
            json_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if json_text.startswith("```"):
                lines = json_text.split("\n")
                # Remove first line (```json) and last line (```)
                json_text = "\n".join(lines[1:-1]) if len(lines) > 2 else json_text
                json_text = json_text.strip()
            
            try:
                extracted = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response as JSON: {e}")
                logger.error(f"Raw response: {json_text[:500]}...")
                raise RuntimeError(f"Failed to parse LLM response as JSON: {e}") from e
            
            logger.info("llm_extraction_success")
            return self._parse_extraction(extracted, transcript)
            
        except Exception as e:
            logger.error("llm_extraction_failed", error=str(e))
            raise RuntimeError(f"Failed to extract artifacts: {e}") from e
    
    @retry(
        retry=retry_if_exception_type(_LLM_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _call_llm_with_retry(self, model, prompt: str, genai):
        """Call the Gemini API with retry and timeout protection."""
        async with asyncio.timeout(LLM_CALL_TIMEOUT_SECONDS):
            return await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=4096,
                ),
            )
    
    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _safe_confidence(raw: object) -> float | None:
        """Coerce an LLM-provided confidence value to a float or None."""
        if raw is None:
            return None
        try:
            val = float(raw)
            if 0.0 <= val <= 1.0:
                return round(val, 2)
            return None  # out of range → treat as missing
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _inject_scores(items: list[dict], artifact_type: str) -> None:
        """Mutate *items* in-place, filling missing confidence_score via heuristic."""
        for item in items:
            parsed = GeminiExtractor._safe_confidence(item.get("confidence_score"))
            if parsed is not None:
                item["confidence_score"] = parsed
            else:
                item["confidence_score"] = calculate_artifact_confidence(item, artifact_type)

    def _parse_extraction(self, data: dict, transcript: str) -> MeetingArtifacts:
        """Parse the LLM JSON response into Pydantic models."""
        
        def parse_priority(p: str) -> Priority:
            mapping = {
                "low": Priority.LOW,
                "medium": Priority.MEDIUM,
                "high": Priority.HIGH,
                "critical": Priority.CRITICAL
            }
            return mapping.get(p.lower() if p else "medium", Priority.MEDIUM)

        # --- Inject heuristic scores where the LLM omitted them ----------
        raw_stories = data.get("user_stories", [])
        raw_tasks = data.get("tasks", [])
        raw_decisions = data.get("decisions", [])
        raw_blockers = data.get("blockers", [])
        raw_action_items = data.get("action_items", [])
        raw_execution = data.get("execution_tasks", [])

        self._inject_scores(raw_stories, "user_story")
        self._inject_scores(raw_tasks, "task")
        self._inject_scores(raw_decisions, "decision")
        self._inject_scores(raw_blockers, "blocker")
        self._inject_scores(raw_action_items, "action_item")
        self._inject_scores(raw_execution, "execution_task")

        user_stories = [
            UserStory(
                title=s.get("title", ""),
                as_a=s.get("as_a", ""),
                i_want=s.get("i_want", ""),
                so_that=s.get("so_that", ""),
                acceptance_criteria=s.get("acceptance_criteria", []),
                priority=parse_priority(s.get("priority", "medium")),
                story_points=s.get("story_points"),
                confidence_score=s.get("confidence_score"),
            )
            for s in raw_stories
        ]
        
        tasks = [
            Task(
                title=t.get("title", ""),
                description=t.get("description", ""),
                assignee=t.get("assignee"),
                priority=parse_priority(t.get("priority", "medium")),
                due_date=t.get("due_date"),
                confidence_score=t.get("confidence_score"),
            )
            for t in raw_tasks
        ]
        
        decisions = [
            Decision(
                title=d.get("title", ""),
                description=d.get("description", ""),
                made_by=d.get("made_by"),
                rationale=d.get("rationale", ""),
                confidence_score=d.get("confidence_score"),
            )
            for d in raw_decisions
        ]
        
        blockers = [
            Blocker(
                title=b.get("title", ""),
                description=b.get("description", ""),
                affected_tasks=b.get("affected_tasks", []),
                owner=b.get("owner"),
                resolution_plan=b.get("resolution_plan", ""),
                confidence_score=b.get("confidence_score"),
            )
            for b in raw_blockers
        ]
        
        action_items = [
            ActionItem(
                description=a.get("description", ""),
                assignee=a.get("assignee"),
                due_date=a.get("due_date"),
                confidence_score=a.get("confidence_score"),
            )
            for a in raw_action_items
        ]
        
        execution_tasks = [
            ActionableTask(
                title=et.get("title", ""),
                description=et.get("description", ""),
                owner_role=et.get("owner_role", "Engineering"),
                priority=et.get("priority", "Medium"),
                task_source=et.get("task_source", "Explicit"),
                dependencies=et.get("dependencies", []),
                confidence_score=et.get("confidence_score"),
            )
            for et in raw_execution
        ]
        
        return MeetingArtifacts(
            meeting_title=data.get("meeting_title", "Meeting"),
            meeting_date=datetime.now(),
            participants=data.get("participants", []),
            summary=data.get("summary", ""),
            user_stories=user_stories,
            tasks=tasks,
            decisions=decisions,
            blockers=blockers,
            action_items=action_items,
            execution_tasks=execution_tasks,
            transcript=transcript,
        )
