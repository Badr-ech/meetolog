"""
LLM Engine Abstraction Layer.

Provides a unified interface for multiple LLM providers
(Google Gemini, OpenAI GPT-4) using the Strategy Pattern with
a factory function for runtime provider selection.
"""

from __future__ import annotations

import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Literal

from ..config import get_settings, Settings
from ..models import (
    MeetingArtifacts,
    UserStory,
    Task,
    Decision,
    Blocker,
    ActionItem,
    Priority,
)

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        ...
    
    @property
    @abstractmethod
    def is_mock(self) -> bool:
        ...
    
    @abstractmethod
    async def extract_artifacts(self, transcript: str) -> MeetingArtifacts:
        """
        Extract Agile artifacts from a meeting transcript.
        
        Args:
            transcript: The meeting transcript text
            
        Returns:
            MeetingArtifacts with all extracted information
            
        Raises:
            RuntimeError: If extraction fails
        """
        ...
    
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
            "story_points": null or number (1, 2, 3, 5, 8, 13)
        }}
    ],
    "tasks": [
        {{
            "title": "string - brief task description",
            "description": "string - detailed description",
            "assignee": "name or null if not assigned",
            "priority": "low|medium|high|critical",
            "due_date": "string or null if not mentioned"
        }}
    ],
    "decisions": [
        {{
            "title": "string - decision summary",
            "description": "string - full decision details",
            "made_by": "name or null",
            "rationale": "string - reason for the decision"
        }}
    ],
    "blockers": [
        {{
            "title": "string - blocker summary",
            "description": "string - details about the blocker",
            "affected_tasks": ["list of affected task titles"],
            "owner": "name responsible for resolving or null",
            "resolution_plan": "string - proposed solution if discussed"
        }}
    ],
    "action_items": [
        {{
            "description": "string - what needs to be done",
            "assignee": "name or null",
            "due_date": "string or null"
        }}
    ]
}}

If no items exist for a category, return an empty array [].
Now analyze the transcript and return the JSON:"""

    def _parse_extraction(self, data: dict, transcript: str) -> MeetingArtifacts:
        """Parse LLM JSON response into Pydantic models."""
        
        def parse_priority(p: str | None) -> Priority:
            if not p:
                return Priority.MEDIUM
            mapping = {
                "low": Priority.LOW,
                "medium": Priority.MEDIUM,
                "high": Priority.HIGH,
                "critical": Priority.CRITICAL
            }
            return mapping.get(p.lower(), Priority.MEDIUM)
        
        user_stories = [
            UserStory(
                title=s.get("title", ""),
                as_a=s.get("as_a", ""),
                i_want=s.get("i_want", ""),
                so_that=s.get("so_that", ""),
                acceptance_criteria=s.get("acceptance_criteria", []),
                priority=parse_priority(s.get("priority")),
                story_points=s.get("story_points"),
            )
            for s in data.get("user_stories", [])
        ]
        
        tasks = [
            Task(
                title=t.get("title", ""),
                description=t.get("description", ""),
                assignee=t.get("assignee"),
                priority=parse_priority(t.get("priority")),
                due_date=t.get("due_date"),
            )
            for t in data.get("tasks", [])
        ]
        
        decisions = [
            Decision(
                title=d.get("title", ""),
                description=d.get("description", ""),
                made_by=d.get("made_by"),
                rationale=d.get("rationale", ""),
            )
            for d in data.get("decisions", [])
        ]
        
        blockers = [
            Blocker(
                title=b.get("title", ""),
                description=b.get("description", ""),
                affected_tasks=b.get("affected_tasks", []),
                owner=b.get("owner"),
                resolution_plan=b.get("resolution_plan", ""),
            )
            for b in data.get("blockers", [])
        ]
        
        action_items = [
            ActionItem(
                description=a.get("description", ""),
                assignee=a.get("assignee"),
                due_date=a.get("due_date"),
            )
            for a in data.get("action_items", [])
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
            transcript=transcript,
        )
    
    def _clean_json_response(self, text: str) -> str:
        """Remove markdown code blocks from LLM response."""
        text = text.strip()
        
        # Remove markdown code blocks if present
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first line (```json) and last line (```)
            if len(lines) > 2:
                text = "\n".join(lines[1:-1])
            text = text.strip()
        
        return text


class GeminiProvider(LLMProvider):
    """
    Google Gemini LLM provider for artifact extraction.
    
    Uses the Gemini 2.5 Flash Lite model for fast, cost-effective
    extraction with good quality.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Gemini provider.
        
        Args:
            api_key: Google Gemini API key
            
        Raises:
            ValueError: If API key is empty
        """
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for GeminiProvider. "
                "Set TEST_MODE=true to use mock extraction instead."
            )
        
        self.api_key = api_key
        self._model = None
        self._genai = None
        logger.info("GeminiProvider initialized")
    
    @property
    def provider_name(self) -> str:
        return "Google Gemini"
    
    @property
    def is_mock(self) -> bool:
        return False
    
    def _get_genai(self):
        """Lazy load google.generativeai module."""
        if self._genai is None:
            try:
                import google.generativeai as genai
                self._genai = genai
            except ImportError as e:
                raise RuntimeError(
                    "google-generativeai is not installed. "
                    "Install it with: pip install google-generativeai"
                ) from e
        return self._genai
    
    def _get_model(self):
        """Initialize and return the Gemini model."""
        if self._model is None:
            genai = self._get_genai()
            logger.info("Initializing Gemini model")
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel("gemini-2.5-flash-lite")
            logger.info("Gemini model initialized successfully")
        return self._model
    
    async def extract_artifacts(self, transcript: str) -> MeetingArtifacts:
        """Extract artifacts using Gemini API."""
        model = self._get_model()
        prompt = self._build_extraction_prompt(transcript)
        
        try:
            logger.info("Calling Gemini API for artifact extraction")
            
            genai = self._get_genai()
            
            # Call Gemini API in thread pool to not block async
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.2,
                    max_output_tokens=4096,
                )
            )
            
            if not response or not response.text:
                raise RuntimeError("Gemini API returned empty response")
            
            # Parse the JSON response
            json_text = self._clean_json_response(response.text)
            
            try:
                extracted = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response as JSON: {e}")
                logger.error(f"Raw response: {json_text[:500]}...")
                raise RuntimeError(f"Failed to parse LLM response as JSON: {e}") from e
            
            logger.info("Successfully extracted artifacts via Gemini")
            return self._parse_extraction(extracted, transcript)
            
        except Exception as e:
            logger.error(f"Gemini extraction failed: {e}")
            raise RuntimeError(f"Failed to extract artifacts: {e}") from e


class OpenAIProvider(LLMProvider):
    """
    OpenAI GPT-4 LLM provider for artifact extraction.
    
    Uses GPT-4o-mini for cost-effective extraction with excellent quality.
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """
        Initialize OpenAI provider.
        
        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
            
        Raises:
            ValueError: If API key is empty
        """
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for OpenAIProvider. "
                "Set TEST_MODE=true to use mock extraction instead."
            )
        
        self.api_key = api_key
        self.model = model
        self._client = None
        logger.info(f"OpenAIProvider initialized with model: {model}")
    
    @property
    def provider_name(self) -> str:
        return "OpenAI"
    
    @property
    def is_mock(self) -> bool:
        return False
    
    def _get_client(self):
        """Lazy load OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
                logger.info("OpenAI client initialized")
            except ImportError as e:
                raise RuntimeError(
                    "openai is not installed. "
                    "Install it with: pip install openai"
                ) from e
        return self._client
    
    async def extract_artifacts(self, transcript: str) -> MeetingArtifacts:
        """Extract artifacts using OpenAI API."""
        client = self._get_client()
        prompt = self._build_extraction_prompt(transcript)
        
        try:
            logger.info(f"Calling OpenAI API ({self.model}) for artifact extraction")
            
            # Call OpenAI API in thread pool to not block async
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert Agile Project Manager assistant. Always respond with valid JSON only, no markdown formatting."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=4096,
                response_format={"type": "json_object"},
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise RuntimeError("OpenAI API returned empty response")
            
            json_text = response.choices[0].message.content
            
            try:
                extracted = json.loads(json_text)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse OpenAI response as JSON: {e}")
                logger.error(f"Raw response: {json_text[:500]}...")
                raise RuntimeError(f"Failed to parse LLM response as JSON: {e}") from e
            
            logger.info("Successfully extracted artifacts via OpenAI")
            return self._parse_extraction(extracted, transcript)
            
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            raise RuntimeError(f"Failed to extract artifacts: {e}") from e


def get_llm_provider(settings: Settings | None = None) -> LLMProvider | "MockExtractor":  # type: ignore[name-defined]
    """
    Return the configured LLM provider based on settings.

    Falls back to MockExtractor when no API key is available.
    """
    if settings is None:
        settings = get_settings()
    
    # Test mode - use mock
    if settings.test_mode:
        from .mock_services import MockExtractor
        logger.info("TEST_MODE enabled: Using MockExtractor")
        return MockExtractor(simulated_delay=0.3)
    
    provider = settings.llm_provider.lower()
    
    # OpenAI provider
    if provider == "openai":
        if not settings.openai_api_key:
            from .mock_services import MockExtractor
            logger.warning(
                "OPENAI_API_KEY is not set. "
                "Using MockExtractor. Set the API key for real extraction."
            )
            return MockExtractor(simulated_delay=0.3)
        
        try:
            return OpenAIProvider(api_key=settings.openai_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize OpenAIProvider: {e}")
            from .mock_services import MockExtractor
            return MockExtractor(simulated_delay=0.3)
    
    # Gemini provider (default)
    if not settings.gemini_api_key:
        from .mock_services import MockExtractor
        logger.warning(
            "GEMINI_API_KEY is not set. "
            "Using MockExtractor. Set the API key for real extraction."
        )
        return MockExtractor(simulated_delay=0.3)
    
    try:
        return GeminiProvider(api_key=settings.gemini_api_key)
    except Exception as e:
        logger.error(f"Failed to initialize GeminiProvider: {e}")
        from .mock_services import MockExtractor
        return MockExtractor(simulated_delay=0.3)


# For backwards compatibility with LLMExtractor interface
# The LLMProvider class provides the same interface
__all__ = [
    "LLMProvider",
    "GeminiProvider", 
    "OpenAIProvider",
    "get_llm_provider",
]
