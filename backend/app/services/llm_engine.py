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

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pydantic import ValidationError

from ..config import get_settings, Settings
from ..core.prompts import build_extraction_prompt
from ..models import MeetingArtifacts
from ..models.artifacts import (
    LLMExtractionResponse,
    strip_markdown_fencing,
    to_meeting_artifacts,
    validate_llm_response,
)

logger = logging.getLogger(__name__)

# Timeout for a single LLM API call (seconds).
_LLM_CALL_TIMEOUT = 60

# Transient errors worth retrying.
_LLM_RETRYABLE = (ConnectionError, TimeoutError, asyncio.TimeoutError)


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

    async def generate_text(self, prompt: str) -> str:
        """Send a free-form prompt and return the raw text response.

        Used by the hierarchical summarization pipeline for chunk
        summarisation and merge steps.  Subclasses that cannot perform
        raw generation (e.g. mocks) should return an empty string or
        raise ``NotImplementedError``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support generate_text"
        )

    def _build_extraction_prompt(self, transcript: str) -> str:
        """Build the artifact extraction prompt using the template system."""
        return build_extraction_prompt(transcript)

    def _parse_extraction(self, data: dict, transcript: str) -> MeetingArtifacts:
        """Parse LLM JSON response into Pydantic models via the validation layer."""
        validated = LLMExtractionResponse.model_validate(data)
        return to_meeting_artifacts(validated, transcript)
    
    def _clean_json_response(self, text: str) -> str:
        """Remove markdown code blocks from LLM response."""
        return strip_markdown_fencing(text)


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider for artifact extraction."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite"):
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required for GeminiProvider. "
                "Set TEST_MODE=true to use mock extraction instead."
            )

        self.api_key = api_key
        self.model_name = model
        self._model = None
        self._genai = None
        logger.info("GeminiProvider initialized (model=%s)", model)

    @property
    def provider_name(self) -> str:
        return "Google Gemini"

    @property
    def is_mock(self) -> bool:
        return False

    def _get_genai(self):
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
        if self._model is None:
            genai = self._get_genai()
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        return self._model
    
    @retry(
        retry=retry_if_exception_type(_LLM_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _call_gemini(
        self,
        prompt: str,
        *,
        temperature: float = 0.1,
        max_output_tokens: int = 4096,
    ) -> str:
        """Call Gemini with retry and timeout; return raw response text."""
        model = self._get_model()
        genai = self._get_genai()
        async with asyncio.timeout(_LLM_CALL_TIMEOUT):
            response = await asyncio.to_thread(
                model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )
        if not response or not response.text:
            raise RuntimeError("Gemini API returned empty response")
        return response.text

    async def extract_artifacts(self, transcript: str) -> MeetingArtifacts:
        """Extract artifacts using Gemini with validation and retry.

        Attempts extraction at temperature 0.1. If the LLM response
        fails JSON parsing or Pydantic validation, retries once at
        temperature 0.0 for maximum determinism.
        """
        prompt = build_extraction_prompt(transcript)
        temperatures = [0.1, 0.0]
        last_error: Exception | None = None

        for attempt, temp in enumerate(temperatures, 1):
            try:
                logger.info(
                    "Gemini extraction attempt %d (temperature=%.1f)",
                    attempt, temp,
                )
                raw = await self._call_gemini(prompt, temperature=temp)
                validated = validate_llm_response(raw)
                artifacts = to_meeting_artifacts(validated, transcript)
                logger.info("Gemini extraction succeeded on attempt %d", attempt)
                return artifacts
            except (ValueError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "Extraction validation failed (attempt %d): %s",
                    attempt, exc,
                )
            except Exception as exc:
                logger.error("Gemini extraction failed: %s", exc)
                raise RuntimeError(
                    f"Failed to extract artifacts: {exc}"
                ) from exc

        raise RuntimeError(
            f"LLM response validation failed after "
            f"{len(temperatures)} attempts: {last_error}"
        ) from last_error

    async def generate_text(self, prompt: str) -> str:
        """Send a free-form prompt to Gemini and return the raw text."""
        return await self._call_gemini(prompt, max_output_tokens=4096)


class OpenAIProvider(LLMProvider):
    """OpenAI Chat Completions LLM provider for artifact extraction."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for OpenAIProvider. "
                "Set TEST_MODE=true to use mock extraction instead."
            )

        self.api_key = api_key
        self.model = model
        self._client = None
        logger.info("OpenAIProvider initialized (model=%s)", model)

    @property
    def provider_name(self) -> str:
        return "OpenAI"

    @property
    def is_mock(self) -> bool:
        return False

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            except ImportError as e:
                raise RuntimeError(
                    "openai is not installed. "
                    "Install it with: pip install openai"
                ) from e
        return self._client
    
    @retry(
        retry=retry_if_exception_type(_LLM_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def _call_openai(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """Call OpenAI with retry and timeout; return raw response text."""
        client = self._get_client()
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        async with asyncio.timeout(_LLM_CALL_TIMEOUT):
            response = await asyncio.to_thread(
                client.chat.completions.create, **kwargs
            )
        if not response.choices or not response.choices[0].message.content:
            raise RuntimeError("OpenAI API returned empty response")
        return response.choices[0].message.content

    async def extract_artifacts(self, transcript: str) -> MeetingArtifacts:
        """Extract artifacts using OpenAI with validation and retry.

        Attempts extraction at temperature 0.1 with JSON mode. If the
        response fails validation, retries at temperature 0.0.
        """
        prompt = build_extraction_prompt(transcript)
        system_msg = (
            "You are an elite Agile Scrum Master and Meeting Analyst. "
            "Always respond with valid JSON only, no markdown formatting."
        )
        temperatures = [0.1, 0.0]
        last_error: Exception | None = None

        for attempt, temp in enumerate(temperatures, 1):
            try:
                logger.info(
                    "OpenAI extraction attempt %d (%s, temperature=%.1f)",
                    attempt, self.model, temp,
                )
                messages = [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ]
                raw = await self._call_openai(
                    messages, temperature=temp, json_mode=True,
                )
                validated = validate_llm_response(raw)
                artifacts = to_meeting_artifacts(validated, transcript)
                logger.info("OpenAI extraction succeeded on attempt %d", attempt)
                return artifacts
            except (ValueError, ValidationError) as exc:
                last_error = exc
                logger.warning(
                    "Extraction validation failed (attempt %d): %s",
                    attempt, exc,
                )
            except Exception as exc:
                logger.error("OpenAI extraction failed: %s", exc)
                raise RuntimeError(
                    f"Failed to extract artifacts: {exc}"
                ) from exc

        raise RuntimeError(
            f"LLM response validation failed after "
            f"{len(temperatures)} attempts: {last_error}"
        ) from last_error

    async def generate_text(self, prompt: str) -> str:
        """Send a free-form prompt to OpenAI and return the raw text."""
        messages = [
            {"role": "system", "content": "You are a senior technical meeting analyst."},
            {"role": "user", "content": prompt},
        ]
        return await self._call_openai(messages)


def _mock_provider():
    from .mock_services import MockExtractor
    return MockExtractor(simulated_delay=0.3)


def get_llm_provider(settings: Settings | None = None) -> LLMProvider | "MockExtractor":  # type: ignore[name-defined]
    """Return the configured LLM provider, falling back to a mock when no key is set."""
    if settings is None:
        settings = get_settings()

    if settings.test_mode:
        logger.info("TEST_MODE enabled: using MockExtractor")
        return _mock_provider()

    provider = settings.llm_provider.lower()

    if provider == "openai":
        if not settings.openai_api_key:
            logger.warning("OPENAI_API_KEY is not set; falling back to MockExtractor")
            return _mock_provider()
        try:
            return OpenAIProvider(
                api_key=settings.openai_api_key,
                model=settings.openai_model,
            )
        except Exception as exc:
            logger.error("Failed to initialise OpenAIProvider: %s", exc)
            return _mock_provider()

    # Default: Gemini
    if not settings.gemini_api_key:
        logger.warning("GEMINI_API_KEY is not set; falling back to MockExtractor")
        return _mock_provider()
    try:
        return GeminiProvider(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )
    except Exception as exc:
        logger.error("Failed to initialise GeminiProvider: %s", exc)
        return _mock_provider()


__all__ = [
    "LLMProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "get_llm_provider",
]
