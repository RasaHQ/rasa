"""Service for handling LLM interactions."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

import openai
import structlog

from rasa.builder import config
from rasa.builder.copilot.copilot import Copilot
from rasa.builder.copilot.copilot_response_handler import CopilotResponseHandler
from rasa.builder.copilot.copilot_templated_message_provider import (
    load_copilot_internal_message_templates,
)
from rasa.builder.copilot.history_store import (
    CopilotHistoryStore,
    SQLiteCopilotHistoryStore,
)
from rasa.builder.exceptions import LLMGenerationError
from rasa.builder.guardrails.clients import (
    GuardrailsClient,
    LakeraAIGuardrails,
)
from rasa.builder.guardrails.policy_checker import GuardrailsPolicyChecker

structlogger = structlog.get_logger()


class LLMService:
    """Handles OpenAI LLM interactions with caching for efficiency."""

    def __init__(self) -> None:
        self._client: Optional[openai.AsyncOpenAI] = None
        self._domain_schema: Optional[Dict[str, Any]] = None
        self._flows_schema: Optional[Dict[str, Any]] = None
        self._copilot: Optional[Copilot] = None
        self._guardrails: Optional[GuardrailsClient] = None
        self._guardrails_policy_checker: Optional[GuardrailsPolicyChecker] = None
        self._copilot_response_handler: Optional[CopilotResponseHandler] = None
        self._copilot_internal_message_templates: Optional[Dict[str, str]] = None
        self._history_store: Optional[CopilotHistoryStore] = None

    @property
    def copilot(self) -> Copilot:
        """Get or lazy create copilot instance."""
        if self._copilot is None:
            self._copilot = Copilot()

        try:
            return self._copilot
        except Exception as e:
            structlogger.error(
                "llm_service.copilot.error",
                event_info="LLM Service: Error getting copilot instance.",
                error=str(e),
            )
            raise

    @property
    def copilot_response_handler(self) -> CopilotResponseHandler:
        """Get or lazy create copilot response handler instance."""
        if self._copilot_response_handler is None:
            self._copilot_response_handler = CopilotResponseHandler(
                rolling_buffer_size=config.COPILOT_HANDLER_ROLLING_BUFFER_SIZE,
            )
        try:
            return self._copilot_response_handler
        except Exception as e:
            structlogger.error(
                "llm_service.copilot_response_handler.error",
                event_info=(
                    "LLM Service: Error getting copilot response handler instance."
                ),
                error=str(e),
            )
            raise

    @property
    def guardrails(self) -> Optional[GuardrailsClient]:
        """Get or lazy create guardrails instance."""
        if not config.ENABLE_GUARDRAILS:
            return None
        # TODO: Replace with Open Source guardrails implementation once it's ready
        try:
            if self._guardrails is None:
                self._guardrails = LakeraAIGuardrails()
            return self._guardrails
        except Exception as e:
            structlogger.error(
                "llm_service.guardrails.error",
                event_info="LLM Service: Error getting guardrails instance.",
                error=str(e),
            )
            raise

    @property
    def guardrails_policy_checker(self) -> Optional[GuardrailsPolicyChecker]:
        """Get or lazy create guardrails policy checker instance."""
        try:
            if self._guardrails_policy_checker is None and self.guardrails is not None:
                self._guardrails_policy_checker = GuardrailsPolicyChecker(
                    self.guardrails
                )
            return self._guardrails_policy_checker
        except Exception as e:
            structlogger.error(
                "llm_service.guardrails_policy_checker.error",
                event_info=(
                    "LLM Service: Error getting guardrails policy checker instance."
                ),
                error=str(e),
            )
            raise

    @property
    def copilot_internal_message_templates(self) -> Dict[str, str]:
        """Get or lazy load copilot internal message templates."""
        if self._copilot_internal_message_templates is None:
            self._copilot_internal_message_templates = (
                load_copilot_internal_message_templates()
            )
        return self._copilot_internal_message_templates

    @property
    def history_store(self) -> CopilotHistoryStore:
        """Get or lazy create history store instance."""
        if self._history_store is None:
            database_path = config.COPILOT_HISTORY_SQLITE_PATH
            structlogger.info("llm_service.history_store.backend", path=database_path)
            self._history_store = SQLiteCopilotHistoryStore(database_path)

        try:
            return self._history_store
        except Exception as e:
            structlogger.error(
                "llm_service.history_store.error",
                event_info="LLM Service: Error getting history store instance.",
                error=str(e),
            )
            raise

    @staticmethod
    def instantiate_copilot() -> Copilot:
        """Instantiate a new Copilot instance."""
        return Copilot()

    @staticmethod
    def instantiate_handler(rolling_buffer_size: int) -> CopilotResponseHandler:
        """Instantiate a new CopilotResponseHandler instance."""
        return CopilotResponseHandler(
            rolling_buffer_size=rolling_buffer_size,
        )

    @asynccontextmanager
    async def _get_client(self) -> AsyncGenerator[openai.AsyncOpenAI, None]:
        """Get or create OpenAI client with proper resource management."""
        if self._client is None:
            self._client = openai.AsyncOpenAI(timeout=config.OPENAI_TIMEOUT)

        try:
            yield self._client
        except Exception as e:
            structlogger.error("llm.client_error", error=str(e))
            raise

    async def generate_text(self, prompt: str, max_tokens: int = 100) -> str:
        """Generate simple text using OpenAI.

        Args:
            prompt: The text prompt to send to the model
            max_tokens: Maximum tokens to generate

        Returns:
            Generated text response

        Raises:
            LLMGenerationError: If generation fails
        """
        try:
            async with self._get_client() as client:
                response = await client.chat.completions.create(
                    model=config.OPENAI_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,  # Lower temperature for consistent messages
                    max_tokens=max_tokens,
                )

                content = response.choices[0].message.content
                if not content:
                    raise LLMGenerationError("Empty response from LLM")

                return content.strip()

        except openai.OpenAIError as e:
            raise LLMGenerationError(f"OpenAI API error: {e}")
        except asyncio.TimeoutError:
            raise LLMGenerationError("LLM request timed out")


# Global service instance
llm_service = LLMService()
