"""Service for handling LLM interactions."""

from typing import Any, Dict, Optional

import openai
import structlog

from rasa.builder import config
from rasa.builder.copilot.history_store import (
    CopilotHistoryStore,
    SQLiteCopilotHistoryStore,
)
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
        self._guardrails: Optional[GuardrailsClient] = None
        self._guardrails_policy_checker: Optional[GuardrailsPolicyChecker] = None
        self._history_store: Optional[CopilotHistoryStore] = None

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


# Global service instance
llm_service = LLMService()
