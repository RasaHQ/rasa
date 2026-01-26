"""Orchestrated Copilot - Routes requests through classification before generation.

Uses a lightweight classifier to determine if a request needs the
full copilot or can be handled with a quick response. This provides:
- More accurate routing through dedicated classification
- Faster responses for simple requests (greetings, goodbyes)
- Lower costs by avoiding the full copilot when unnecessary
"""

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from rasa.builder import config
from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot
from rasa.builder.copilot.base_copilot import BaseCopilot
from rasa.builder.copilot.message_classifier.message_classifier import MessageClassifier
from rasa.builder.copilot.models import (
    CopilotContext,
    CopilotGenerationContext,
    ResponseCategory,
    UsageStatistics,
)
from rasa.builder.copilot.response_handling.message_classifier_response_handler import (
    MessageClassifierResponseHandler,
)

if TYPE_CHECKING:
    from rasa.builder.copilot.response_handling.agent_copilot_response_handler import (
        AgentCopilotResponseHandler,
    )
    from rasa.builder.copilot.response_handling.legacy_copilot_response_handler import (
        LegacyCopilotResponseHandler,
    )

    CopilotResponseHandler = Union[
        AgentCopilotResponseHandler,
        LegacyCopilotResponseHandler,
        MessageClassifierResponseHandler,
    ]


class OrchestratedCopilot(BaseCopilot):
    """Copilot that uses classifier for classification and routing."""

    def __init__(self) -> None:
        super().__init__()
        self._classifier = MessageClassifier(
            chat_history_size=config.MESSAGE_CLASSIFIER_CHAT_HISTORY_SIZE,
        )
        self._agent_copilot = AgentCopilot()
        self._llm_config: Optional[Dict[str, Any]] = None
        self._orchestration_handler: Optional[MessageClassifierResponseHandler] = None

        self._classification_usage: Optional[UsageStatistics] = None
        self._agent_usage: Optional[UsageStatistics] = None

    @staticmethod
    def _extract_user_message(context: CopilotContext) -> str:
        """Extract the last user message from context.

        Args:
            context: The copilot context containing conversation history.

        Returns:
            The flattened text content of the last user message.
        """
        if context.copilot_chat_history:
            last_message = context.copilot_chat_history[-1]
            if hasattr(last_message, "get_flattened_text_content"):
                return last_message.get_flattened_text_content()

        return ""

    @property
    def llm_config(self) -> Dict[str, Any]:
        """Get the LLM config used for the last response.

        Returns:
            LLM configuration dictionary from the last model used.
        """
        if self._llm_config is None:
            return {}

        return self._llm_config

    @property
    def usage_statistics(self) -> UsageStatistics:
        """Get aggregated usage statistics from all LLM calls in the last response.

        Returns:
            - If routed to full copilot: Agent's usage statistics (classification
              tokens are negligible ~10-15 vs hundreds/thousands from agent)
            - If handled by orchestrator: Aggregated classification + generation
              usage (same model, so aggregation is accurate)
        """
        # If routed to full copilot, return agent usage (classification is negligible)
        if self._agent_usage:
            return self._agent_usage

        # Otherwise, aggregate classification + generation (same model/pricing)
        total = UsageStatistics(
            input_token_price=config.COPILOT_INPUT_TOKEN_PRICE,
            output_token_price=config.COPILOT_OUTPUT_TOKEN_PRICE,
            cached_token_price=config.COPILOT_CACHED_TOKEN_PRICE,
        )

        if self._classification_usage:
            total = total + self._classification_usage

        if self._orchestration_handler and self._orchestration_handler.generation_usage:
            total = total + self._orchestration_handler.generation_usage

        return total

    async def generate_response(
        self, context: CopilotContext
    ) -> Tuple["CopilotResponseHandler", CopilotGenerationContext]:
        """Generate a response using classifier or full copilot.

        Args:
            context: The copilot context containing conversation history.

        Returns:
            Tuple of (response handler, generation context).
        """
        user_message = self._extract_user_message(context)
        classifier_result = await self._classifier.classify(context)

        self._classification_usage = classifier_result.classification_usage

        if classifier_result.requires_full_copilot:
            return await self._handle_full_copilot(context)
        else:
            return self._handle_orchestrated_response(
                classifier_result.category,
                user_message,
            )

    async def _handle_full_copilot(
        self,
        context: CopilotContext,
    ) -> Tuple["CopilotResponseHandler", CopilotGenerationContext]:
        """Delegate to the full agent copilot.

        Args:
            context: The copilot context containing conversation history.

        Returns:
            Tuple of (response handler, generation context).
        """
        handler, generation_context = await self._agent_copilot.generate_response(
            context
        )

        # Store agent usage (classification usage already stored in generate_response)
        self._agent_usage = self._agent_copilot.usage_statistics
        self._llm_config = self._agent_copilot.llm_config
        return handler, generation_context

    def _handle_orchestrated_response(
        self,
        category: ResponseCategory,
        user_message: str,
    ) -> Tuple["CopilotResponseHandler", CopilotGenerationContext]:
        """Handle quick responses through orchestration.

        Args:
            category: The ResponseCategory determined by the classifier.
            user_message: The last user message text.

        Returns:
            Tuple of (response handler, generation context).
        """
        handler = MessageClassifierResponseHandler(
            response_category=category,
            user_message=user_message,
        )

        generation_context = CopilotGenerationContext(
            system_message={
                "role": "system",
                "content": f"Orchestration: {category.value}",
            },
            chat_history=[],
            last_user_message={"role": "user", "content": user_message},
            tracker_event_attachments=[],
        )

        self._orchestration_handler = handler
        self._llm_config = {
            "model": config.ORCHESTRATOR_MODEL,
            "temperature": 0,
            "max_tokens": 50,
        }

        return handler, generation_context
