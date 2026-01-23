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
        self._usage_statistics: Optional[UsageStatistics] = None
        self._llm_config: Optional[Dict[str, Any]] = None
        self._orchestration_handler: Optional[MessageClassifierResponseHandler] = None

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
        """Get usage statistics from the last response.

        For orchestrated responses, always includes classification usage statistics.
        For greetings/goodbyes, also aggregates generation usage.

        Returns:
            UsageStatistics aggregated from all LLM calls in the last response.
        """
        if self._usage_statistics is None:
            return UsageStatistics(
                input_token_price=config.COPILOT_INPUT_TOKEN_PRICE,
                output_token_price=config.COPILOT_OUTPUT_TOKEN_PRICE,
                cached_token_price=config.COPILOT_CACHED_TOKEN_PRICE,
            )

        # For orchestrated responses with generation (greetings/goodbyes),
        # aggregate classification usage statistics + generation usage
        if self._orchestration_handler and self._orchestration_handler.generation_usage:
            classification_usage = self._usage_statistics
            generation_usage = self._orchestration_handler.generation_usage

            return UsageStatistics(
                model=config.ORCHESTRATOR_MODEL,
                prompt_tokens=(classification_usage.prompt_tokens or 0)
                + (generation_usage.prompt_tokens or 0),
                completion_tokens=(classification_usage.completion_tokens or 0)
                + (generation_usage.completion_tokens or 0),
                total_tokens=(classification_usage.total_tokens or 0)
                + (generation_usage.total_tokens or 0),
                cached_prompt_tokens=0,
                input_token_price=config.COPILOT_INPUT_TOKEN_PRICE,
                output_token_price=config.COPILOT_OUTPUT_TOKEN_PRICE,
                cached_token_price=config.COPILOT_CACHED_TOKEN_PRICE,
            )

        # For responses without generation return classification usage statistics only
        return self._usage_statistics

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

        if classifier_result.requires_full_copilot:
            return await self._handle_full_copilot(context)
        else:
            return self._handle_orchestrated_response(
                classifier_result.category,
                user_message,
                classifier_result.classification_usage,
            )

    async def _handle_full_copilot(
        self, context: CopilotContext
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
        self._usage_statistics = self._agent_copilot.usage_statistics
        self._llm_config = self._agent_copilot.llm_config
        self._orchestration_handler = None
        return handler, generation_context

    def _handle_orchestrated_response(
        self,
        category: ResponseCategory,
        user_message: str,
        classification_usage: UsageStatistics,
    ) -> Tuple["CopilotResponseHandler", CopilotGenerationContext]:
        """Handle quick responses through orchestration.

        Args:
            category: The ResponseCategory determined by the classifier.
            user_message: The last user message text.
            classification_usage: Usage stats from the classification LLM call.

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

        # Store classification usage; generation usage will be added after streaming
        self._usage_statistics = classification_usage
        self._orchestration_handler = handler
        self._llm_config = {
            "model": config.ORCHESTRATOR_MODEL,
            "temperature": 0,
            "max_tokens": 50,
        }

        return handler, generation_context
