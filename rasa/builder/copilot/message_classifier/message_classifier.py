"""MessageClassifier - Classifies user requests for routing.

The message classifier uses a lightweight LLM (gpt-4o-mini) to classify user requests
and decide whether to handle them directly or delegate to the full copilot.

Benefits:
- Dedicated logic for more precise categorization
- Simple requests for faster responses and lower cost
- Clean separation between classification and generation
"""

import importlib.resources
import json
import re
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, ClassVar, Dict, List, Optional

import openai
import structlog
from jinja2 import Template
from openai.types.chat import ChatCompletion

from rasa.builder import config
from rasa.builder.copilot.constants import (
    MESSAGE_CLASSIFIER_ATTACHMENTS_CONTEXT_PROMPT_FILE,
    MESSAGE_CLASSIFIER_PROMPT_FILE,
    MESSAGE_CLASSIFIER_PROMPTS_DIR,
)
from rasa.builder.copilot.message_classifier.models import MessageClassifierResult
from rasa.builder.copilot.models import (
    ChatMessage,
    CopilotContext,
    CopilotSystemMessage,
    EventContent,
    ResponseCategory,
    UsageStatistics,
)
from rasa.builder.copilot.utils import filter_chat_history_messages
from rasa.builder.telemetry.langfuse.langfuse_compat import observe
from rasa.builder.telemetry.langfuse.message_classifier_langfuse_telemetry import (
    MessageClassifierLangfuseTelemetry,
)
from rasa.shared.constants import PACKAGE_NAME

structlogger = structlog.get_logger()

# Token pattern for extraction from LLM response
TOKEN_PATTERN = re.compile(r"\[([A-Z_]+)\]")


class MessageClassifier:
    """Classifies user requests and routes them to the appropriate handler.

    Uses a lightweight LLM (e.g. gpt-4o-mini) to categorize incoming requests.
    Technical questions are routed to the full copilot agent, while simple
    requests (greetings, goodbyes) and out-of-scope queries are handled directly.

    The focused classification model provides more accurate routing decisions
    than embedding classification logic in the main copilot prompt.
    """

    # LLM parameters for classification
    CLASSIFICATION_MAX_TOKENS: ClassVar[int] = 50
    CLASSIFICATION_TEMPERATURE: ClassVar[float] = 0.0

    # Categories that the MessageClassifier can classify into
    CLASSIFIER_CATEGORIES: ClassVar[List[ResponseCategory]] = [
        ResponseCategory.GREETING_DETECTION,
        ResponseCategory.GOODBYE_DETECTION,
        ResponseCategory.ROLEPLAY_DETECTION,
        ResponseCategory.OUT_OF_SCOPE_DETECTION,
        ResponseCategory.KNOWLEDGE_BASE_ACCESS_REQUESTED,
        ResponseCategory.UNCLEAR_INPUT_DETECTION,
        ResponseCategory.COPILOT,
        ResponseCategory.RASA_INTRODUCTION_DETECTION,
        ResponseCategory.COPILOT_INTRODUCTION_DETECTION,
    ]

    # Map classifier tokens to ResponseCategory
    TOKEN_TO_CATEGORY: ClassVar[Dict[str, ResponseCategory]] = {
        f"[{category.value.upper()}]": category for category in CLASSIFIER_CATEGORIES
    }

    def __init__(self, chat_history_size: Optional[int] = None) -> None:
        """Initialize the MessageClassifier.

        Args:
            chat_history_size: Maximum number of chat history messages to include.
                If None, uses all of the chat history.
        """
        self._client: Optional[openai.AsyncOpenAI] = None
        self._system_prompt_template = self._load_system_prompt_template()
        self._attachments_context_prompt_template = (
            self._load_attachments_context_prompt_template()
        )
        self._chat_history_size = chat_history_size

    @staticmethod
    def _load_system_prompt_template() -> Template:
        """Load the orchestrator prompt template from resources.

        Returns:
            Template object containing the prompt template.
        """
        template_content = importlib.resources.read_text(
            f"{PACKAGE_NAME}.{MESSAGE_CLASSIFIER_PROMPTS_DIR}",
            MESSAGE_CLASSIFIER_PROMPT_FILE,
        )
        return Template(template_content)

    @staticmethod
    def _load_attachments_context_prompt_template() -> Template:
        """Load the last user message context prompt template from resources.

        Returns:
            Template object containing the prompt template.
        """
        template_content = importlib.resources.read_text(
            f"{PACKAGE_NAME}.{MESSAGE_CLASSIFIER_PROMPTS_DIR}",
            MESSAGE_CLASSIFIER_ATTACHMENTS_CONTEXT_PROMPT_FILE,
        )
        return Template(template_content)

    @asynccontextmanager
    async def _get_client(self) -> AsyncGenerator[openai.AsyncOpenAI, None]:
        """Get or lazy create OpenAI client with proper resource management."""
        if self._client is None:
            self._client = openai.AsyncOpenAI()

        try:
            yield self._client
        except Exception as e:
            structlogger.error("classifier.llm_client_error", error=str(e))
            raise

    @MessageClassifierLangfuseTelemetry.trace_classification
    async def classify(self, context: CopilotContext) -> MessageClassifierResult:
        """Classify a user message and decide how to handle it.

        Args:
            context: The copilot context containing conversation history.

        Returns:
            MessageClassifierResult with the decision category and usage stats.
        """
        # Build messages: system, chat_history, latest user message
        messages = await self._build_messages(context)

        try:
            # If messages list is empty, raise ValueError to trigger fallback to copilot
            if not messages:
                raise ValueError(
                    "MessageClassifier._build_messages returned empty list"
                )

            response = await self._call_llm(messages)

            raw_response = (response.choices[0].message.content or "").strip()
            category = self._parse_category(raw_response)

            # Create usage statistics for classification
            classification_usage = UsageStatistics(
                model=config.ORCHESTRATOR_MODEL,
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=(
                    response.usage.completion_tokens if response.usage else 0
                ),
                total_tokens=response.usage.total_tokens if response.usage else 0,
                cached_prompt_tokens=0,
                input_token_price=config.COPILOT_INPUT_TOKEN_PRICE,
                output_token_price=config.COPILOT_OUTPUT_TOKEN_PRICE,
                cached_token_price=config.COPILOT_CACHED_TOKEN_PRICE,
            )

            structlogger.info(
                "classifier.classify.success",
                event_info="Orchestrator classification completed",
                category=category.value,
                raw_response=raw_response,
            )

            return MessageClassifierResult(
                category=category,
                classification_usage=classification_usage,
                raw_response=raw_response,
            )

        except Exception as e:
            structlogger.error(
                "classifier.classify.error",
                event_info="Orchestrator classification failed",
                error=str(e),
            )
            # Default to triggering copilot on error (fail-safe)
            return MessageClassifierResult(
                category=ResponseCategory.COPILOT,
                classification_usage=UsageStatistics(
                    model=config.ORCHESTRATOR_MODEL,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    cached_prompt_tokens=0,
                    input_token_price=config.COPILOT_INPUT_TOKEN_PRICE,
                    output_token_price=config.COPILOT_OUTPUT_TOKEN_PRICE,
                    cached_token_price=config.COPILOT_CACHED_TOKEN_PRICE,
                ),
                raw_response=f"ERROR: {e!s}",
            )

    @observe(as_type="generation")
    async def _call_llm(self, messages: List[Dict[str, Any]]) -> ChatCompletion:
        """Call the LLM with the given messages.

        Args:
            messages: The messages to call the LLM with.

        Returns:
            The ChatCompletion response from the LLM.
        """
        async with self._get_client() as client:
            return await client.chat.completions.create(
                model=config.ORCHESTRATOR_MODEL,
                messages=messages,
                max_tokens=self.CLASSIFICATION_MAX_TOKENS,
                temperature=self.CLASSIFICATION_TEMPERATURE,
            )

    async def _build_messages(
        self,
        context: CopilotContext,
    ) -> List[Dict[str, Any]]:
        """Build the complete message list for the OpenAI API.

        Args:
            context: The context of the copilot.

        Returns:
            A list of messages in OpenAI format.
        """
        if not context.copilot_chat_history:
            return []

        latest_user_message = self._process_latest_message(context)
        if not latest_user_message:
            return []

        # Render the system prompt and convert it to OpenAI format
        system_message = self._create_system_message()

        # Get chat history (excluding the latest message)
        # Filter chat history messages and convert to OpenAI format (excluding the
        # latest message)
        chat_history = context.copilot_chat_history[:-1]
        chat_history_messages = self._create_chat_history_messages(chat_history)

        return [
            system_message,
            *chat_history_messages,
            latest_user_message,
        ]

    def _create_system_message(self) -> Dict[str, Any]:
        """Render the system prompt for the classification LLM.

        Returns:
            System prompt in string format.
        """
        system_prompt = self._system_prompt_template.render()
        return CopilotSystemMessage().build_openai_message(prompt=system_prompt)

    def _create_chat_history_messages(
        self, chat_history: List[ChatMessage]
    ) -> List[Dict[str, Any]]:
        """Filter and convert past messages to OpenAI format.

        Excludes guardrails policy violations and non-user/copilot messages. The chat
        history is limited to the configured size.

        Args:
            chat_history: List of chat messages to filter and convert.

        Returns:
            List of messages in OpenAI format
        """
        filtered_messages = filter_chat_history_messages(
            chat_history,
            excluded_response_categories=[ResponseCategory.GUARDRAILS_POLICY_VIOLATION],
        )

        # Limit chat history to configured size (get last N messages)
        if self._chat_history_size is not None and self._chat_history_size > 0:
            filtered_messages = filtered_messages[-self._chat_history_size :]

        return [message.build_openai_message() for message in filtered_messages]

    def _process_latest_message(
        self,
        context: CopilotContext,
    ) -> Optional[Dict[str, Any]]:
        """Process the latest message and convert it to OpenAI format.

        Args:
            context: The copilot context containing conversation state.

        Returns:
            Message in OpenAI format.

        Raises:
            ValueError: If the message type is not supported.
        """
        latest_message = context.get_last_user_message()
        if not latest_message:
            return None
        tracker_event_attachments = latest_message.get_content_blocks_by_type(
            EventContent
        )
        rendered_prompt = self._render_attachments_context_prompt(
            tracker_event_attachments
        )
        return latest_message.build_openai_message(prompt=rendered_prompt)

    def _render_attachments_context_prompt(
        self,
        attachments: List[EventContent],
    ) -> Optional[str]:
        """Render the attachments context prompt.

        Args:
            attachments: The attachments.

        Returns:
            The rendered prompt if there are attachments, otherwise None.
        """
        if not attachments:
            return None
        attachments_json = json.dumps(
            [attachment.model_dump() for attachment in attachments],
            ensure_ascii=False,
            indent=2,
        )
        return self._attachments_context_prompt_template.render(
            attachments=attachments_json,
        )

    def _parse_category(self, raw_response: str) -> ResponseCategory:
        """Parse the LLM response into a ResponseCategory.

        Args:
            raw_response: Raw text response from the classification LLM.

        Returns:
            ResponseCategory enum value.
        """
        # Extract token from response using regex
        if match := TOKEN_PATTERN.search(raw_response):
            token = f"[{match.group(1)}]"
            if token in self.TOKEN_TO_CATEGORY:
                return self.TOKEN_TO_CATEGORY[token]

        # If we can't parse, default to copilot (fail-safe)
        structlogger.warning(
            "classifier.parse.fallback",
            event_info="Could not parse orchestrator response, defaulting to COPILOT",
            raw_response=raw_response,
        )
        return ResponseCategory.COPILOT
