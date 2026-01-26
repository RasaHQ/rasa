"""Response handler for simple requests handled by the MessageClassifier.

When a user sends a simple request (like "hi", "bye", or an off-topic question),
the classifier classifies it and this handler generates the appropriate response
without invoking the full copilot.

For complex requests (like "create a booking flow"), the classifier delegates
to the full AgentCopilot, which uses AgentCopilotResponseHandler instead.
"""

import copy
import importlib.resources
from contextlib import asynccontextmanager
from typing import AsyncGenerator, AsyncIterator, List, Optional, Union

import openai
import structlog
from openai.types.chat import ChatCompletionChunk

from rasa.builder import config
from rasa.builder.copilot.constants import (
    GOODBYE_PROMPT_FILE,
    GREETING_PROMPT_FILE,
    RESPONSE_HANDLER_PROMPTS_DIR,
)
from rasa.builder.copilot.copilot_templated_message_provider import (
    copilot_handler_default_responses,
)
from rasa.builder.copilot.models import (
    ControlledPredictionContent,
    CopilotTextEndContent,
    CopilotTextStartContent,
    GeneratedContent,
    ResponseCategory,
    ResponseCompleteness,
    UsageStatistics,
)
from rasa.builder.copilot.response_handling.base_copilot_response_handler import (
    BaseCopilotResponseHandler,
)
from rasa.builder.copilot.response_handling.constants import (
    ERROR_FALLBACK_RESPONSE_KEY,
    GOODBYE_FALLBACK_RESPONSE_KEY,
    GREETING_FALLBACK_RESPONSE_KEY,
    KNOWLEDGE_BASE_ACCESS_REQUESTED_RESPONSE_KEY,
    OUT_OF_SCOPE_RESPONSE_KEY,
    ROLEPLAY_RESPONSE_KEY,
    UNCLEAR_INPUT_RESPONSE_KEY,
)
from rasa.builder.document_retrieval.models import Document
from rasa.builder.telemetry.langfuse.message_classifier_langfuse_telemetry import (
    MessageClassifierResponseHandlerLangfuseTelemetry,
)
from rasa.shared.constants import PACKAGE_NAME

structlogger = structlog.get_logger()


class MessageClassifierResponseHandler(BaseCopilotResponseHandler):
    """Handler for simple requests (greetings, goodbyes, out-of-scope).

    Generates responses for requests that don't require the full copilot.
    For complex requests, AgentCopilotResponseHandler is used instead.
    """

    # LLM parameters for response generation
    GREETING_MAX_TOKENS = 50
    GOODBYE_MAX_TOKENS = 30
    GENERATION_TEMPERATURE = 0.7  # More creative for greetings/goodbyes

    def __init__(self, response_category: ResponseCategory, user_message: str) -> None:
        """Initialize the classification response handler.

        Args:
            response_category: The classified message category.
            user_message: The user's original message.
        """
        super().__init__()
        self._user_message = user_message
        self._response_category = response_category
        self._response_content: Optional[
            Union[GeneratedContent, ControlledPredictionContent]
        ] = None
        self._response_text: str = ""

        self._generation_usage: Optional[UsageStatistics] = None
        self._generated_responses: List[
            Union[GeneratedContent, ControlledPredictionContent]
        ] = []

        self._client: Optional[openai.AsyncOpenAI] = None

        # Load prompt templates
        self._greeting_prompt = importlib.resources.read_text(
            f"{PACKAGE_NAME}.{RESPONSE_HANDLER_PROMPTS_DIR}",
            GREETING_PROMPT_FILE,
        ).strip()
        self._goodbye_prompt = importlib.resources.read_text(
            f"{PACKAGE_NAME}.{RESPONSE_HANDLER_PROMPTS_DIR}",
            GOODBYE_PROMPT_FILE,
        ).strip()

    @asynccontextmanager
    async def _get_client(self) -> AsyncGenerator[openai.AsyncOpenAI, None]:
        """Get or lazy create OpenAI client with proper resource management."""
        if self._client is None:
            self._client = openai.AsyncOpenAI()

        try:
            yield self._client
        except Exception as e:
            structlogger.error("response_handler.llm_client_error", error=str(e))
            raise

    async def _stream_llm_response(
        self,
        system_prompt: str,
        max_tokens: int,
        fallback_text: str,
        log_prefix: str,
    ) -> AsyncIterator[GeneratedContent]:
        """Stream an LLM response token by token.

        Args:
            system_prompt: The system prompt to use.
            max_tokens: Maximum tokens for the response.
            fallback_text: Fallback text if generation fails.
            log_prefix: Prefix for log messages (e.g., "greeting", "goodbye").

        Yields:
            GeneratedContent objects for each token.
        """
        accumulated_text = ""
        try:
            stream = await self._call_llm(system_prompt, self._user_message, max_tokens)

            async for chunk in stream:
                # Extract usage statistics from the final chunk
                if chunk.usage:
                    self._generation_usage = UsageStatistics(
                        model=config.ORCHESTRATOR_MODEL,
                        prompt_tokens=chunk.usage.prompt_tokens,
                        completion_tokens=chunk.usage.completion_tokens,
                        total_tokens=chunk.usage.total_tokens,
                        cached_prompt_tokens=0,
                        input_token_price=config.COPILOT_INPUT_TOKEN_PRICE,
                        output_token_price=config.COPILOT_OUTPUT_TOKEN_PRICE,
                        cached_token_price=config.COPILOT_CACHED_TOKEN_PRICE,
                    )

                # Extract token content
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    accumulated_text += token

                    # Yield token as streaming content
                    token_content = GeneratedContent(
                        content=token,
                        response_category=self._response_category,
                        response_completeness=ResponseCompleteness.TOKEN,
                    )
                    self._generated_responses.append(token_content)
                    yield token_content

            self._response_text = accumulated_text.strip()
            structlogger.info(
                f"classifier.{log_prefix}.streamed",
                event_info=f"Streamed {log_prefix} response",
                prompt_tokens=(
                    self._generation_usage.prompt_tokens
                    if self._generation_usage
                    else 0
                ),
                completion_tokens=(
                    self._generation_usage.completion_tokens
                    if self._generation_usage
                    else 0
                ),
            )
        except Exception as e:
            structlogger.warning(
                f"classifier.{log_prefix}.error",
                event_info=f"Failed to stream {log_prefix} response",
                error=str(e),
            )
            # Yield fallback as a single token
            self._response_text = fallback_text
            self._generation_usage = None
            fallback_content = GeneratedContent(
                content=fallback_text,
                response_category=self._response_category,
                response_completeness=ResponseCompleteness.TOKEN,
            )
            self._generated_responses.append(fallback_content)
            yield fallback_content

    async def _call_llm(
        self,
        system_prompt: str,
        user_message: str,
        max_tokens: int,
    ) -> AsyncIterator[ChatCompletionChunk]:
        """Call the LLM with the given messages.

        Args:
            system_prompt: The system prompt to use.
            user_message: The user's message.
            max_tokens: The maximum tokens for the response.

        Returns:
            The AsyncIterator of ChatCompletionChunk objects from the LLM.
        """
        async with self._get_client() as client:
            return await client.chat.completions.create(
                model=config.ORCHESTRATOR_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=self.GENERATION_TEMPERATURE,
                stream=True,
                stream_options={"include_usage": True},
            )

    @MessageClassifierResponseHandlerLangfuseTelemetry.trace_response_generation(
        "greeting", max_tokens=GREETING_MAX_TOKENS
    )
    async def _stream_greeting(self) -> AsyncIterator[GeneratedContent]:
        """Stream a greeting response matching user's tone."""
        responses = copilot_handler_default_responses()
        fallback_text = responses.get(
            GREETING_FALLBACK_RESPONSE_KEY, "Hello! How can I help?"
        )

        async for token in self._stream_llm_response(
            system_prompt=self._greeting_prompt,
            max_tokens=self.GREETING_MAX_TOKENS,
            fallback_text=fallback_text,
            log_prefix="greeting",
        ):
            yield token

    @MessageClassifierResponseHandlerLangfuseTelemetry.trace_response_generation(
        "goodbye", max_tokens=GOODBYE_MAX_TOKENS
    )
    async def _stream_goodbye(self) -> AsyncIterator[GeneratedContent]:
        """Stream a goodbye response matching user's tone."""
        responses = copilot_handler_default_responses()
        fallback_text = responses.get(
            GOODBYE_FALLBACK_RESPONSE_KEY, "Goodbye! Happy building!"
        )

        async for token in self._stream_llm_response(
            system_prompt=self._goodbye_prompt,
            max_tokens=self.GOODBYE_MAX_TOKENS,
            fallback_text=fallback_text,
            log_prefix="goodbye",
        ):
            yield token

    def _generate_template_response(self) -> None:
        """Generate a template-based response for out-of-scope categories."""
        responses = copilot_handler_default_responses()

        category_to_response_key = {
            ResponseCategory.ROLEPLAY_DETECTION: ROLEPLAY_RESPONSE_KEY,
            ResponseCategory.OUT_OF_SCOPE_DETECTION: OUT_OF_SCOPE_RESPONSE_KEY,
            ResponseCategory.UNCLEAR_INPUT_DETECTION: UNCLEAR_INPUT_RESPONSE_KEY,
            ResponseCategory.KNOWLEDGE_BASE_ACCESS_REQUESTED: (
                KNOWLEDGE_BASE_ACCESS_REQUESTED_RESPONSE_KEY
            ),
            ResponseCategory.ERROR_FALLBACK: ERROR_FALLBACK_RESPONSE_KEY,
        }

        response_key = category_to_response_key.get(self._response_category)
        self._response_text = responses.get(response_key, "") if response_key else ""

        self._response_content = ControlledPredictionContent(
            content=self._response_text,
            response_category=self._response_category,
            response_completeness=ResponseCompleteness.COMPLETE,
        )
        self._generated_responses = [self._response_content]

    async def stream(
        self,
    ) -> AsyncIterator[Union[GeneratedContent, ControlledPredictionContent]]:
        """Stream the classified response with START/END markers.

        For greetings/goodbyes: streams token by token from LLM.
        For template responses: yields complete response.

        Yields:
            START marker, response content (streamed or complete), END marker.
        """
        yield CopilotTextStartContent()

        # Stream LLM responses token by token
        if self._response_category == ResponseCategory.GREETING_DETECTION:
            async for token in self._stream_greeting():
                yield token
        elif self._response_category == ResponseCategory.GOODBYE_DETECTION:
            async for token in self._stream_goodbye():
                yield token
        else:
            # Template responses (out-of-scope, roleplay, etc.) - yield as complete
            if self._response_content is None:
                self._generate_template_response()
            assert self._response_content is not None, "Response content must be set"
            yield self._response_content

        yield CopilotTextEndContent()

    def reset(self) -> None:
        """Reset handler state (no-op for classified responses)."""
        pass

    @property
    def generated_responses(self) -> List[GeneratedContent]:
        """Return the list of generated responses."""
        return copy.deepcopy(self._generated_responses)

    @property
    def generated_responses_count(self) -> int:
        """Return the count of generated responses."""
        return len(self._generated_responses)

    @property
    def raw_llm_stream_data(self) -> List[str]:
        """Return raw LLM stream data from the message classifier.

        Returns:
            List of strings summarizing LLM usage.
        """
        data = [f"Classification: {self._response_category.value}"]
        if self._generation_usage:
            data.append(f"Generation: {self._response_category.value}")
        return data

    @property
    def raw_llm_stream_text_content(self) -> str:
        """Return raw LLM stream text content."""
        return self._response_text

    @property
    def raw_llm_stream_item_count(self) -> int:
        """Return raw LLM stream item count.

        Returns:
            0 if handler hasn't run yet.
            1 if handler ran with only classification (template responses).
            2 if handler ran with classification + generation (greetings/goodbyes).
        """
        if self.generated_responses_count == 0:
            return 0

        return 2 if self._generation_usage else 1

    @property
    def generation_usage(self) -> UsageStatistics | None:
        """Return usage statistics from response generation (if any).

        Returns:
            UsageStatistics for greeting/goodbye generation, None for templates.
        """
        return self._generation_usage

    @property
    def retrieved_documents(self) -> List[Document]:
        """Return the list of retrieved documents.

        MessageClassifierResponseHandler doesn't retrieve documents as it handles
        simple requests (greetings, goodbyes, out-of-scope) that don't require
        document retrieval.

        Returns:
            Empty list since this handler doesn't retrieve documents.
        """
        return []

    def extract_text_from_generated_responses(self) -> str:
        """Extract the full text from generated responses."""
        return self._response_text

    def extract_response_category(self) -> ResponseCategory:
        """Extract the response category."""
        return self._response_category
