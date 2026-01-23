from typing import AsyncIterator, List
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest
from openai.types.chat import ChatCompletionChunk

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
from rasa.builder.copilot.response_handling.constants import (
    GOODBYE_FALLBACK_RESPONSE_KEY,
    GREETING_FALLBACK_RESPONSE_KEY,
    OUT_OF_SCOPE_RESPONSE_KEY,
    ROLEPLAY_RESPONSE_KEY,
    UNCLEAR_INPUT_RESPONSE_KEY,
)
from rasa.builder.copilot.response_handling.message_classifier_response_handler import (
    MessageClassifierResponseHandler,
)


async def _create_mock_chunk_stream(
    content: str, prompt_tokens: int, completion_tokens: int
) -> AsyncIterator[ChatCompletionChunk]:
    """Create a mock stream of ChatCompletionChunk objects."""
    # Split content into tokens (words)
    tokens = content.split() if content else []

    for i, token in enumerate(tokens):
        chunk = MagicMock(spec=ChatCompletionChunk)
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta = MagicMock()
        chunk.choices[0].delta.content = token + " "
        chunk.usage = None
        yield chunk

    # Final chunk with usage statistics
    if tokens:
        final_chunk = MagicMock(spec=ChatCompletionChunk)
        final_chunk.choices = [MagicMock()]
        final_chunk.choices[0].delta = MagicMock()
        final_chunk.choices[0].delta.content = None
        final_chunk.usage = MagicMock()
        final_chunk.usage.prompt_tokens = prompt_tokens
        final_chunk.usage.completion_tokens = completion_tokens
        final_chunk.usage.total_tokens = prompt_tokens + completion_tokens
        yield final_chunk


@pytest.fixture(autouse=True)
@patch.object(MessageClassifierResponseHandler, "_get_client")
def mock_get_client(mock_get_client):
    mock_client = AsyncMock(spec=openai.AsyncOpenAI)
    mock_get_client.return_value = AsyncMock()
    mock_get_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_get_client.return_value.__aexit__ = AsyncMock(return_value=None)


class TestMessageClassifierResponseHandler:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response_category,user_message,response_key",
        [
            (
                ResponseCategory.OUT_OF_SCOPE_DETECTION,
                "What's the weather?",
                OUT_OF_SCOPE_RESPONSE_KEY,
            ),
            (
                ResponseCategory.ROLEPLAY_DETECTION,
                "I want to book a flight",
                ROLEPLAY_RESPONSE_KEY,
            ),
            (
                ResponseCategory.UNCLEAR_INPUT_DETECTION,
                "asdfkjh",
                UNCLEAR_INPUT_RESPONSE_KEY,
            ),
        ],
    )
    async def test_stream_template_response(
        self,
        response_category: ResponseCategory,
        user_message: str,
        response_key: str,
    ):
        handler = MessageClassifierResponseHandler(
            response_category=response_category,
            user_message=user_message,
        )

        responses: List[GeneratedContent] = []
        async for content in handler.stream():
            responses.append(content)

        assert len(responses) == 3
        assert isinstance(responses[0], CopilotTextStartContent)
        assert isinstance(responses[1], ControlledPredictionContent)
        assert isinstance(responses[2], CopilotTextEndContent)

        assert responses[1].response_category == response_category

        expected_response = copilot_handler_default_responses()[response_key]
        assert responses[1].content == expected_response

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response_category,user_message,expected_content,fallback_key,should_error",
        [
            (
                ResponseCategory.GREETING_DETECTION,
                "hi",
                "Hi",
                GREETING_FALLBACK_RESPONSE_KEY,
                False,
            ),
            (
                ResponseCategory.GOODBYE_DETECTION,
                "bye",
                "Goodbye",
                GOODBYE_FALLBACK_RESPONSE_KEY,
                False,
            ),
            (
                ResponseCategory.GREETING_DETECTION,
                "hi",
                None,
                GREETING_FALLBACK_RESPONSE_KEY,
                True,
            ),
            (
                ResponseCategory.GOODBYE_DETECTION,
                "bye",
                None,
                GOODBYE_FALLBACK_RESPONSE_KEY,
                True,
            ),
        ],
    )
    @patch.object(MessageClassifierResponseHandler, "_call_llm")
    async def test_stream_with_llm(
        self,
        mock_call_llm: AsyncMock,
        response_category: ResponseCategory,
        user_message: str,
        expected_content: str | None,
        fallback_key: str,
        should_error: bool,
    ):
        handler = MessageClassifierResponseHandler(
            response_category=response_category,
            user_message=user_message,
        )

        if should_error:
            # Mock _call_llm to raise an exception
            # This will trigger the exception handling in _stream_llm_response
            mock_call_llm.side_effect = Exception("LLM API failed")
        else:
            # Mock _call_llm to return a stream of chunks
            mock_call_llm.return_value = _create_mock_chunk_stream(
                expected_content, prompt_tokens=10, completion_tokens=5
            )

        responses: List[GeneratedContent] = []
        async for content in handler.stream():
            responses.append(content)

        assert len(responses) == 3
        assert isinstance(responses[0], CopilotTextStartContent)
        assert isinstance(responses[1], GeneratedContent)
        assert responses[1].response_category == response_category
        assert isinstance(responses[2], CopilotTextEndContent)

        if should_error:
            # Should use the fallback text from the template
            expected_fallback = copilot_handler_default_responses()[fallback_key]
            assert responses[1].content == expected_fallback
            # Generation usage should be None since LLM failed
            assert handler.generation_usage is None
        else:
            # The mock adds a space after each token, so expect "Hi " or "Goodbye "
            assert responses[1].content == expected_content + " "

    def test_generation_usage_no_generation(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.OUT_OF_SCOPE_DETECTION,
            user_message="test",
        )

        assert handler.generation_usage is None

    @pytest.mark.asyncio
    async def test_generation_usage_after_streaming_template(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.OUT_OF_SCOPE_DETECTION,
            user_message="test",
        )

        responses: List[GeneratedContent] = []
        async for content in handler.stream():
            responses.append(content)

        # For template responses, generation usage should be None
        assert handler.generation_usage is None

    def test_raw_llm_stream_data_without_generation(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.ROLEPLAY_DETECTION,
            user_message="hi",
        )

        data = handler.raw_llm_stream_data

        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0] == "Classification: roleplay_detection"

    @pytest.mark.asyncio
    async def test_raw_llm_stream_data_with_generation(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.GREETING_DETECTION,
            user_message="hi",
        )

        async def mock_greeting_stream(*args, **kwargs):
            handler._generation_usage = UsageStatistics(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            )
            yield GeneratedContent(
                content="Hello!",
                response_category=ResponseCategory.GREETING_DETECTION,
                response_completeness=ResponseCompleteness.TOKEN,
            )

        handler._stream_llm_response = mock_greeting_stream

        # Stream to populate generation_usage
        async for _ in handler.stream():
            pass

        data = handler.raw_llm_stream_data

        assert isinstance(data, list)
        assert len(data) == 2
        assert data[0] == "Classification: greeting_detection"
        assert data[1] == "Generation: greeting_detection"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "should_stream,expected_item_count,expected_has_been_run",
        [
            (False, 0, False),
            (True, 1, True),
        ],
    )
    async def test_has_been_run(
        self,
        should_stream: bool,
        expected_item_count: int,
        expected_has_been_run: bool,
    ):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.OUT_OF_SCOPE_DETECTION,
            user_message="test",
        )

        if should_stream:
            async for _ in handler.stream():
                pass

        assert handler.raw_llm_stream_item_count == expected_item_count
        assert handler.has_been_run() == expected_has_been_run
