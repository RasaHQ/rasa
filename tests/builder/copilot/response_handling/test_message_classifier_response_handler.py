from typing import List
from unittest.mock import AsyncMock, MagicMock

import pytest

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


@pytest.fixture(autouse=True)
def mock_openai_client(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(
        "rasa.builder.copilot.response_handling.message_classifier_response_handler.AsyncOpenAI",
        lambda: mock_client,
    )


class TestMessageClassifierResponseHandler:
    @pytest.mark.asyncio
    async def test_stream_out_of_scope_response(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.OUT_OF_SCOPE_DETECTION,
            user_message="What's the weather?",
        )

        responses: List[GeneratedContent] = []
        async for content in handler.stream():
            responses.append(content)

        assert len(responses) == 3
        assert isinstance(responses[0], CopilotTextStartContent)
        assert isinstance(responses[1], ControlledPredictionContent)
        assert isinstance(responses[2], CopilotTextEndContent)

        assert responses[1].response_category == ResponseCategory.OUT_OF_SCOPE_DETECTION

        expected_response = copilot_handler_default_responses()[
            OUT_OF_SCOPE_RESPONSE_KEY
        ]
        assert responses[1].content == expected_response

    @pytest.mark.asyncio
    async def test_stream_roleplay_response(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.ROLEPLAY_DETECTION,
            user_message="I want to book a flight",
        )

        responses: List[GeneratedContent] = []
        async for content in handler.stream():
            responses.append(content)

        assert len(responses) == 3
        assert isinstance(responses[1], ControlledPredictionContent)
        assert responses[1].response_category == ResponseCategory.ROLEPLAY_DETECTION

        expected_response = copilot_handler_default_responses()[ROLEPLAY_RESPONSE_KEY]
        assert responses[1].content == expected_response

    @pytest.mark.asyncio
    async def test_stream_unclear_input_response(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.UNCLEAR_INPUT_DETECTION,
            user_message="asdfkjh",
        )

        responses: List[GeneratedContent] = []
        async for content in handler.stream():
            responses.append(content)

        assert len(responses) == 3
        assert isinstance(responses[1], ControlledPredictionContent)
        assert (
            responses[1].response_category == ResponseCategory.UNCLEAR_INPUT_DETECTION
        )

        expected_response = copilot_handler_default_responses()[
            UNCLEAR_INPUT_RESPONSE_KEY
        ]
        assert responses[1].content == expected_response

    @pytest.mark.asyncio
    async def test_stream_greeting_with_llm(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.GREETING_DETECTION,
            user_message="hi",
        )

        async def mock_greeting_stream(*args, **kwargs):
            yield GeneratedContent(
                content="Hi",
                response_category=ResponseCategory.GREETING_DETECTION,
                response_completeness=ResponseCompleteness.TOKEN,
            )

        handler._stream_llm_response = mock_greeting_stream

        responses: List[GeneratedContent] = []
        async for content in handler.stream():
            responses.append(content)

        assert len(responses) == 3
        assert isinstance(responses[0], CopilotTextStartContent)
        assert isinstance(responses[1], GeneratedContent)
        assert responses[1].content == "Hi"
        assert responses[1].response_category == ResponseCategory.GREETING_DETECTION
        assert isinstance(responses[2], CopilotTextEndContent)

    @pytest.mark.asyncio
    async def test_stream_goodbye_with_llm(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.GOODBYE_DETECTION,
            user_message="bye",
        )

        async def mock_goodbye_stream(*args, **kwargs):
            yield GeneratedContent(
                content="Goodbye",
                response_category=ResponseCategory.GOODBYE_DETECTION,
                response_completeness=ResponseCompleteness.TOKEN,
            )

        handler._stream_llm_response = mock_goodbye_stream

        responses: List[GeneratedContent] = []
        async for content in handler.stream():
            responses.append(content)

        assert len(responses) == 3
        assert isinstance(responses[0], CopilotTextStartContent)
        assert isinstance(responses[1], GeneratedContent)
        assert responses[1].content == "Goodbye"
        assert responses[1].response_category == ResponseCategory.GOODBYE_DETECTION
        assert isinstance(responses[2], CopilotTextEndContent)

    @pytest.mark.asyncio
    async def test_stream_greeting_fallback_on_error(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.GREETING_DETECTION,
            user_message="hi",
        )

        # Mock _client to raise an exception during LLM call
        # This will trigger the exception handling in _stream_llm_response
        handler._client.chat.completions.create = AsyncMock(
            side_effect=Exception("LLM API failed")
        )

        responses: List[GeneratedContent] = []
        async for content in handler.stream():
            responses.append(content)

        # Should have start, fallback token, and end
        assert len(responses) == 3
        assert isinstance(responses[0], CopilotTextStartContent)
        assert isinstance(responses[1], GeneratedContent)

        # Should use the fallback text from the template
        expected_fallback = copilot_handler_default_responses()[
            GREETING_FALLBACK_RESPONSE_KEY
        ]
        assert responses[1].content == expected_fallback
        assert responses[1].response_category == ResponseCategory.GREETING_DETECTION
        assert isinstance(responses[2], CopilotTextEndContent)

        # Generation usage should be None since LLM failed
        assert handler.generation_usage is None

    @pytest.mark.asyncio
    async def test_stream_goodbye_fallback_on_error(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.GOODBYE_DETECTION,
            user_message="bye",
        )

        # Mock _client to raise an exception during LLM call
        # This will trigger the exception handling in _stream_llm_response
        handler._client.chat.completions.create = AsyncMock(
            side_effect=Exception("LLM API failed")
        )

        responses: List[GeneratedContent] = []
        async for content in handler.stream():
            responses.append(content)

        # Should have start, fallback token, and end
        assert len(responses) == 3
        assert isinstance(responses[0], CopilotTextStartContent)
        assert isinstance(responses[1], GeneratedContent)

        # Should use the fallback text from the template
        expected_fallback = copilot_handler_default_responses()[
            GOODBYE_FALLBACK_RESPONSE_KEY
        ]
        assert responses[1].content == expected_fallback
        assert responses[1].response_category == ResponseCategory.GOODBYE_DETECTION
        assert isinstance(responses[2], CopilotTextEndContent)

        # Generation usage should be None since LLM failed
        assert handler.generation_usage is None

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

    def test_generated_responses_initially_empty(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.OUT_OF_SCOPE_DETECTION,
            user_message="test",
        )

        assert handler.generated_responses == []

    @pytest.mark.asyncio
    async def test_generated_responses_populated_after_streaming(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.OUT_OF_SCOPE_DETECTION,
            user_message="test",
        )

        async for _ in handler.stream():
            pass

        # Should have at least the content response
        assert len(handler.generated_responses) >= 1
        assert isinstance(handler.generated_responses[0], ControlledPredictionContent)

    def test_has_not_been_run_initially(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.OUT_OF_SCOPE_DETECTION,
            user_message="test",
        )

        assert handler.raw_llm_stream_item_count == 0
        assert not handler.has_been_run()

    @pytest.mark.asyncio
    async def test_has_been_run_after_streaming(self):
        handler = MessageClassifierResponseHandler(
            response_category=ResponseCategory.OUT_OF_SCOPE_DETECTION,
            user_message="test",
        )

        async for _ in handler.stream():
            pass

        assert handler.raw_llm_stream_item_count == 1
        assert handler.has_been_run()
