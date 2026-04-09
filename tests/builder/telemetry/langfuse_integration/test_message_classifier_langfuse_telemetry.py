from unittest.mock import MagicMock

import pytest

from rasa.builder.copilot.message_classifier.message_classifier import (
    MessageClassifier,
)
from rasa.builder.copilot.message_classifier.models import MessageClassifierResult
from rasa.builder.copilot.models import (
    CopilotContext,
    ResponseCategory,
    TextContent,
    UsageStatistics,
    UserChatMessage,
)
from rasa.builder.telemetry.langfuse_integration.message_classifier_langfuse_telemetry import (  # noqa: E501
    MessageClassifierLangfuseTelemetry,
    MessageClassifierResponseHandlerLangfuseTelemetry,
)


class TestMessageClassifierLangfuseTelemetry:
    @pytest.fixture
    def mock_langfuse_client(self):
        mock_client = MagicMock()
        mock_generation = MagicMock()
        mock_client.start_as_current_generation.return_value.__enter__.return_value = (
            mock_generation
        )
        return mock_client, mock_generation

    @pytest.mark.asyncio
    async def test_trace_classification_decorator_no_langfuse(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "rasa.builder.telemetry.langfuse_integration.message_classifier_langfuse_telemetry.is_langfuse_available",
            lambda: False,
        )

        # Create a simple async function to decorate
        async def mock_classify(
            self, context: CopilotContext
        ) -> MessageClassifierResult:
            return MessageClassifierResult(
                category=ResponseCategory.GREETING_DETECTION,
                classification_usage=UsageStatistics(
                    model="gpt-4o-mini",
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cached_prompt_tokens=0,
                    input_token_price=0.0,
                    output_token_price=0.0,
                    cached_token_price=0.0,
                ),
                raw_response="[GREETING_DETECTION]",
            )

        decorated = MessageClassifierLangfuseTelemetry.trace_classification(
            mock_classify
        )

        # Should be the same function (no wrapping)
        assert decorated == mock_classify

    @pytest.mark.asyncio
    async def test_trace_classification_captures_input_output(
        self, monkeypatch: pytest.MonkeyPatch, mock_langfuse_client
    ):
        mock_client, mock_generation = mock_langfuse_client

        monkeypatch.setattr(
            "rasa.builder.telemetry.langfuse_integration.message_classifier_langfuse_telemetry.is_langfuse_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "rasa.builder.telemetry.langfuse_integration.message_classifier_langfuse_telemetry.langfuse.get_client",
            lambda: mock_client,
        )

        # Create a mock classifier
        classifier = MagicMock(spec=MessageClassifier)
        classifier.__class__.__name__ = "MessageClassifier"

        # Create a test context with a user message
        context = CopilotContext(
            copilot_chat_history=[
                UserChatMessage(
                    role="user",
                    content=[TextContent(type="text", text="hi there")],
                )
            ],
            assistant_id="test-assistant",
        )

        # Create the actual function to test
        async def mock_classify(
            self, context: CopilotContext
        ) -> MessageClassifierResult:
            return MessageClassifierResult(
                category=ResponseCategory.GREETING_DETECTION,
                classification_usage=UsageStatistics(
                    model="gpt-4o-mini",
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                    cached_prompt_tokens=0,
                    input_token_price=0.15,
                    output_token_price=0.6,
                    cached_token_price=0.075,
                ),
                raw_response="[GREETING_DETECTION]",
            )

        decorated = MessageClassifierLangfuseTelemetry.trace_classification(
            mock_classify
        )

        # Call the decorated function
        result = await decorated(classifier, context)

        # Verify Langfuse was called correctly
        mock_client.start_as_current_generation.assert_called_once_with(
            name="MessageClassifier.mock_classify",
            input={"user_message": "hi there"},
        )

        # Verify generation.update was called
        assert mock_generation.update.called

        # Verify result is returned correctly
        assert result.category == ResponseCategory.GREETING_DETECTION


class TestMessageClassifierResponseHandlerLangfuseTelemetry:
    @pytest.fixture
    def mock_langfuse_client(self):
        mock_client = MagicMock()
        mock_generation = MagicMock()
        mock_client.start_as_current_generation.return_value.__enter__.return_value = (
            mock_generation
        )
        return mock_client, mock_generation

    @pytest.mark.asyncio
    async def test_trace_response_generation_no_langfuse(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        monkeypatch.setattr(
            "rasa.builder.telemetry.langfuse_integration.message_classifier_langfuse_telemetry.is_langfuse_available",
            lambda: False,
        )

        # Create a simple async generator to decorate
        async def mock_stream_greeting(self):
            yield MagicMock(content="Hello!")

        decorated = (
            MessageClassifierResponseHandlerLangfuseTelemetry.trace_response_generation(
                "greeting", max_tokens=50
            )(mock_stream_greeting)
        )

        # Should be the same function (no wrapping)
        assert decorated == mock_stream_greeting

    @pytest.mark.asyncio
    async def test_trace_greeting_generation_captures_output(
        self, monkeypatch: pytest.MonkeyPatch, mock_langfuse_client
    ):
        mock_client, mock_generation = mock_langfuse_client

        monkeypatch.setattr(
            "rasa.builder.telemetry.langfuse_integration.message_classifier_langfuse_telemetry.is_langfuse_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "rasa.builder.telemetry.langfuse_integration.message_classifier_langfuse_telemetry.langfuse.get_client",
            lambda: mock_client,
        )

        # Create a mock handler
        handler = MagicMock()
        handler._user_message = "hi there"
        handler._generation_usage = UsageStatistics(
            model="gpt-4o-mini",
            prompt_tokens=20,
            completion_tokens=10,
            total_tokens=30,
            cached_prompt_tokens=0,
            input_token_price=0.15,
            output_token_price=0.6,
            cached_token_price=0.075,
        )

        # Create the actual function to test
        async def mock_stream_greeting(self):
            tokens = ["Hello", "! ", "How ", "can ", "I ", "help?"]
            for token in tokens:
                yield MagicMock(content=token)

        decorated = (
            MessageClassifierResponseHandlerLangfuseTelemetry.trace_response_generation(
                "greeting", max_tokens=50
            )(mock_stream_greeting)
        )

        # Call the decorated function and collect output
        output_tokens = []
        async for token in decorated(handler):
            output_tokens.append(token.content)

        # Verify Langfuse was called correctly
        mock_client.start_as_current_generation.assert_called_once_with(
            name="MessageClassifierResponseHandler.generate_greeting",
            input={"user_message": "hi there"},
        )

        # Verify generation.update was called
        assert mock_generation.update.called

        # Verify output was yielded correctly
        assert output_tokens == ["Hello", "! ", "How ", "can ", "I ", "help?"]

    @pytest.mark.asyncio
    async def test_trace_goodbye_generation_captures_output(
        self, monkeypatch: pytest.MonkeyPatch, mock_langfuse_client
    ):
        mock_client, mock_generation = mock_langfuse_client

        monkeypatch.setattr(
            "rasa.builder.telemetry.langfuse_integration.message_classifier_langfuse_telemetry.is_langfuse_available",
            lambda: True,
        )
        monkeypatch.setattr(
            "rasa.builder.telemetry.langfuse_integration.message_classifier_langfuse_telemetry.langfuse.get_client",
            lambda: mock_client,
        )

        # Create a mock handler
        handler = MagicMock()
        handler._user_message = "bye"
        handler._generation_usage = UsageStatistics(
            model="gpt-4o-mini",
            prompt_tokens=15,
            completion_tokens=8,
            total_tokens=23,
            cached_prompt_tokens=0,
            input_token_price=0.15,
            output_token_price=0.6,
            cached_token_price=0.075,
        )

        # Create the actual function to test
        async def mock_stream_goodbye(self):
            tokens = ["Goodbye", "! ", "Happy ", "building!"]
            for token in tokens:
                yield MagicMock(content=token)

        decorated = (
            MessageClassifierResponseHandlerLangfuseTelemetry.trace_response_generation(
                "goodbye", max_tokens=30
            )(mock_stream_goodbye)
        )

        # Call the decorated function and collect output
        output_tokens = []
        async for token in decorated(handler):
            output_tokens.append(token.content)

        # Verify Langfuse was called correctly
        mock_client.start_as_current_generation.assert_called_once_with(
            name="MessageClassifierResponseHandler.generate_goodbye",
            input={"user_message": "bye"},
        )

        # Verify generation.update was called
        assert mock_generation.update.called

        # Verify output was yielded correctly
        assert output_tokens == ["Goodbye", "! ", "Happy ", "building!"]
