from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from rasa.builder.copilot.agent_sdk.agent_copilot import AgentCopilot
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
from rasa.builder.copilot.orchestrated_copilot import OrchestratedCopilot
from rasa.builder.copilot.response_handling.message_classifier_response_handler import (
    MessageClassifierResponseHandler,
)


def _create_mock_classifier_result(
    category: ResponseCategory, raw_response: str = ""
) -> MessageClassifierResult:
    if not raw_response:
        raw_response = f"[{category.value.upper()}]"

    return MessageClassifierResult(
        category=category,
        classification_usage=UsageStatistics(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        ),
        raw_response=raw_response,
    )


@pytest.fixture(autouse=True)
@patch.object(MessageClassifier, "_get_client")
@patch.object(MessageClassifierResponseHandler, "_get_client")
def mock_openai_clients(mock_response_handler_client, mock_classifier_client):
    # Create mock async context manager for MessageClassifier
    mock_classifier_context = AsyncMock()
    mock_classifier_context.__aenter__.return_value = AsyncMock(spec=openai.AsyncOpenAI)
    mock_classifier_context.__aexit__.return_value = None
    mock_classifier_client.return_value = mock_classifier_context

    # Create mock async context manager for MessageClassifierResponseHandler
    mock_handler_context = AsyncMock()
    mock_handler_context.__aenter__.return_value = AsyncMock(spec=openai.AsyncOpenAI)
    mock_handler_context.__aexit__.return_value = None
    mock_response_handler_client.return_value = mock_handler_context


@pytest.fixture
def mock_context():
    return CopilotContext(
        copilot_chat_history=[
            UserChatMessage(
                role="user", content=[{"type": "text", "text": "test message"}]
            )
        ],
        assistant_id="test-assistant",
    )


class TestOrchestratedCopilot:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "category,user_message,requires_full_copilot",
        [
            (ResponseCategory.GREETING_DETECTION, "hi", False),
            (ResponseCategory.COPILOT, "How do I create a flow?", True),
            (ResponseCategory.OUT_OF_SCOPE_DETECTION, "What's the weather?", False),
            (ResponseCategory.GOODBYE_DETECTION, "bye", False),
            (ResponseCategory.ROLEPLAY_DETECTION, "I want to book a flight", False),
            (ResponseCategory.UNCLEAR_INPUT_DETECTION, "asdfkjh", False),
            (ResponseCategory.RASA_INTRODUCTION_DETECTION, "What is Rasa?", False),
            (
                ResponseCategory.COPILOT_INTRODUCTION_DETECTION,
                "How you can help me?",
                False,
            ),
        ],
    )
    async def test_classifier_categories(
        self,
        category: ResponseCategory,
        user_message: str,
        requires_full_copilot: bool,
    ):
        orchestrated_copilot = OrchestratedCopilot()
        mock_classifier_result = _create_mock_classifier_result(category)

        orchestrated_copilot._classifier.classify = AsyncMock(
            return_value=mock_classifier_result
        )

        context = CopilotContext(
            copilot_chat_history=[
                UserChatMessage(
                    role="user",
                    content=[TextContent(type="text", text=user_message)],
                )
            ],
            assistant_logs="",
            assistant_files={},
            tracker_context=None,
        )
        result = await orchestrated_copilot._classifier.classify(context)

        assert result.category == category
        assert result.requires_full_copilot is requires_full_copilot

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "category",
        [
            ResponseCategory.GREETING_DETECTION,
            ResponseCategory.GOODBYE_DETECTION,
            ResponseCategory.OUT_OF_SCOPE_DETECTION,
            ResponseCategory.ROLEPLAY_DETECTION,
            ResponseCategory.UNCLEAR_INPUT_DETECTION,
            ResponseCategory.RASA_INTRODUCTION_DETECTION,
            ResponseCategory.COPILOT_INTRODUCTION_DETECTION,
        ],
    )
    async def test_generate_response_uses_orchestrated_handler(
        self,
        mock_context: CopilotContext,
        category: ResponseCategory,
    ):
        orchestrated_copilot = OrchestratedCopilot()
        mock_classifier_result = _create_mock_classifier_result(category)

        orchestrated_copilot._classifier.classify = AsyncMock(
            return_value=mock_classifier_result
        )

        handler, generation_context = await orchestrated_copilot.generate_response(
            mock_context
        )

        assert isinstance(handler, MessageClassifierResponseHandler)
        assert handler._response_category == category
        assert orchestrated_copilot._classification_usage is not None
        assert orchestrated_copilot._orchestration_handler is handler

    @pytest.mark.asyncio
    async def test_generate_response_uses_agent_copilot_for_technical_questions(
        self,
        mock_context: CopilotContext,
    ):
        orchestrated_copilot = OrchestratedCopilot()
        mock_classifier_result = MessageClassifierResult(
            category=ResponseCategory.COPILOT,
            classification_usage=UsageStatistics(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )

        orchestrated_copilot._classifier.classify = AsyncMock(
            return_value=mock_classifier_result
        )

        # Mock the agent copilot's generate_response
        mock_agent_handler = MagicMock()
        mock_generation_context = MagicMock()
        mock_agent_usage = UsageStatistics(
            prompt_tokens=500, completion_tokens=200, total_tokens=700
        )

        orchestrated_copilot._agent_copilot.generate_response = AsyncMock(
            return_value=(mock_agent_handler, mock_generation_context)
        )
        orchestrated_copilot._agent_copilot._usage_statistics = mock_agent_usage
        orchestrated_copilot._agent_copilot._llm_config = {"model": "gpt-4o"}

        handler, generation_context = await orchestrated_copilot.generate_response(
            mock_context
        )

        assert handler is mock_agent_handler
        assert generation_context is mock_generation_context

        # Verify separate usage fields are stored
        assert orchestrated_copilot._classification_usage.prompt_tokens == 10
        assert orchestrated_copilot._classification_usage.completion_tokens == 5
        assert orchestrated_copilot._agent_usage.prompt_tokens == 500
        assert orchestrated_copilot._agent_usage.completion_tokens == 200

        # Verify usage_statistics returns agent usage (classification is negligible)
        usage = orchestrated_copilot.usage_statistics
        assert usage.prompt_tokens == 500
        assert usage.completion_tokens == 200
        assert usage.total_tokens == 700
        assert orchestrated_copilot._orchestration_handler is None

    def test_extract_user_message_from_context(self):
        orchestrated_copilot = OrchestratedCopilot()
        context = CopilotContext(
            copilot_chat_history=[
                UserChatMessage(
                    role="user", content=[{"type": "text", "text": "first message"}]
                ),
                UserChatMessage(
                    role="user", content=[{"type": "text", "text": "last message"}]
                ),
            ],
            assistant_id="test-assistant",
        )

        user_message = orchestrated_copilot._extract_user_message(context)
        assert user_message == "last message"

    def test_extract_user_message_empty_context(self):
        orchestrated_copilot = OrchestratedCopilot()
        context = CopilotContext(
            copilot_chat_history=[],
            assistant_id="test-assistant",
        )

        user_message = orchestrated_copilot._extract_user_message(context)
        assert user_message == ""

    @pytest.mark.asyncio
    async def test_usage_statistics_aggregates_classification_and_generation(
        self,
        mock_context: CopilotContext,
    ):
        orchestrated_copilot = OrchestratedCopilot()
        mock_classifier_result = MessageClassifierResult(
            category=ResponseCategory.GREETING_DETECTION,
            classification_usage=UsageStatistics(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )

        orchestrated_copilot._classifier.classify = AsyncMock(
            return_value=mock_classifier_result
        )

        handler, _ = await orchestrated_copilot.generate_response(mock_context)

        # Simulate generation usage being set after streaming
        handler._generation_usage = UsageStatistics(
            prompt_tokens=20, completion_tokens=10, total_tokens=30
        )

        usage = orchestrated_copilot.usage_statistics

        # Should aggregate both classification and generation
        assert usage.prompt_tokens == 30  # 10 + 20
        assert usage.completion_tokens == 15  # 5 + 10
        assert usage.total_tokens == 45  # 15 + 30

    @pytest.mark.asyncio
    async def test_usage_statistics_only_classification_for_template_responses(
        self,
        mock_context: CopilotContext,
    ):
        orchestrated_copilot = OrchestratedCopilot()
        mock_classifier_result = MessageClassifierResult(
            category=ResponseCategory.OUT_OF_SCOPE_DETECTION,
            classification_usage=UsageStatistics(
                prompt_tokens=10, completion_tokens=5, total_tokens=15
            ),
        )

        orchestrated_copilot._classifier.classify = AsyncMock(
            return_value=mock_classifier_result
        )

        handler, _ = await orchestrated_copilot.generate_response(mock_context)

        usage = orchestrated_copilot.usage_statistics

        # Should only have classification usage (no generation for templates)
        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 5
        assert usage.total_tokens == 15

    def test_initial_usage_statistics_empty(self):
        orchestrated_copilot = OrchestratedCopilot()
        usage = orchestrated_copilot.usage_statistics
        assert usage.prompt_tokens is None
        assert usage.completion_tokens is None
        assert usage.total_tokens is None

    def test_initial_llm_config_empty(self):
        orchestrated_copilot = OrchestratedCopilot()
        config = orchestrated_copilot.llm_config
        assert config == {}

    def test_has_classifier_instance(self):
        orchestrated_copilot = OrchestratedCopilot()
        assert orchestrated_copilot._classifier is not None
        assert isinstance(orchestrated_copilot._classifier, MessageClassifier)

    def test_has_agent_copilot_instance(self):
        orchestrated_copilot = OrchestratedCopilot()
        assert orchestrated_copilot._agent_copilot is not None
        assert isinstance(orchestrated_copilot._agent_copilot, AgentCopilot)
