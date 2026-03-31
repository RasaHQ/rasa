from unittest.mock import AsyncMock, MagicMock, patch

import openai
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


def _create_mock_response(content: str | None, prompt_tokens: int = 10) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=content or ""))]
    mock_response.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=5,
        total_tokens=prompt_tokens + 5,
    )
    return mock_response


@pytest.fixture(autouse=True)
@patch.object(MessageClassifier, "_get_client")
def classifier(mock_get_client: MagicMock) -> None:
    mock_client = AsyncMock(spec=openai.AsyncOpenAI)
    mock_get_client.return_value = AsyncMock()
    mock_get_client.return_value.__aenter__ = AsyncMock(return_value=mock_client)
    mock_get_client.return_value.__aexit__ = AsyncMock(return_value=None)


def _create_context_with_message(message_text: str) -> CopilotContext:
    """Helper to create a CopilotContext with a user message."""
    return CopilotContext(
        copilot_chat_history=[
            UserChatMessage(
                role="user",
                content=[TextContent(type="text", text=message_text)],
            )
        ],
        assistant_logs="",
        assistant_files={},
        tracker_context=None,
    )


class TestMessageClassifier:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "response_content,message_text,expected_category,prompt_tokens,should_error",
        [
            (
                "[GREETING_DETECTION]",
                "hi",
                ResponseCategory.GREETING_DETECTION,
                10,
                False,
            ),
            (
                "[GOODBYE_DETECTION]",
                "bye",
                ResponseCategory.GOODBYE_DETECTION,
                10,
                False,
            ),
            (
                "[COPILOT]",
                "How do I create a flow?",
                ResponseCategory.COPILOT,
                20,
                False,
            ),
            (
                "[OUT_OF_SCOPE_DETECTION]",
                "What's the weather?",
                ResponseCategory.OUT_OF_SCOPE_DETECTION,
                15,
                False,
            ),
            (
                "[ROLEPLAY_DETECTION]",
                "I want to book a flight",
                ResponseCategory.ROLEPLAY_DETECTION,
                12,
                False,
            ),
            (
                "[UNCLEAR_INPUT_DETECTION]",
                "asdfkjh",
                ResponseCategory.UNCLEAR_INPUT_DETECTION,
                12,
                False,
            ),
            (
                "[RASA_INTRODUCTION_DETECTION]",
                "What is Rasa?",
                ResponseCategory.RASA_INTRODUCTION_DETECTION,
                15,
                False,
            ),
            (
                "[COPILOT_INTRODUCTION_DETECTION]",
                "How you can help me?",
                ResponseCategory.COPILOT_INTRODUCTION_DETECTION,
                15,
                False,
            ),
            (
                "[KNOWLEDGE_BASE_ACCESS_REQUESTED]",
                "Show me the knowledge base",
                ResponseCategory.KNOWLEDGE_BASE_ACCESS_REQUESTED,
                15,
                False,
            ),
            (
                "I think this is a greeting",
                "hi",
                ResponseCategory.COPILOT,
                10,
                False,
            ),  # Fallback on parse error
            (
                None,
                "hi",
                ResponseCategory.COPILOT,
                0,
                True,
            ),  # Error handling
        ],
    )
    @patch.object(MessageClassifier, "_call_llm")
    async def test_classify(
        self,
        mock_call_llm: AsyncMock,
        response_content: str | None,
        message_text: str,
        expected_category: ResponseCategory,
        prompt_tokens: int,
        should_error: bool,
    ):
        if should_error:
            mock_call_llm.side_effect = Exception("API Error")
        else:
            mock_call_llm.return_value = _create_mock_response(
                response_content, prompt_tokens=prompt_tokens
            )

        classifier = MessageClassifier()
        context = _create_context_with_message(message_text)
        result = await classifier.classify(context)

        assert isinstance(result, MessageClassifierResult)
        assert result.category == expected_category
        assert isinstance(result.classification_usage, UsageStatistics)
        assert result.classification_usage.prompt_tokens == prompt_tokens
        assert result.classification_usage.completion_tokens == (
            0 if should_error else 5
        )

    @pytest.mark.parametrize(
        "token,expected_category",
        [
            # Valid tokens
            *[
                (f"[{category.value.upper()}]", category)
                for category in MessageClassifier.CLASSIFIER_CATEGORIES
            ],
            # Invalid tokens
            ("invalid response", ResponseCategory.COPILOT),
            ("", ResponseCategory.COPILOT),
            ("[UNKNOWN_TOKEN]", ResponseCategory.COPILOT),
        ],
    )
    def test_parse_category(self, token: str, expected_category: ResponseCategory):
        classifier = MessageClassifier()
        assert classifier._parse_category(token) == expected_category

    def test_classifier_categories_defined(self):
        classifier = MessageClassifier()
        expected_categories = {
            ResponseCategory.GREETING_DETECTION,
            ResponseCategory.GOODBYE_DETECTION,
            ResponseCategory.ROLEPLAY_DETECTION,
            ResponseCategory.OUT_OF_SCOPE_DETECTION,
            ResponseCategory.KNOWLEDGE_BASE_ACCESS_REQUESTED,
            ResponseCategory.UNCLEAR_INPUT_DETECTION,
            ResponseCategory.RASA_INTRODUCTION_DETECTION,
            ResponseCategory.COPILOT_INTRODUCTION_DETECTION,
            ResponseCategory.COPILOT,
        }

        assert set(classifier.CLASSIFIER_CATEGORIES) == expected_categories

    def test_token_to_category_mapping(self):
        classifier = MessageClassifier()
        for category in MessageClassifier.CLASSIFIER_CATEGORIES:
            token = f"[{category.value.upper()}]"
            assert classifier.TOKEN_TO_CATEGORY[token] == category
