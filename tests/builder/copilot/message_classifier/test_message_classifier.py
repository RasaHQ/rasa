from unittest.mock import AsyncMock, MagicMock

import pytest

from rasa.builder.copilot.message_classifier.message_classifier import (
    MessageClassifier,
)
from rasa.builder.copilot.message_classifier.models import MessageClassifierResult
from rasa.builder.copilot.models import ResponseCategory, UsageStatistics


def _create_mock_response(content: str, prompt_tokens: int = 10) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=content))]
    mock_response.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=5,
        total_tokens=prompt_tokens + 5,
    )
    return mock_response


@pytest.fixture
def classifier(monkeypatch):
    mock_client = MagicMock()
    monkeypatch.setattr(
        "rasa.builder.copilot.message_classifier.message_classifier.AsyncOpenAI",
        lambda: mock_client,
    )
    return MessageClassifier()


class TestMessageClassifier:
    @pytest.mark.asyncio
    async def test_classify_greeting(self, classifier: MessageClassifier):
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_create_mock_response("[GREETING_DETECTION]")
        )

        result = await classifier.classify("hi")

        assert isinstance(result, MessageClassifierResult)
        assert result.category == ResponseCategory.GREETING_DETECTION
        assert isinstance(result.classification_usage, UsageStatistics)
        assert result.classification_usage.prompt_tokens == 10
        assert result.classification_usage.completion_tokens == 5

    @pytest.mark.asyncio
    async def test_classify_goodbye(self, classifier: MessageClassifier):
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_create_mock_response("[GOODBYE_DETECTION]")
        )
        result = await classifier.classify("bye")
        assert result.category == ResponseCategory.GOODBYE_DETECTION

    @pytest.mark.asyncio
    async def test_classify_copilot(self, classifier: MessageClassifier):
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_create_mock_response("[COPILOT]", prompt_tokens=20)
        )
        result = await classifier.classify("How do I create a flow?")
        assert result.category == ResponseCategory.COPILOT

    @pytest.mark.asyncio
    async def test_classify_out_of_scope(self, classifier: MessageClassifier):
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_create_mock_response(
                "[OUT_OF_SCOPE_DETECTION]", prompt_tokens=15
            )
        )
        result = await classifier.classify("What's the weather?")
        assert result.category == ResponseCategory.OUT_OF_SCOPE_DETECTION

    @pytest.mark.asyncio
    async def test_classify_roleplay(self, classifier: MessageClassifier):
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_create_mock_response("[ROLEPLAY_DETECTION]", prompt_tokens=12)
        )
        result = await classifier.classify("I want to book a flight")
        assert result.category == ResponseCategory.ROLEPLAY_DETECTION

    @pytest.mark.asyncio
    async def test_classify_unclear_input(self, classifier: MessageClassifier):
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_create_mock_response(
                "[UNCLEAR_INPUT_DETECTION]", prompt_tokens=12
            )
        )
        result = await classifier.classify("asdfkjh")
        assert result.category == ResponseCategory.UNCLEAR_INPUT_DETECTION

    @pytest.mark.asyncio
    async def test_classify_knowledge_base_request(self, classifier: MessageClassifier):
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_create_mock_response(
                "[KNOWLEDGE_BASE_ACCESS_REQUESTED]", prompt_tokens=15
            )
        )
        result = await classifier.classify("Show me the knowledge base")
        assert result.category == ResponseCategory.KNOWLEDGE_BASE_ACCESS_REQUESTED

    @pytest.mark.asyncio
    async def test_classify_fallback_on_parse_error(
        self, classifier: MessageClassifier
    ):
        classifier._client.chat.completions.create = AsyncMock(
            return_value=_create_mock_response("I think this is a greeting")
        )

        result = await classifier.classify("hi")

        # Should default to COPILOT when can't parse
        assert result.category == ResponseCategory.COPILOT

    @pytest.mark.asyncio
    async def test_classify_error_handling(self, classifier: MessageClassifier):
        classifier._client.chat.completions.create = AsyncMock(
            side_effect=Exception("API Error")
        )

        result = await classifier.classify("hi")

        # Should return COPILOT as fallback on error
        assert result.category == ResponseCategory.COPILOT
        assert result.classification_usage.prompt_tokens == 0
        assert result.classification_usage.completion_tokens == 0

    def test_parse_category_valid_tokens(self, classifier: MessageClassifier):
        for category in MessageClassifier.CLASSIFIER_CATEGORIES:
            token = f"[{category.value.upper()}]"
            assert classifier._parse_category(token) == category

    def test_parse_category_invalid_token(self, classifier: MessageClassifier):
        assert (
            classifier._parse_category("invalid response") == ResponseCategory.COPILOT
        )
        assert classifier._parse_category("") == ResponseCategory.COPILOT
        assert classifier._parse_category("[UNKNOWN_TOKEN]") == ResponseCategory.COPILOT

    def test_classifier_categories_defined(self, classifier: MessageClassifier):
        expected_categories = {
            ResponseCategory.GREETING_DETECTION,
            ResponseCategory.GOODBYE_DETECTION,
            ResponseCategory.ROLEPLAY_DETECTION,
            ResponseCategory.OUT_OF_SCOPE_DETECTION,
            ResponseCategory.KNOWLEDGE_BASE_ACCESS_REQUESTED,
            ResponseCategory.UNCLEAR_INPUT_DETECTION,
            ResponseCategory.COPILOT,
        }

        assert set(classifier.CLASSIFIER_CATEGORIES) == expected_categories

    def test_token_to_category_mapping(self, classifier: MessageClassifier):
        for category in MessageClassifier.CLASSIFIER_CATEGORIES:
            token = f"[{category.value.upper()}]"
            assert classifier.TOKEN_TO_CATEGORY[token] == category
