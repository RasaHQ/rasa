from rasa.builder.copilot.message_classifier.message_classifier import (
    MessageClassifier,
)
from rasa.builder.copilot.message_classifier.models import MessageClassifierResult
from rasa.builder.copilot.models import ResponseCategory, UsageStatistics


class TestMessageClassifierResult:
    def test_message_classifier_result_creation(self):
        category = ResponseCategory.GREETING_DETECTION
        classification_usage = UsageStatistics(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        result = MessageClassifierResult(
            category=category,
            classification_usage=classification_usage,
        )

        assert result.category == category
        assert result.classification_usage == classification_usage

    def test_requires_full_copilot_true(self):
        classification_usage = UsageStatistics(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )
        result = MessageClassifierResult(
            category=ResponseCategory.COPILOT,
            classification_usage=classification_usage,
        )

        assert result.requires_full_copilot is True

    def test_requires_full_copilot_false(self):
        classification_usage = UsageStatistics(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )

        for category in MessageClassifier.CLASSIFIER_CATEGORIES:
            if category == ResponseCategory.COPILOT:
                continue

            result = MessageClassifierResult(
                category=category,
                classification_usage=classification_usage,
            )
            assert result.requires_full_copilot is False

    def test_requires_full_copilot_defensive_unknown_category(self):
        classification_usage = UsageStatistics(
            prompt_tokens=10, completion_tokens=5, total_tokens=15
        )

        # Create result with a category not in CLASSIFIER_CATEGORIES
        result = MessageClassifierResult(
            category=ResponseCategory.REASONING,
            classification_usage=classification_usage,
        )

        # Should default to requiring full copilot for safety
        assert result.requires_full_copilot is True
