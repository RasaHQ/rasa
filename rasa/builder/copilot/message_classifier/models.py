"""Pydantic models for the Copilot Orchestrator."""

from typing import TYPE_CHECKING

from pydantic import BaseModel

from rasa.builder.copilot.models import ResponseCategory, UsageStatistics

if TYPE_CHECKING:
    pass


class MessageClassifierResult(BaseModel):
    """Result from the MessageClassifier classification.

    Attributes:
        category: The ResponseCategory determined by the MessageClassifier.
        classification_usage: Usage stats from the classification LLM call.
    """

    category: ResponseCategory
    classification_usage: UsageStatistics

    class Config:
        use_enum_values = False

    @property
    def requires_full_copilot(self) -> bool:
        """Check if this result requires the full copilot.

        Returns True if:
        - Category is COPILOT (explicit full copilot request)
        - Category is not in CLASSIFIER_CATEGORIES (unexpected/unknown category)

        This ensures we safely fall back to full copilot for any unexpected cases.

        Returns:
            True if full copilot is needed, False for orchestrated responses.
        """
        # Import here to avoid circular dependency
        from rasa.builder.copilot.message_classifier.message_classifier import (
            MessageClassifier,
        )

        # If it's explicitly COPILOT, use full copilot
        if self.category == ResponseCategory.COPILOT:
            return True

        # If it's not in the classifier's known categories, default to
        # full copilot (defensive: handle unexpected classification safely)
        if self.category not in MessageClassifier.CLASSIFIER_CATEGORIES:
            return True

        # Otherwise, it's a known orchestrated response category
        return False
