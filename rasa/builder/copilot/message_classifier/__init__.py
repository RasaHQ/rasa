"""MessageClassifier module.

The MessageClassifier is a lightweight classifier that decides how to handle
user requests before invoking the full agentic copilot. It classifies requests
into categories, and delegates response generation to the response handler layer.
"""

from rasa.builder.copilot.message_classifier.message_classifier import MessageClassifier
from rasa.builder.copilot.message_classifier.models import MessageClassifierResult

__all__ = [
    "MessageClassifier",
    "MessageClassifierResult",
]
