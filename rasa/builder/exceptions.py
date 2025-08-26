"""Custom exceptions for the prompt-to-bot service."""

from typing import Any, Optional


class PromptToBotError(Exception):
    """Base exception for prompt-to-bot service."""

    pass


class ValidationError(PromptToBotError):
    """Raised when Rasa project validation fails."""

    def __init__(self, message: str, validation_logs: Optional[Any] = None):
        super().__init__(message)
        self.validation_logs = validation_logs


class TrainingError(PromptToBotError):
    """Raised when model training fails."""

    pass


class LLMGenerationError(PromptToBotError):
    """Raised when LLM generation fails."""

    pass


class DocumentRetrievalError(PromptToBotError):
    """Raised when document retrieval fails."""

    pass


class SchemaValidationError(PromptToBotError):
    """Raised when schema validation fails."""

    pass


class AgentLoadError(PromptToBotError):
    """Raised when agent loading fails."""

    pass


class ProjectGenerationError(PromptToBotError):
    """Raised when project generation fails after retries."""

    def __init__(self, message: str, attempts: int):
        super().__init__(f"{message} (failed after {attempts} attempts)")
        self.attempts = attempts
