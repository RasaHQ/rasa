"""Custom exceptions for the prompt-to-bot service."""

from typing import Any, Dict, List, Optional


class PromptToBotError(Exception):
    """Base exception for prompt-to-bot service."""

    pass


class ValidationError(PromptToBotError):
    """Raised when Rasa project validation fails."""

    def __init__(
        self, message: str, validation_logs: Optional[List[Dict[str, Any]]] = None
    ):
        super().__init__(message)
        self.validation_logs = validation_logs or []

    def get_logs(self, log_levels: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Get logs filtered by given log levels."""
        if len(self.validation_logs) == 0:
            return []

        error_log_entries: List[Dict[str, Any]] = []
        for log_entry in self.validation_logs:
            if log_levels is not None and log_entry.get("log_level") in log_levels:
                error_log_entries.append(log_entry)

        return error_log_entries

    def get_error_message_with_logs(
        self, log_levels: Optional[List[str]] = None
    ) -> str:
        """Get the error message with validation logs for specified log levels.

        Args:
            log_levels: List of log levels to include. If None, includes all logs.

        Returns:
            Error message with validation logs appended.
        """
        error_message = str(self)

        # Include all logs when no specific levels are specified
        logs = self.validation_logs if log_levels is None else self.get_logs(log_levels)

        if logs:
            logs_text = "\n".join([str(log) for log in logs])
            error_message += f"\n\nValidation Logs:\n{logs_text}\n"

        return error_message


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
