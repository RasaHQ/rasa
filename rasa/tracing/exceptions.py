from rasa.shared.exceptions import RasaException


class InvalidLangfuseConfigException(RasaException):
    """Raised when a Langfuse config is invalid."""

    def __init__(self, message: str) -> None:
        """Initializes the exception."""
        self.message = message

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return self.message


class DuplicateTracingConfigException(RasaException):
    """Raised when a duplicate tracing config is found."""

    def __init__(self, message: str) -> None:
        """Initializes the exception."""
        self.message = message

    def __str__(self) -> str:
        """Return a string representation of the exception."""
        return self.message
