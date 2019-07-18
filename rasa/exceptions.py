class RasaException(Exception):
    """Base exception class for all errors raised by Rasa."""


class ModelNotFound(RasaException):
    """Raised when a model is not found in the path provided by the user."""
