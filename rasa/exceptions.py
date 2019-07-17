class RasaException(Exception):
    """Base exception class for all errors raised by Rasa."""


class ModelNotFound(RasaException):
    """Raised when a model is not found in the path provided by the user."""


class NoModelData(RasaException):
    """Raised if an unpacked model doesn't contain any NLU or Core data."""
