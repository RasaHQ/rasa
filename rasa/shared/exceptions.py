class RasaException(Exception):
    """Base exception class for all errors raised by Rasa."""


class RasaCoreException(RasaException):
    """Basic exception for errors raised by Rasa Core."""
