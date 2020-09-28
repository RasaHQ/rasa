class RasaException(Exception):
    """Base exception class for all errors raised by Rasa."""


class RasaCoreException(RasaException):
    """Basic exception for errors raised by Rasa Core."""


class RasaXTermsError(RasaException):
    """Error in case the user didn't accept the Rasa X terms."""
