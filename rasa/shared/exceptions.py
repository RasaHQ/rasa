class RasaException(Exception):
    """Base exception class for all errors raised by Rasa."""


class RasaOpenSourceException(RasaException):
    """Basic exception for errors raised by Rasa Open Source."""


class RasaCoreException(RasaOpenSourceException):
    """Basic exception for errors raised by Rasa Core."""


class RasaXTermsError(RasaException):
    """Error in case the user didn't accept the Rasa X terms."""


class ActionNotFoundException(ValueError, RasaOpenSourceException):
    """Raised when an action name could not be found."""
