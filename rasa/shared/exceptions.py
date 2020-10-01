from typing import Text


class RasaException(Exception):
    """Base exception class for all errors raised by Rasa Open Source."""


class RasaCoreException(RasaException):
    """Basic exception for errors raised by Rasa Core."""


class RasaXTermsError(RasaException):
    """Error in case the user didn't accept the Rasa X terms."""


class YamlSyntaxException(RasaException):
    """Raised when a YAML file can not be parsed properly due to a syntax error."""

    def __init__(self, filename: Text, underlying_yaml_exception: Exception):
        self.underlying_yaml_exception = underlying_yaml_exception
        self.filename = filename

    def __str__(self) -> Text:
        exception_text = (
            f"Failed to read '{self.filename}'. " f"{self.underlying_yaml_exception}"
        )
        exception_text = exception_text.replace(
            'in "<unicode string>"', f'in "{self.filename}"'
        )
        return exception_text
