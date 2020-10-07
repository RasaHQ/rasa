from typing import Optional, Text


class RasaException(Exception):
    """Base exception class for all errors raised by Rasa Open Source."""


class RasaCoreException(RasaException):
    """Basic exception for errors raised by Rasa Core."""


class RasaXTermsError(RasaException):
    """Error in case the user didn't accept the Rasa X terms."""


class YamlException(RasaException):
    """Raised if there is an error reading yaml."""

    def __init__(self, filename: Optional[Text] = None):
        """Create exception.

        Args:
            filename: optional file the error occurred in"""
        self.filename = filename


class YamlSyntaxException(YamlException):
    """Raised when a YAML file can not be parsed properly due to a syntax error."""

    def __init__(
        self,
        filename: Optional[Text] = None,
        underlying_yaml_exception: Optional[Exception] = None,
    ):
        self.underlying_yaml_exception = underlying_yaml_exception
        super(YamlSyntaxException, self).__init__(filename)

    def __str__(self) -> Text:
        if self.filename:
            exception_text = f"Failed to read '{self.filename}'."
        else:
            exception_text = f"Failed to read YAML."

        if self.underlying_yaml_exception:
            self.underlying_yaml_exception.warn = None
            self.underlying_yaml_exception.note = None
            exception_text += f" {self.underlying_yaml_exception}"

        if self.filename:
            exception_text = exception_text.replace(
                'in "<unicode string>"', f'in "{self.filename}"'
            )

        exception_text += (
            "\n\nYou can use http://www.yamllint.com/ to validate the "
            "YAML syntax of your file."
        )
        return exception_text
