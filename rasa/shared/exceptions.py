from typing import Optional, Text


class RasaException(Exception):
    """Base exception class for all errors raised by Rasa Open Source."""


class RasaCoreException(RasaException):
    """Basic exception for errors raised by Rasa Core."""


class RasaXTermsError(RasaException):
    """Error in case the user didn't accept the Rasa X terms."""


class InvalidParameterException(RasaException, ValueError):
    """Raised when an invalid parameter is used."""


class YamlException(RasaException):
    """Raised if there is an error reading yaml."""

    def __init__(self, filename: Optional[Text] = None) -> None:
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
    ) -> None:
        super(YamlSyntaxException, self).__init__(filename)

        self.underlying_yaml_exception = underlying_yaml_exception

    def __str__(self) -> Text:
        if self.filename:
            exception_text = f"Failed to read '{self.filename}'."
        else:
            exception_text = "Failed to read YAML."

        if self.underlying_yaml_exception:
            self.underlying_yaml_exception.warn = None
            self.underlying_yaml_exception.note = None
            exception_text += f" {self.underlying_yaml_exception}"

        if self.filename:
            exception_text = exception_text.replace(
                'in "<unicode string>"', f'in "{self.filename}"'
            )

        exception_text += (
            "\n\nYou can use https://yamlchecker.com/ to validate the "
            "YAML syntax of your file."
        )
        return exception_text


class FileNotFoundException(RasaException, FileNotFoundError):
    """Raised when a file, expected to exist, doesn't exist."""


class FileIOException(RasaException):
    """Raised if there is an error while doing file IO."""
