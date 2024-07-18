import json
from typing import Optional, Text, Dict, Any

import jsonschema
from ruamel.yaml.error import (
    MarkedYAMLError,
    MarkedYAMLWarning,
    MarkedYAMLFutureWarning,
)


class RasaException(Exception):
    """Base exception class for all errors raised by Rasa Pro.

    These exceptions result from invalid use cases and will be reported
    to the users, but will be ignored in telemetry.
    """


class RasaCoreException(RasaException):
    """Basic exception for errors raised by Rasa Core."""


class InvalidParameterException(RasaException, ValueError):
    """Raised when an invalid parameter is used."""


class YamlException(RasaException):
    """Raised if there is an error reading yaml."""

    def __init__(self, filename: Optional[Text] = None) -> None:
        """Create exception.

        Args:
        filename: optional file the error occurred in
        """
        self.filename = filename

    def file_error_message(self) -> str:
        if self.filename:
            return f"Error in '{self.filename}'."
        else:
            return "Error found."

    def __str__(self) -> str:
        msg = self.file_error_message()
        if self.__cause__:
            msg += f" {self.__cause__}"
        return msg


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
        exception_text = self.file_error_message()
        exception_text += " Failed to read YAML."

        if self.underlying_yaml_exception:
            if isinstance(
                self.underlying_yaml_exception,
                (MarkedYAMLError, MarkedYAMLWarning, MarkedYAMLFutureWarning),
            ):
                self.underlying_yaml_exception.note = None
            if isinstance(
                self.underlying_yaml_exception,
                (MarkedYAMLWarning, MarkedYAMLFutureWarning),
            ):
                self.underlying_yaml_exception.warn = None
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


class InvalidConfigException(ValueError, RasaException):
    """Raised if an invalid configuration is encountered."""


class UnsupportedFeatureException(RasaCoreException):
    """Raised if a requested feature is not supported."""


class SchemaValidationError(RasaException, jsonschema.ValidationError):
    """Raised if schema validation via `jsonschema` failed."""


class InvalidEntityFormatException(RasaException, json.JSONDecodeError):
    """Raised if the format of an entity is invalid."""

    @classmethod
    def create_from(
        cls, other: json.JSONDecodeError, msg: Text
    ) -> "InvalidEntityFormatException":
        """Creates `InvalidEntityFormatException` from `JSONDecodeError`."""
        return cls(msg, other.doc, other.pos)


class ConnectionException(RasaException):
    """Raised when a connection to a 3rd party service fails.

    It's used by our broker and tracker store classes, when
    they can't connect to services like postgres, dynamoDB, mongo.
    """


class ProviderClientAPIException(RasaException):
    """Raised for errors that occur during API interactions
    with LLM / embedding providers.

    Attributes:
        original_exception (Exception): The original exception that was
            caught and led to this custom exception.
        message: (Optional[str]): Optional explanation of the error.
    """

    def __init__(
        self,
        original_exception: Exception,
        message: Optional[str] = None,
        info: Optional[Dict[Text, Any]] = None,
    ):
        super().__init__(
            f"{message if message is not None else ''}"
            f"\nOriginal error: {original_exception})"
        )
        self.message = message
        self.original_exception = original_exception
        self.info = info

    def __str__(self) -> Text:
        s = f"{self.__class__.__name__}:"
        if self.message is not None:
            s += f"\n{self.message}"
        s += f"\nOriginal error: {self.original_exception}\n"
        if self.info:
            s += f"\nInfo: \n{self.info}\n"
        return s


class ProviderClientValidationError(RasaException):
    """Raised for errors that occur during validation of the API client."""
