#Old file : nlu/components.py

from typing import Text


class MissingArgumentError(ValueError):
    """Raised when a function is called and not all parameters can be
    filled from the context / config.

    Attributes:
        message -- explanation of which parameter is missing
    """

    def __init__(self, message: Text) -> None:
        super(MissingArgumentError, self).__init__(message)
        self.message = message

    def __str__(self) -> Text:
        return self.message


class UnsupportedLanguageError(Exception):
    """Raised when a component is created but the language is not supported.

    Attributes:
        component -- component name
        language -- language that component doesn't support
    """

    def __init__(self, component: Text, language: Text) -> None:
        self.component = component
        self.language = language

        super(UnsupportedLanguageError, self).__init__(component, language)

    def __str__(self) -> Text:
        return "component {} does not support language {}".format(
            self.component, self.language
        )