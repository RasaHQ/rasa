#Old file : nlu/config.py

from typing import Text


class InvalidConfigError(ValueError):
    """Raised if an invalid configuration is encountered."""

    def __init__(self, message: Text) -> None:
        super(InvalidConfigError, self).__init__(message)
