import logging
from typing import Text

from rasa.shared.exceptions import RasaException

logger = logging.getLogger(__name__)


# TODO: remove/move
class InvalidModelError(RasaException):
    """Raised when a model failed to load.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message: Text) -> None:
        """Initialize message attribute."""
        self.message = message
        super(InvalidModelError, self).__init__(message)

    def __str__(self) -> Text:
        return self.message
