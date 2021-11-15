from typing import Text
from packaging import version

from rasa.shared.exceptions import RasaException
from rasa.constants import MINIMUM_COMPATIBLE_VERSION


class UnsupportedModelVersionError(RasaException):
    """Raised when a model is too old to be loaded."""

    def __init__(self, model_version: Text) -> None:
        """Instantiates a new exception indicating an unsupported model version.

        Args:
          model_version: the used model version that is not supported and triggered
             this exception
        """
        self.model_version = model_version

    def __str__(self) -> Text:
        minimum_version = version.parse(MINIMUM_COMPATIBLE_VERSION)
        return (
            f"The model version is trained using Rasa Open Source {self.model_version} "
            f"and is not compatible with your current installation "
            f"which supports models build with Rasa Open Source {minimum_version} "
            f"or higher. "
            f"This means that you either need to retrain your model "
            f"or revert back to the Rasa version that trained the model "
            f"to ensure that the versions match up again."
        )


class ModelNotFound(RasaException):
    """Raised when a model is not found in the path provided by the user."""


class NoEventsToMigrateError(RasaException):
    """Raised when no events to be migrated are found."""


class NoConversationsInTrackerStoreError(RasaException):
    """Raised when a tracker store does not contain any conversations."""


class NoEventsInTimeRangeError(RasaException):
    """Raised when a tracker store does not contain events within a given time range."""


class MissingDependencyException(RasaException):
    """Raised if a python package dependency is needed, but not installed."""


class PublishingError(RasaException):
    """Raised when publishing of an event fails.

    Attributes:
        timestamp -- Unix timestamp of the event during which publishing fails.
    """

    def __init__(self, timestamp: float) -> None:
        self.timestamp = timestamp
        super(PublishingError, self).__init__()

    def __str__(self) -> Text:
        """Returns string representation of exception."""
        return str(self.timestamp)


class ActionLimitReached(RasaException):
    """Raised when predicted action limit is reached."""
