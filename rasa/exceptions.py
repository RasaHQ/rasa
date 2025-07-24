from dataclasses import dataclass
from typing import Any, Dict, Text

from packaging import version

from rasa.constants import MINIMUM_COMPATIBLE_VERSION
from rasa.shared.exceptions import RasaException


@dataclass
class UnsupportedModelVersionError(RasaException):
    """Raised when a model is too old to be loaded.

    Args:
        model_version: the used model version that is not supported and triggered
            this exception
    """

    model_version: Text

    def __str__(self) -> Text:
        minimum_version = version.parse(MINIMUM_COMPATIBLE_VERSION)
        return (
            f"The model version is trained using Rasa Pro {self.model_version} "
            f"and is not compatible with your current installation "
            f"which supports models build with Rasa Pro {minimum_version} "
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


class DetailedRasaException(RasaException):
    """Base class for exceptions that carry an error code and extra context."""

    def __init__(self, *, code: str, event_info: str, **ctx: Any) -> None:
        super().__init__(event_info)
        self.code: str = code
        self.info: str = event_info
        self.ctx: Dict[str, Any] = ctx

    def __str__(self) -> str:
        return self.info


class HealthCheckError(DetailedRasaException):
    """Raised when an error occurs during health checks."""


class EnterpriseSearchPolicyError(DetailedRasaException):
    """Raised when an error occurs in EnterpriseSearchPolicy."""


class ValidationError(DetailedRasaException):
    """Raised when an error occurs during validation."""


@dataclass
class PublishingError(RasaException):
    """Raised when publishing of an event fails.

    Attributes:
        timestamp -- Unix timestamp of the event during which publishing fails.
    """

    timestamp: float

    def __str__(self) -> Text:
        """Returns string representation of exception."""
        return str(self.timestamp)


class ActionLimitReached(RasaException):
    """Raised when predicted action limit is reached."""
