from typing import Text


class RasaException(Exception):
    """Base exception class for all errors raised by Rasa."""


class ModelNotFound(RasaException):
    """Raised when a model is not found in the path provided by the user."""


class NoEventsToMigrate(RasaException):
    """Raised when no events to be migrated are found."""


class NoConversationsInTrackerStore(RasaException):
    """Raised when a tracker store does not contain any conversations."""


class NoEventsInTimeRange(RasaException):
    """Raised when a tracker store does not contain events within a given time range."""


class PublishingError(RasaException):
    """Raised when publishing of an event fails.

    Attributes:
        timestamp -- Unix timestamp of the event during which publishing fails.
    """

    def __init__(self, timestamp: float) -> None:
        self.timestamp = timestamp

    def __str__(self) -> Text:
        return str(self.timestamp)

    def __float__(self) -> float:
        return self.timestamp
