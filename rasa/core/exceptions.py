from typing import Optional, Text

from rasa.shared.exceptions import RasaCoreException


class UnsupportedDialogueModelError(RasaCoreException):
    """Raised when a model is too old to be loaded.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message: Text, model_version: Optional[Text] = None) -> None:
        """Initialize message and model_version attributes."""
        self.message = message
        self.model_version = model_version
        super(UnsupportedDialogueModelError, self).__init__()

    def __str__(self) -> Text:
        return self.message


class AgentNotReady(RasaCoreException):
    """Raised if someone tries to use an agent that is not ready.

    An agent might be created, e.g. without an interpreter attached. But
    if someone tries to parse a message with that agent, this exception
    will be thrown."""

    def __init__(self, message: Text) -> None:
        """Initialize message attribute."""
        self.message = message
        super(AgentNotReady, self).__init__()


class ChannelConfigError(RasaCoreException):
    """Raised if a channel is not configured correctly."""


class InvalidTrackerFeaturizerUsageError(RasaCoreException):
    """Raised if a tracker featurizer is incorrectly used."""
