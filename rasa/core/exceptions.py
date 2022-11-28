from typing import Text

from rasa.shared.exceptions import RasaCoreException, RasaException


class AgentNotReady(RasaCoreException):
    """Raised if someone tries to use an agent that is not ready.

    An agent might be created, e.g. without an processor attached. But
    if someone tries to parse a message with that agent, this exception
    will be thrown.
    """

    def __init__(self, message: Text) -> None:
        """Initialize message attribute."""
        self.message = message
        super(AgentNotReady, self).__init__()


class ChannelConfigError(RasaCoreException):
    """Raised if a channel is not configured correctly."""


class InvalidTrackerFeaturizerUsageError(RasaCoreException):
    """Raised if a tracker featurizer is incorrectly used."""


class KafkaProducerInitializationError(RasaException):
    """Raised if the Kafka Producer cannot be properly initialized."""
