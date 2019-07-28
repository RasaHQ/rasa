from rasa.exceptions import RasaException


class RasaCoreException(RasaException):
    """Basic exception for errors raised by Rasa Core."""


class StoryParseError(RasaCoreException, ValueError):
    """Raised if there is an error while parsing a story file."""

    def __init__(self, message):
        self.message = message


class UnsupportedDialogueModelError(RasaCoreException):
    """Raised when a model is to old to be loaded.

    Attributes:
        message -- explanation of why the model is invalid
    """

    def __init__(self, message, model_version=None):
        self.message = message
        self.model_version = model_version

    def __str__(self):
        return self.message


class AgentNotReady(RasaCoreException):
    """Raised if someone tries to use an agent that is not ready.

    An agent might be created, e.g. without an interpreter attached. But
    if someone tries to parse a message with that agent, this exception
    will be thrown."""

    def __init__(self, message):
        self.message = message
