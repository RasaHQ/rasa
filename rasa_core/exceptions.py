from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


class RasaCoreException(Exception):
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


class AgentNotLoaded(RasaCoreException):
    """Raised if there is an error while parsing the story file."""

    def __init__(self, message):
        self.message = message

