from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import structlog

from rasa.dialogue_understanding.commands import ErrorCommand

structlogger = structlog.get_logger()


DEFAULT_MESSAGE = (
    "I'm sorry, but your message is too long for me to process. Please keep your "
    "message concise and within a reasonable length."
)


@dataclass
class UserInputExceedsLimitErrorCommand(ErrorCommand):
    """A command to indicate that the bot failed to handle the dialogue."""

    message: str = DEFAULT_MESSAGE
    """Message uttered to the user"""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ErrorCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return UserInputExceedsLimitErrorCommand(DEFAULT_MESSAGE)
