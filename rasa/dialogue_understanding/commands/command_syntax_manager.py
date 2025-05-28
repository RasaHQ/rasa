from enum import Enum
from typing import Optional

import structlog


class CommandSyntaxVersion(Enum):
    """Defines different syntax versions for commands."""

    v1 = "v1"
    v2 = "v2"
    v3 = "v3"


structlogger = structlog.get_logger()


class CommandSyntaxManager:
    """A class to manage the command syntax version. It is used to set and get the
    command syntax version. This class provides a way to introduce new syntax versions
    for commands in the future. Hence, it is for internal use only.
    """

    _version = None  # Directly store the version as a class attribute

    @classmethod
    def set_syntax_version(cls, version: CommandSyntaxVersion) -> None:
        """Sets the command syntax version on the class itself.
        This method is called only once at the time of LLMCommandGenerator
        initialization to set the command syntax version, which ensures that the command
        syntax version remains consistent throughout the lifetime of the generator.
        """
        if cls._version:
            structlogger.debug(
                "command_syntax_manager.syntax_version_already_set",
                event_info=(
                    f"The command syntax version has already been set. Overwriting "
                    f"the existing version with the new version - {version}."
                ),
            )
        cls._version = version

    @classmethod
    def get_syntax_version(cls) -> Optional[CommandSyntaxVersion]:
        """Fetches the stored command syntax version."""
        return cls._version

    @staticmethod
    def get_default_syntax_version() -> CommandSyntaxVersion:
        """Returns the default command syntax version."""
        return CommandSyntaxVersion.v1

    @classmethod
    def reset_syntax_version(cls) -> None:
        """Resets the command syntax version. Implemented for use in testing."""
        cls._version = None
