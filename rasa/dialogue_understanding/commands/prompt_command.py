from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


@runtime_checkable
class PromptCommand(Protocol):
    """
    A protocol for commands predicted by the LLM model and incorporated into the prompt.
    """

    @classmethod
    def command(cls) -> str:
        """
        Returns the command name.

        This class method should be implemented to return the name of the command.
        """
        ...

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PromptCommand:
        """
        Converts the dictionary to a command.

        This class method should be implemented to create a command instance from the
        given dictionary.
        """
        ...

    def run_command_on_tracker(
        self,
        tracker: DialogueStateTracker,
        all_flows: FlowsList,
        original_tracker: DialogueStateTracker,
    ) -> List[Event]:
        """
        Runs the command on the tracker.

        This method should be implemented to execute the command on the given tracker.
        """
        ...

    def __hash__(self) -> int:
        """
        Returns the hash of the command.

        This method should be implemented to return the hash of the command.
        Useful for comparing commands and storing them in sets.
        """
        ...

    def __eq__(self, other: object) -> bool:
        """
        Compares the command with another object.

        This method should be implemented to compare the command with another object.
        """
        ...

    def to_dsl(self) -> str:
        """
        Converts the command to a DSL string.

        This method should be implemented to convert the command to a DSL string.
        A DSL string is a string representation of the command that is used in the
        prompt.
        """
        ...

    @classmethod
    def from_dsl(cls, match: re.Match, **kwargs: Any) -> Optional[PromptCommand]:
        """
        Converts the regex match to a command.

        This class method should be implemented to create a command instance from the
        given DSL string.
        """
        ...

    @staticmethod
    def regex_pattern() -> str:
        """
        Returns the regex pattern for the command.

        This method should be implemented to return the regex pattern that matches the
        command in the prompt.
        """
        ...
