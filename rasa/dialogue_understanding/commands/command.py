from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, List

import rasa.shared.utils.common
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker


@dataclass
class Command:
    """A command that can be executed on a tracker."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the command."""
        raise NotImplementedError()

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        raise NotImplementedError()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Command:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        raise NotImplementedError()

    @staticmethod
    def command_from_json(data: Dict[str, Any]) -> "Command":
        """Converts a dictionary to a command object.

        First, resolves the command type and then converts the dictionary to
        the corresponding command object.

        Args:
            data: The dictionary to convert.

        Returns:
        The converted command object.
        """
        for cls in rasa.shared.utils.common.all_subclasses(Command):
            try:
                if data.get("command") == cls.command():
                    return cls.from_dict(data)
            except NotImplementedError:
                # we don't want to raise an error if the frame type is not
                # implemented, as this is ok to be raised by an abstract class
                pass
        else:
            raise ValueError(f"Unknown command type: {data}")

    def as_dict(self) -> Dict[str, Any]:
        """Converts the command to a dictionary.

        Returns:
            The converted dictionary.
        """
        data = dataclasses.asdict(self)
        data["command"] = self.command()
        return data

    def run_command_on_tracker(
        self,
        tracker: DialogueStateTracker,
        all_flows: FlowsList,
        original_tracker: DialogueStateTracker,
    ) -> List[Event]:
        """Runs the command on the tracker.

        Args:
            tracker: The tracker to run the command on.
            all_flows: All flows in the assistant.
            original_tracker: The tracker before any command was executed.

        Returns:
            The events to apply to the tracker.
        """
        raise NotImplementedError()
