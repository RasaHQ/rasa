from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import structlog
from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.utils import filled_slots_for_active_flow
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.flow import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()


@dataclass
class SetSlotCommand(Command):
    """A command to set a slot."""

    name: str
    value: Any

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "set slot"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SetSlotCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        try:
            return SetSlotCommand(name=data["name"], value=data["value"])
        except KeyError as e:
            raise ValueError(f"Missing key when parsing SetSlotCommand: {e}") from e

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
        stack = DialogueStack.from_tracker(tracker)
        slots_so_far = filled_slots_for_active_flow(stack, all_flows)
        if tracker.get_slot(self.name) == self.value:
            # value hasn't changed, skip this one
            structlogger.debug(
                "command_executor.skip_command.slot_already_set", command=self
            )
            return []
        if self.name not in slots_so_far:
            # only fill slots that belong to a collect infos that can be asked
            use_slot_fill = any(
                step.collect_information == self.name and not step.ask_before_filling
                for flow in all_flows.underlying_flows
                for step in flow.get_collect_information_steps()
            )

            if not use_slot_fill:
                structlogger.debug(
                    "command_executor.skip_command.slot_not_asked_for", command=self
                )
                return []

        structlogger.debug("command_executor.set_slot", command=self)
        return [SlotSet(self.name, self.value)]
