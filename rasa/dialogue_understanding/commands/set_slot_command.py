from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import structlog
from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.utils import (
    get_collect_steps_excluding_ask_before_filling_for_active_flow,
)
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import SET_SLOT_COMMAND

structlogger = structlog.get_logger()


class SetSlotExtractor(Enum):
    """The extractors that can set a slot."""

    LLM = "LLM"
    COMMAND_PAYLOAD_READER = "CommandPayloadReader"
    NLU = "NLU"

    def __str__(self) -> str:
        return self.value


def get_flows_predicted_to_start_from_tracker(
    tracker: DialogueStateTracker,
) -> List[str]:
    """Returns the flows that are predicted to start from the current state.

    Args:
        tracker: The tracker to use.

    Returns:
        The flows that are predicted to start from the current state.
    """
    from rasa.dialogue_understanding.processor.command_processor import (
        get_commands_from_tracker,
        filter_start_flow_commands,
    )

    commands = get_commands_from_tracker(tracker)
    return filter_start_flow_commands(commands)


@dataclass
class SetSlotCommand(Command):
    """A command to set a slot."""

    name: str
    value: Any
    extractor: str = SetSlotExtractor.LLM.value

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return SET_SLOT_COMMAND

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SetSlotCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        try:
            return SetSlotCommand(
                name=data["name"],
                value=data["value"],
                extractor=data.get("extractor", SetSlotExtractor.LLM.value),
            )
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
        slot = tracker.slots.get(self.name)
        if slot is None:
            structlogger.debug(
                "command_executor.skip_command.slot_not_in_domain", command=self
            )
            return []

        if slot.has_same_coerced_value(self.value):
            # value hasn't changed, skip this one
            structlogger.debug(
                "command_executor.skip_command.slot_already_set", command=self
            )
            return []

        # Get slots of the active flow
        slots_of_active_flow = (
            get_collect_steps_excluding_ask_before_filling_for_active_flow(
                tracker.stack, all_flows
            )
        )

        # Add slots that are asked in the current collect step. This is needed
        # to include slots that has ask_before_filling set to True.
        top_frame = tracker.stack.top()
        if isinstance(top_frame, CollectInformationPatternFlowStackFrame):
            slots_of_active_flow.add(top_frame.collect)

        if (
            self.name not in slots_of_active_flow
            and self.name != ROUTE_TO_CALM_SLOT
            and self.extractor
            in {
                SetSlotExtractor.LLM.value,
                SetSlotExtractor.COMMAND_PAYLOAD_READER.value,
            }
        ):
            # Get the other predicted flows from the most recent message on the tracker.
            predicted_flows = get_flows_predicted_to_start_from_tracker(tracker)
            use_slot_fill = any(
                step.collect == self.name and not step.ask_before_filling
                for flow in all_flows.underlying_flows
                if flow.id in predicted_flows
                for step in flow.get_collect_steps()
            )
            if not use_slot_fill:
                structlogger.debug(
                    "command_executor.skip_command.slot_not_asked_for", command=self
                )
                return []

        structlogger.debug("command_executor.set_slot", command=self)
        return [SlotSet(self.name, slot.coerce_value(self.value))]

    def __hash__(self) -> int:
        return hash(self.value) + hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SetSlotCommand):
            return False

        return other.value == self.value and other.name == self.name
