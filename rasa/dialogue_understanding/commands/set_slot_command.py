from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List

import structlog

from rasa.dialogue_understanding.commands.command import Command
from rasa.dialogue_understanding.commands.command_syntax_manager import (
    CommandSyntaxManager,
    CommandSyntaxVersion,
)
from rasa.dialogue_understanding.commands.utils import (
    clean_extracted_value,
    find_default_flows_collecting_slot,
    get_nullable_slot_value,
)
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.utils import (
    get_collect_steps_excluding_ask_before_filling_for_active_flow,
)
from rasa.shared.constants import ROUTE_TO_CALM_SLOT
from rasa.shared.core.constants import SetSlotExtractor
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import SET_SLOT_COMMAND

structlogger = structlog.get_logger()


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
        filter_start_flow_commands,
        get_commands_from_tracker,
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
                "set_slot_command.skip_command.slot_not_in_domain", command=self
            )
            return []

        if slot.has_same_coerced_value(self.value):
            # value hasn't changed, skip this one
            structlogger.debug(
                "set_slot_command.skip_command.slot_already_set", command=self
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
            and not slot.is_builtin
            and self.extractor
            in {
                SetSlotExtractor.LLM.value,
                SetSlotExtractor.COMMAND_PAYLOAD_READER.value,
                SetSlotExtractor.NLU.value,
            }
        ):
            # Get the other predicted flows from the most recent message on the tracker.
            predicted_flows = get_flows_predicted_to_start_from_tracker(tracker)
            if not predicted_flows:
                # If no predicted flows, check for default flows collecting the slot.
                predicted_flows = find_default_flows_collecting_slot(
                    self.name, all_flows
                )
            use_slot_fill = any(
                step.collect == self.name and not step.ask_before_filling
                for flow in all_flows.underlying_flows
                if flow.id in predicted_flows
                for step in flow.get_collect_steps()
            )
            if not use_slot_fill:
                structlogger.debug(
                    "set_slot_command.skip_command.slot_not_asked_for", command=self
                )
                return []

        structlogger.debug("set_slot_command.set_slot", command=self)
        return [
            SlotSet(self.name, slot.coerce_value(self.value), filled_by=self.extractor)
        ]

    def __hash__(self) -> int:
        return hash(self.value) + hash(self.name)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SetSlotCommand):
            return False

        return (
            str(other.value).lower() == str(self.value).lower()
            and other.name == self.name
        )

    def to_dsl(self) -> str:
        """Converts the command to a DSL string."""
        mapper = {
            CommandSyntaxVersion.v1: f"SetSlot({self.name}, {self.value})",
            CommandSyntaxVersion.v2: f"set slot {self.name} {self.value}",
            CommandSyntaxVersion.v3: f"set slot {self.name} {self.value}",
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )

    @classmethod
    def from_dsl(cls, match: re.Match, **kwargs: Any) -> SetSlotCommand:
        """Converts the DSL string to a command."""
        slot_name = str(match.group(1).strip())
        slot_value = clean_extracted_value(match.group(2))
        return SetSlotCommand(name=slot_name, value=get_nullable_slot_value(slot_value))

    @staticmethod
    def regex_pattern() -> str:
        mapper = {
            CommandSyntaxVersion.v1: (
                r"""SetSlot\(['"]?([a-zA-Z_][a-zA-Z0-9_-]*)['"]?, ?['"]?(.*)['"]?\)"""
            ),
            CommandSyntaxVersion.v2: (
                r"""^[\s\W\d]*set slot ['"`]?([a-zA-Z_][a-zA-Z0-9_-]*)['"`]? ['"`]?(.+?)[\W]*$"""  # noqa: E501
            ),
            CommandSyntaxVersion.v3: (
                r"""^[\s\W\d]*set slot ['"`]?([a-zA-Z_][a-zA-Z0-9_-]*)['"`]? ['"`]?(.+?)[\W]*$"""  # noqa: E501
            ),
        }
        return mapper.get(
            CommandSyntaxManager.get_syntax_version(),
            mapper[CommandSyntaxManager.get_default_syntax_version()],
        )
