from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import structlog

from rasa.dialogue_understanding.patterns.validate_slot import (
    ValidateSlotPatternFlowStackFrame,
)
from rasa.shared.constants import ACTION_ASK_PREFIX, UTTER_ASK_PREFIX
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.slots import Slot
from rasa.shared.core.trackers import DialogueStateTracker

if TYPE_CHECKING:
    from rasa.dialogue_understanding.commands import StartFlowCommand

structlogger = structlog.get_logger()


def start_flow_by_name(
    flow_name: str, flows: FlowsList
) -> Optional["StartFlowCommand"]:
    from rasa.dialogue_understanding.commands import StartFlowCommand

    if flow_name in flows.user_flow_ids:
        return StartFlowCommand(flow=flow_name)
    else:
        structlogger.debug(
            "command_parser.start_flow_by_name.invalid_flow_id", flow=flow_name
        )
        return None


def extract_cleaned_options(options_str: str) -> List[str]:
    """Extract and clean options from a string."""
    delimiters = [",", " "]

    for delimiter in delimiters:
        options_str = options_str.replace(delimiter, " ")

    return sorted(
        opt.strip().strip('"').strip("'") for opt in options_str.split() if opt.strip()
    )


def is_none_value(value: str) -> bool:
    """Check if the value is a none value."""
    if not value:
        return True
    return value in {
        "[missing information]",
        "[missing]",
        "None",
        "undefined",
        "null",
    }


def clean_extracted_value(value: str) -> str:
    """Clean up the extracted value from the llm."""
    # replace any combination of single quotes, double quotes, and spaces
    # from the beginning and end of the string
    return value.strip("'\" ")


def get_nullable_slot_value(slot_value: str) -> Union[str, None]:
    """Get the slot value or None if the value is a none value.

    Args:
        slot_value: the value to coerce

    Returns:
        The slot value or None if the value is a none value.
    """
    return slot_value if not is_none_value(slot_value) else None


def initialize_pattern_validate_slot(
    slot: Slot,
) -> Optional[ValidateSlotPatternFlowStackFrame]:
    """Initialize the pattern to validate a slot value."""
    if not slot.requires_validation():
        return None

    validation = slot.validation
    slot_name = slot.name
    return ValidateSlotPatternFlowStackFrame(
        validate=slot_name,
        refill_utter=validation.refill_utter or f"{UTTER_ASK_PREFIX}{slot_name}",  # type: ignore[union-attr]
        refill_action=f"{ACTION_ASK_PREFIX}{slot_name}",
        rejections=validation.rejections,  # type: ignore[union-attr]
    )


def create_validate_frames_from_slot_set_events(
    tracker: DialogueStateTracker,
    events: List[Event],
    validate_frames: List[ValidateSlotPatternFlowStackFrame] = [],
    should_break: bool = False,
) -> Tuple[DialogueStateTracker, List[ValidateSlotPatternFlowStackFrame]]:
    """Process SlotSet events and create validation frames.

    Args:
        tracker: The dialogue state tracker.
        events: List of events to process.
        should_break:  whether or not to break after the first non-SlotSet event.
            if True, break out of the event loop as soon as the first non-SlotSet
            event is encountered.
            if False, continue processing the events until the end.

    Returns:
        Tuple of (updated tracker, list of validation frames).
    """
    for event in events:
        if not isinstance(event, SlotSet):
            if should_break:
                # we want to only process the most recent SlotSet events
                # so we break once we encounter a different event
                break
            continue

        slot = tracker.slots.get(event.key)
        frame = initialize_pattern_validate_slot(slot)

        if frame:
            validate_frames.append(frame)

    return tracker, validate_frames


def find_default_flows_collecting_slot(
    slot_name: str, all_flows: FlowsList
) -> List[str]:
    """Find default flows that have collect steps matching the specified slot name.

    Args:
        slot_name: The name of the slot to search for.
        all_flows: All flows in the assistant.

    Returns:
        List of flow IDs for default flows that collect the specified slot
        without asking before filling.
    """
    return [
        flow.id
        for flow in all_flows.underlying_flows
        if flow.is_rasa_default_flow
        and any(
            step.collect == slot_name and not step.ask_before_filling
            for step in flow.get_collect_steps()
        )
    ]
