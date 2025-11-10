from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import structlog

from rasa.dialogue_understanding.patterns.validate_slot import (
    ValidateSlotPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.dialogue_stack_frame import (
    DialogueStackFrame,
)
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames.pattern_frame import PatternFlowStackFrame
from rasa.shared.constants import ACTION_ASK_PREFIX, UTTER_ASK_PREFIX
from rasa.shared.core.events import (
    AgentResumed,
    Event,
    FlowCompleted,
    FlowResumed,
    SlotSet,
)
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
        validate_frames: List to collect validation frames.
        should_break:  whether to break after the first non-SlotSet event.
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


def resume_flow(
    flow_to_resume: str,
    tracker: DialogueStateTracker,
    stack: DialogueStack,
) -> List[Event]:
    """Resumes a flow by reordering frames."""
    applied_events: List[Event] = []

    # Resume existing flow by reordering frames
    frames_to_resume, user_frame_to_resume = collect_frames_to_resume(
        stack, flow_to_resume
    )

    # if the flow is not on the stack, do nothing
    # this should not happen, but just in case
    if user_frame_to_resume is None:
        structlogger.error(
            "resume_flow.no_user_frame_to_resume",
            flow_to_resume=flow_to_resume,
        )
        return []

    # move the frames to the top of the stack, e.g. reorder the frames
    # on the stack
    stack.move_frames_to_top(frames_to_resume)

    # create agent resumed events if the agent frame is now on top of the stack
    agent_stack_frame = next(
        (frame for frame in frames_to_resume if isinstance(frame, AgentStackFrame)),
        None,
    )
    if agent_stack_frame:
        agent_id = agent_stack_frame.agent_id
        applied_events.append(AgentResumed(agent_id, agent_stack_frame.flow_id))

    # Create flow interruption and resumption events
    applied_events.extend(
        [
            # the flow, which was on the stack, is resumed
            FlowResumed(user_frame_to_resume.flow_id, user_frame_to_resume.step_id),
        ]
    )

    return applied_events + tracker.create_stack_updated_events(stack)


def collect_frames_to_resume(
    stack: DialogueStack,
    target_flow_id: str,  # pyright: ignore[reportUndefinedVariable]
) -> Tuple[List[DialogueStackFrame], Optional[UserFlowStackFrame]]:
    """Collect frames that need to be resumed for the target flow.

    Args:
        stack: The stack to collect frames from.
        target_flow_id: The ID of the flow to resume.

    Returns:
        A tuple containing (frames_to_resume, frame_to_resume).
    """
    frames_to_resume: List[DialogueStackFrame] = []
    frame_found = False
    frame_to_resume = None

    for frame in stack.frames:
        if isinstance(frame, UserFlowStackFrame) and (
            frame.frame_type == FlowStackFrameType.REGULAR
            or frame.frame_type == FlowStackFrameType.INTERRUPT
        ):
            if frame.flow_id == target_flow_id:
                frames_to_resume.append(frame)
                frame_to_resume = frame
                frame_found = True
                continue
            elif frame_found:
                break

        if frame_found:
            frames_to_resume.append(frame)

    return list(frames_to_resume), frame_to_resume


def remove_pattern_continue_interrupted_frames(
    stack: DialogueStack,
) -> Tuple[DialogueStack, List[FlowCompleted]]:
    """Remove pattern_continue_interrupted frames from the stack and return events.

    Returns:
        A tuple containing (updated_stack, flow_completed_events)
    """
    from rasa.dialogue_understanding.patterns.continue_interrupted import (
        ContinueInterruptedPatternFlowStackFrame,
    )
    from rasa.dialogue_understanding.stack.utils import (
        is_pattern_active,
    )

    if not is_pattern_active(stack, ContinueInterruptedPatternFlowStackFrame):
        return stack, []

    return _remove_pattern_frames_from_stack(stack)


def remove_pattern_completed_frames(
    stack: DialogueStack,
) -> Tuple[DialogueStack, List[FlowCompleted]]:
    """Remove pattern_completed frames from the stack and return events.

    Returns:
        A tuple containing (updated_stack, flow_completed_events)
    """
    from rasa.dialogue_understanding.patterns.completed import (
        CompletedPatternFlowStackFrame,
    )
    from rasa.dialogue_understanding.stack.utils import (
        is_pattern_active,
    )

    if not is_pattern_active(stack, CompletedPatternFlowStackFrame):
        return stack, []

    return _remove_pattern_frames_from_stack(stack)


def _remove_pattern_frames_from_stack(
    stack: DialogueStack,
) -> Tuple[DialogueStack, List[FlowCompleted]]:
    events: List[FlowCompleted] = []
    top_frame = stack.top()
    while isinstance(top_frame, PatternFlowStackFrame):
        # Create FlowCompleted event for the pattern frame being removed
        events.append(FlowCompleted(top_frame.flow_id, top_frame.step_id))
        # If the top frame is a pattern frame, we need to remove it
        # before continuing with the active user flow frame.
        # This prevents the pattern frame
        # from being left on the stack when the flow is started
        # which would prevent pattern_completed to be triggered
        # once the user flow is completed.
        stack.pop()
        top_frame = stack.top()
    return stack, events
