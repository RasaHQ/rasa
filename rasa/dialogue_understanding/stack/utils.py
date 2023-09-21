from typing import Optional, Set
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames import BaseFlowStackFrame
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import UserFlowStackFrame
from rasa.shared.core.flows.flow import FlowsList


def top_flow_frame(
    dialogue_stack: DialogueStack, ignore_collect_information_pattern: bool = True
) -> Optional[BaseFlowStackFrame]:
    """Returns the topmost flow frame from the tracker.

    By default, the topmost flow frame is ignored if it is the
    `pattern_collect_information`. This is because the `pattern_collect_information`
    is a special flow frame that is used to collect information from the user
    and commonly, is not what you are looking for when you want the topmost frame.

    Args:
        dialogue_stack: The dialogue stack to use.
        ignore_collect_information_pattern: Whether to ignore the
            `pattern_collect_information` frame.

    Returns:
        The topmost flow frame from the tracker. `None` if there
        is no frame on the stack.
    """

    for frame in reversed(dialogue_stack.frames):
        if ignore_collect_information_pattern and isinstance(
            frame, CollectInformationPatternFlowStackFrame
        ):
            continue
        if isinstance(frame, BaseFlowStackFrame):
            return frame
    return None


def top_user_flow_frame(dialogue_stack: DialogueStack) -> Optional[UserFlowStackFrame]:
    """Returns the topmost user flow frame from the tracker.

    A user flow frame is a flow defined by a bot builder. Other frame types
    (e.g. patterns, search frames, chitchat, ...) are ignored when looking
    for the topmost frame.

    Args:
        tracker: The tracker to use.


    Returns:
        The topmost user flow frame from the tracker."""
    for frame in reversed(dialogue_stack.frames):
        if isinstance(frame, UserFlowStackFrame):
            return frame
    return None


def filled_slots_for_active_flow(
    dialogue_stack: DialogueStack, all_flows: FlowsList
) -> Set[str]:
    """Get all slots that have been filled for the 'current user flow'.

    The 'current user flow' is the top-most flow that is user created. All
    patterns that sit ontop of that user flow, are also included. So any
    collect information step that is part of a pattern that is part of the
    current user flow is also included.

    Args:
        tracker: The tracker to get the filled slots from.
        all_flows: All flows.

    Returns:
    All slots that have been filled for the current flow.
    """
    filled_slots = set()

    for frame in reversed(dialogue_stack.frames):
        if not isinstance(frame, BaseFlowStackFrame):
            # we skip all frames that are not flows, e.g. chitchat / search
            # frames, because they don't have slots.
            continue
        flow = frame.flow(all_flows)
        for q in flow.previous_collect_information_steps(frame.step_id):
            filled_slots.add(q.collect_information)

        if isinstance(frame, UserFlowStackFrame):
            # as soon as we hit the first stack frame that is a "normal"
            # user defined flow we stop looking for previously asked collect infos
            # because we only want to ask collect infos that are part of the
            # current flow.
            break

    return filled_slots


def user_flows_on_the_stack(dialogue_stack: DialogueStack) -> Set[str]:
    """Get all user flows that are currently on the stack.

    Args:
        dialogue_stack: The dialogue stack.

    Returns:
        All user flows that are currently on the stack."""
    return {
        f.flow_id for f in dialogue_stack.frames if isinstance(f, UserFlowStackFrame)
    }
