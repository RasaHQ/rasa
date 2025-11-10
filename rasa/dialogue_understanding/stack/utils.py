import typing
from typing import List, Optional, Set, Tuple, Type

from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import (
    BaseFlowStackFrame,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import FlowStackFrameType
from rasa.dialogue_understanding.stack.frames.pattern_frame import PatternFlowStackFrame
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.steps.collect import CollectInformationFlowStep
from rasa.shared.core.flows.steps.constants import END_STEP
from rasa.shared.core.flows.steps.continuation import ContinueFlowStep

if typing.TYPE_CHECKING:
    from rasa.shared.core.trackers import DialogueStateTracker


def top_flow_frame(
    dialogue_stack: DialogueStack,
    ignore_collect_information_pattern: bool = True,
    ignore_call_frames: bool = True,
) -> Optional[BaseFlowStackFrame]:
    """Returns the topmost flow frame from the tracker.

    By default, the topmost flow frame is ignored if it is the
    `pattern_collect_information`. This is because the `pattern_collect_information`
    is a special flow frame that is used to collect information from the user
    and commonly, is not what you are looking for when you want the topmost frame.

    Also excludes frames created by a call step. They are treated as if they are
    directly part of the flow that called them.

    Args:
        dialogue_stack: The dialogue stack to use.
        ignore_collect_information_pattern: Whether to ignore the
            `pattern_collect_information` frame.
        ignore_call_frames: Whether to ignore user frames of type `call`

    Returns:
        The topmost flow frame from the tracker. `None` if there
        is no frame on the stack.
    """
    for frame in reversed(dialogue_stack.frames):
        if ignore_collect_information_pattern and isinstance(
            frame, CollectInformationPatternFlowStackFrame
        ):
            continue
        if ignore_call_frames and (
            isinstance(frame, UserFlowStackFrame)
            and frame.frame_type == FlowStackFrameType.CALL
        ):
            continue
        if isinstance(frame, BaseFlowStackFrame):
            return frame
    return None


def top_user_flow_frame(
    dialogue_stack: DialogueStack, ignore_call_and_link_frames: bool = True
) -> Optional[UserFlowStackFrame]:
    """Returns the topmost user flow frame from the tracker.

    User flows are flows that are created by developers of an assistant and
    describe tasks the assistant can fulfil. The topmost user flow frame is
    the topmost frame on the stack for one of these user flows. While looking
    for this topmost user flow, other frame types such as PatternFrames,
    SearchFrames, ChitchatFrames, and user flows started through call steps
    are skipped. The latter are treated as part of the calling user flow.

    Args:
        dialogue_stack: The dialogue stack to use.
        ignore_call_and_link_frames: Whether to ignore user frames of type `call`
            and `link`. By default, these frames are ignored.

    Returns:
    The topmost user flow frame from the tracker.
    """
    for frame in reversed(dialogue_stack.frames):
        if isinstance(frame, UserFlowStackFrame):
            if ignore_call_and_link_frames and (
                frame.frame_type == FlowStackFrameType.CALL
                or frame.frame_type == FlowStackFrameType.LINK
            ):
                continue
            return frame
    return None


def filled_slots_for_active_flow(
    tracker: "DialogueStateTracker", all_flows: FlowsList
) -> Tuple[Set[str], Optional[str]]:
    """Get all slots that have been filled for the 'current user flow'.

    All patterns that sit ontop of that user flow as well as
    flows called by this flow, are also included.

    Any collect information step that is part of a pattern on top of the current
    user flow are also included.

    Args:
        tracker: The tracker to get the filled slots from.
        all_flows: All flows.

    Returns:
    All slots that have been filled for the current flow and the id of the currently
    active flow.
    """
    stack = tracker.stack
    user_frame = top_user_flow_frame(stack)
    active_flow = user_frame.flow_id if user_frame else None

    filled_slots = set()
    for collect_step, _ in previous_collect_steps_for_active_flow(tracker, all_flows):
        filled_slots.add(collect_step.collect)

    return filled_slots, active_flow


def previous_collect_steps_for_active_flow(
    tracker: "DialogueStateTracker", all_flows: FlowsList
) -> List[Tuple[CollectInformationFlowStep, str]]:
    stack = tracker.stack
    user_frame = top_user_flow_frame(stack)

    if not user_frame:
        return []

    collect_steps: List[Tuple[CollectInformationFlowStep, str]] = []

    active_frames = {frame.frame_id for frame in stack.frames}

    for previous_stack in tracker.previous_stack_states():
        active_frame = top_user_flow_frame(previous_stack)
        if not active_frame or active_frame.frame_id != user_frame.frame_id:
            continue

        top_frame = previous_stack.top()
        if not isinstance(top_frame, BaseFlowStackFrame):
            continue

        step = top_frame.step(all_flows)
        if not isinstance(step, CollectInformationFlowStep):
            continue

        if isinstance(top_frame, UserFlowStackFrame):
            collect_steps.append((step, top_frame.flow_id))
        elif (
            isinstance(top_frame, PatternFlowStackFrame)
            and top_frame.frame_id in active_frames
        ):
            # if this is a pattern, it is only relevant if it is still
            # active in the current state of the conversation.
            # completed patterns in the past are not relevant
            collect_steps.append((step, top_frame.flow_id))

    return collect_steps


def user_flows_on_the_stack(dialogue_stack: DialogueStack) -> Set[str]:
    """Get all user flows that are currently on the stack.

    Args:
        dialogue_stack: The dialogue stack.

    Returns:
    All user flows that are currently on the stack.
    """
    return {
        frame.flow_id
        for frame in user_frames_on_the_stack(
            dialogue_stack, ignore_call_and_link_frames=False
        )
    }


def user_frames_on_the_stack(
    dialogue_stack: DialogueStack, ignore_call_and_link_frames: bool = True
) -> List[UserFlowStackFrame]:
    """Get all user frames that are currently on the stack.

    Args:
        dialogue_stack: The dialogue stack.
        ignore_call_and_link_frames: Whether to ignore user frames of type `call`
            and `link`. By default, these frames are ignored.

    Returns:
    All user frames that are currently on the stack.
    """
    return [
        frame
        for frame in dialogue_stack.frames
        if isinstance(frame, UserFlowStackFrame)
        and (
            not ignore_call_and_link_frames
            or (
                frame.frame_type != FlowStackFrameType.CALL
                and frame.frame_type != FlowStackFrameType.LINK
            )
        )
    ]


def end_top_user_flow(stack: DialogueStack) -> DialogueStack:
    """Ends all frames on top of the stack including the topmost user frame.

    Ends all flows until the next user flow is reached. This is useful
    if you want to end all flows that are currently on the stack and
    the user flow that triggered them.

    Args:
        stack: The dialogue stack.
    """
    updated_stack = stack.copy()

    for frame in reversed(updated_stack.frames):
        if isinstance(frame, BaseFlowStackFrame):
            frame.step_id = ContinueFlowStep.continue_step_for_id(END_STEP)
            if isinstance(frame, UserFlowStackFrame):
                break
    return updated_stack


def get_collect_steps_excluding_ask_before_filling_for_active_flow(
    dialogue_stack: DialogueStack, all_flows: FlowsList
) -> Set[str]:
    """Get all collect steps that are part of the current flow.

    Collect steps that have to be asked before filling are not considered.

    Args:
        dialogue_stack: The dialogue stack.
        all_flows: All flows.

    Returns:
        All collect steps that are part of the current active flow,
        excluding the collect steps that have to be asked before filling.
    """
    active_primary_frame = top_user_flow_frame(dialogue_stack)
    any_active_frame = top_user_flow_frame(
        dialogue_stack, ignore_call_and_link_frames=False
    )

    active_flows = []
    if any_active_frame:
        active_flows.append(any_active_frame.flow(all_flows))

    if active_primary_frame and active_primary_frame != any_active_frame:
        active_flows.append(active_primary_frame.flow(all_flows))

    if not active_flows:
        return set()

    return set(
        step.collect
        for active_flow in active_flows
        for step in active_flow.get_collect_steps()
        if not step.ask_before_filling
    )


def is_pattern_active(
    stack: DialogueStack, pattern_type: Type[PatternFlowStackFrame]
) -> bool:
    """Check if the pattern frame of the given type is active."""
    return get_active_pattern_frame(stack, pattern_type) is not None


def get_active_pattern_frame(
    stack: DialogueStack,
    pattern_type: Type[PatternFlowStackFrame],
) -> Optional[PatternFlowStackFrame]:
    """Get the active pattern frame of the given type."""
    for frame in reversed(stack.frames):
        if isinstance(frame, pattern_type):
            return frame
        if isinstance(frame, UserFlowStackFrame):
            return None
    return None
