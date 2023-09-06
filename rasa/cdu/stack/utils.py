from typing import Optional
from rasa.cdu.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.cdu.stack.frames import BaseFlowStackFrame
from rasa.cdu.stack.dialogue_stack import DialogueStack
from rasa.cdu.stack.frames import UserFlowStackFrame


def top_flow_frame(
    dialogue_stack: DialogueStack, ignore_collect_information_pattern: bool = True
) -> Optional[BaseFlowStackFrame]:
    """Returns the topmost flow frame from the tracker.

    Args:
        tracker: The tracker to use.

    Returns:
        The topmost flow frame from the tracker.
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

    Args:
        tracker: The tracker to use.


    Returns:
        The topmost user flow frame from the tracker."""
    for frame in reversed(dialogue_stack.frames):
        if isinstance(frame, UserFlowStackFrame):
            return frame
    return None
