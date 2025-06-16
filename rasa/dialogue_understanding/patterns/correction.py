from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Text

import structlog

from rasa.core.actions import action
from rasa.core.channels import OutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import (
    BaseFlowStackFrame,
    PatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames.dialogue_stack_frame import (
    DialogueStackFrame,
)
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.utils import top_user_flow_frame
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from rasa.shared.core.constants import ACTION_CORRECT_FLOW_SLOT
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.flow import END_STEP, START_STEP, ContinueFlowStep
from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()

FLOW_PATTERN_CORRECTION_ID = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "correction"


@dataclass
class CorrectionPatternFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame which gets added if a slot value is corrected."""

    flow_id: str = FLOW_PATTERN_CORRECTION_ID
    """The ID of the flow."""
    is_reset_only: bool = False
    """Whether the correction is only a reset of the flow.

    This is the case if all corrected slots have `ask_before_filling=True`.
    In this case, we do not set their value directly but rather reset the flow
    to the position where the first question is asked to fill the slot."""
    corrected_slots: Dict[str, Any] = field(default_factory=dict)
    """The slots that were corrected."""
    reset_flow_id: Optional[str] = None
    """The ID of the flow to reset to."""
    reset_step_id: Optional[str] = None
    """The ID of the step to reset to."""
    new_slot_values: List[Any] = field(default_factory=list)
    """The new values for the corrected slots."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_CORRECTION_ID

    @staticmethod
    def from_dict(data: Dict[Text, Any]) -> CorrectionPatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        new_slot_values = [
            val.get("value") for _, val in data["corrected_slots"].items()
        ]

        return CorrectionPatternFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
            is_reset_only=data["is_reset_only"],
            corrected_slots=data["corrected_slots"],
            reset_flow_id=data["reset_flow_id"],
            reset_step_id=data["reset_step_id"],
            new_slot_values=new_slot_values,
        )


class ActionCorrectFlowSlot(action.Action):
    """Action which corrects a slots value in a flow."""

    def __init__(self) -> None:
        """Creates a `ActionCorrectFlowSlot`."""
        super().__init__()

    def name(self) -> Text:
        """Return the flow name."""
        return ACTION_CORRECT_FLOW_SLOT

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Correct the slots."""
        stack = tracker.stack
        if not (top := stack.top()):
            structlogger.warning("action.correct_flow_slot.no_active_flow")
            return []

        if not isinstance(top, CorrectionPatternFlowStackFrame):
            structlogger.warning(
                "action.correct_flow_slot.no_correction_frame",
                top=top,  # no PII
            )
            return []

        events: List[Event] = []
        if top.reset_flow_id:
            updated_stack = reset_stack_on_tracker_to_prior_state(
                top.reset_flow_id, top.reset_step_id or START_STEP, tracker
            )
            events.extend(tracker.create_stack_updated_events(updated_stack))

        events.extend(
            [
                SlotSet(name, value=val.get("value"), filled_by=val.get("filled_by"))
                for name, val in top.corrected_slots.items()
            ]
        )
        return events


def find_previous_state_to_reset_to(
    reset_flow_id: str, reset_step_id: str, tracker: DialogueStateTracker
) -> Optional[DialogueStack]:
    """Find the previous stack state where the reset step was the active stack frame.

    Looks for the most recent state of the stack in the tracker where the top
    flow has the same flow id and step id as the reset flow id and reset step id.
    """
    stack_to_reset_to: Optional[DialogueStack] = None
    for previous_stack_state in tracker.previous_stack_states():
        top_frame = previous_stack_state.top()
        if (
            isinstance(top_frame, BaseFlowStackFrame)
            and top_frame.flow_id == reset_flow_id
            and top_frame.step_id == reset_step_id
        ):
            stack_to_reset_to = previous_stack_state
    return stack_to_reset_to


def slice_of_stack_below_target(
    target_frame: DialogueStackFrame, stack: DialogueStack
) -> List[DialogueStackFrame]:
    """Get all the frames from the current stack up until the target frame.

    The target frame is not included."""
    replacement_stack: List[DialogueStackFrame] = []

    for frame in stack.frames:
        if frame.frame_id == target_frame.frame_id:
            break
        replacement_stack.append(frame)
    return replacement_stack


def slice_of_stack_with_target_and_above(
    target_frame: DialogueStackFrame, stack: DialogueStack
) -> List[DialogueStackFrame]:
    """Get all the frames from the current stack starting at the target frame."""
    replacement_frames: List[DialogueStackFrame] = []
    for frame in reversed(stack.frames):
        replacement_frames.insert(0, frame)  # push to front, since reversed
        if frame.frame_id == target_frame.frame_id:
            break
    return replacement_frames


def set_topmost_flow_frame_to_continue(stack_frames: List[DialogueStackFrame]) -> None:
    """Ensure that the topmost flow frame continues its step.

    This ensures that the topmost step can be re-executed."""
    for frame in reversed(stack_frames):
        if isinstance(frame, BaseFlowStackFrame) and frame.step_id != START_STEP:
            frame.step_id = ContinueFlowStep.continue_step_for_id(frame.step_id)
            break


def create_termination_frames_for_missing_frames(
    new_stack_frames: List[DialogueStackFrame], previous_stack: DialogueStack
) -> List[DialogueStackFrame]:
    """Terminate frames that are part in the previous stack but not in the new stack.

    The frames are terminated by setting them to the END_STEP. This allows
    them to be properly wrapped up.
    """

    reused_frame_ids = {frame.frame_id for frame in new_stack_frames}

    frames_to_terminate: List[DialogueStackFrame] = []
    for frame in reversed(previous_stack.frames):
        if frame.frame_id in reused_frame_ids:
            # this frame already exists in the replacement stack, skip it
            # shouldn't be terminated
            continue
        if (
            isinstance(frame, UserFlowStackFrame)
            and frame.frame_type == FlowStackFrameType.CALL
        ):
            frame.step_id = ContinueFlowStep.continue_step_for_id(END_STEP)
        if isinstance(frame, CollectInformationPatternFlowStackFrame):
            frame.step_id = ContinueFlowStep.continue_step_for_id(END_STEP)
        frames_to_terminate.insert(0, frame)
    return frames_to_terminate


def reset_stack_on_tracker_to_prior_state(
    reset_flow_id: str, reset_step_id: str, tracker: DialogueStateTracker
) -> DialogueStack:
    """Reset the stack on the tracker to the prior state."""
    # assumption is that we have already been at the reset step id in the reset flow
    # at some point before in the tracker. we'll search for that point in time and
    # update the current stack to be as close to that time as possible,
    # jumping back in time.
    stack_to_reset_to: Optional[DialogueStack] = find_previous_state_to_reset_to(
        reset_flow_id, reset_step_id, tracker
    )

    if not stack_to_reset_to:
        structlogger.warning(
            "action.correct_flow_slot.no_frame_found",
            reset_step_id=reset_step_id,
            reset_flow_id=reset_flow_id,
        )
        return tracker.stack

    target_frame = top_user_flow_frame(stack_to_reset_to)

    if not target_frame:
        structlogger.warning(
            "action.correct_flow_slot.no_target_frame_found",
            reset_step_id=reset_step_id,
            reset_flow_id=reset_flow_id,
        )
        return tracker.stack

    current_stack = tracker.stack

    # Approach we use to reset the stack:
    # 1. Search through all past stack states and identify the most recent
    #    state where the stack had the given flow and step id on top. This
    #    stack is the reference stack.
    # 2. On the reference stack, identify the top most user flow frame that
    #    was not started through a flow call. This frame is the reference frame.
    # 3. Create a new stack by stacking the following frames: A) all frames
    #    from the current stack that are strictly below the reference
    #    frame. B) all frames including the reference frame and frames above
    #    it from the reference stack. C) all frames from the current stack
    #    that are strictly above the reference frame and set them to their
    #    end step.
    # 4. This will ensure A) we keep anything from the current stack that is
    #    below the current active user flow. B) We reset the current user
    #    flow to the desired position. C) we wrap up all unfinished frames
    #    above the current user flow from the current state.

    replacement_stack: List[DialogueStackFrame] = []

    # get all the frames from the current stack up until the target frame
    # and put them into the replacement stack. the target frame is not included
    replacement_stack.extend(slice_of_stack_below_target(target_frame, current_stack))

    replacement_stack.extend(
        slice_of_stack_with_target_and_above(target_frame, stack_to_reset_to)
    )

    # ensure that we continue the topmost frames step
    set_topmost_flow_frame_to_continue(replacement_stack)

    # terminate all the frames from the original stack that are not part of the
    # replacement stack. this will add these frames to the replacement stack,
    # but with their step set to END_STEP. this allows them to be properly
    # wrapped up before we head into the correction.
    replacement_stack.extend(
        create_termination_frames_for_missing_frames(replacement_stack, current_stack)
    )
    return DialogueStack(frames=replacement_stack)
