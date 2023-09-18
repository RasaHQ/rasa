from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Text, List, Optional
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from rasa.shared.core.constants import (
    DIALOGUE_STACK_SLOT,
)
from rasa.shared.core.flows.flow import (
    START_STEP,
)
from rasa.shared.core.trackers import (
    DialogueStateTracker,
)
import structlog
from rasa.core.actions import action
from rasa.core.channels import OutputChannel
from rasa.dialogue_understanding.stack.frames import (
    BaseFlowStackFrame,
    PatternFlowStackFrame,
)

from rasa.shared.core.constants import (
    ACTION_CORRECT_FLOW_SLOT,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    Event,
    SlotSet,
)
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.core.flows.flow import END_STEP, ContinueFlowStep

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

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return "pattern_correction"

    @staticmethod
    def from_dict(data: Dict[Text, Any]) -> CorrectionPatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return CorrectionPatternFlowStackFrame(
            data["frame_id"],
            step_id=data["step_id"],
            is_reset_only=data["is_reset_only"],
            corrected_slots=data["corrected_slots"],
            reset_flow_id=data["reset_flow_id"],
            reset_step_id=data["reset_step_id"],
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
        stack = DialogueStack.from_tracker(tracker)
        if not (top := stack.top()):
            structlogger.warning("action.correct_flow_slot.no_active_flow")
            return []

        if not isinstance(top, CorrectionPatternFlowStackFrame):
            structlogger.warning(
                "action.correct_flow_slot.no_correction_frame", top=top
            )
            return []

        for idx_of_flow_to_cancel, frame in enumerate(stack.frames):
            if (
                isinstance(frame, BaseFlowStackFrame)
                and frame.flow_id == top.reset_flow_id
            ):
                frame.step_id = (
                    ContinueFlowStep.continue_step_for_id(top.reset_step_id)
                    if top.reset_step_id
                    else START_STEP
                )
                break

        # also need to end any running collect information
        if len(stack.frames) > idx_of_flow_to_cancel + 1:
            frame_ontop_of_user_frame = stack.frames[idx_of_flow_to_cancel + 1]
            # if the frame on top of the user frame is a collect information frame,
            # we need to end it as well
            if isinstance(
                frame_ontop_of_user_frame, CollectInformationPatternFlowStackFrame
            ):
                frame_ontop_of_user_frame.step_id = (
                    ContinueFlowStep.continue_step_for_id(END_STEP)
                )

        events: List[Event] = [SlotSet(DIALOGUE_STACK_SLOT, stack.as_dict())]

        events.extend([SlotSet(k, v) for k, v in top.corrected_slots.items()])

        return events
