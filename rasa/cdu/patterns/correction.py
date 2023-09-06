from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Text, List, Optional
from rasa.cdu.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.cdu.stack.dialogue_stack import DialogueStack
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
from rasa.cdu.stack.frames import (
    BaseFlowStackFrame,
    DialogueStackFrame,
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
    flow_id: str = FLOW_PATTERN_CORRECTION_ID
    is_reset_only: bool = False
    corrected_slots: Dict[str, Any] = field(default_factory=dict)
    reset_flow_id: Optional[str] = None
    reset_step_id: Optional[str] = None

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

    def as_dict(self) -> Dict[Text, Any]:
        super_dict = super().as_dict()
        super_dict.update(
            {
                "is_reset_only": self.is_reset_only,
                "corrected_slots": self.corrected_slots,
                "reset_flow_id": self.reset_flow_id,
                "reset_step_id": self.reset_step_id,
            }
        )
        return super_dict

    def context_as_dict(
        self, underlying_frames: List[DialogueStackFrame]
    ) -> Dict[Text, Any]:
        super_dict = super().context_as_dict(underlying_frames)
        super_dict.update(
            {
                "is_reset_only": self.is_reset_only,
                "corrected_slots": self.corrected_slots,
                "reset_flow_id": self.reset_flow_id,
                "reset_step_id": self.reset_step_id,
            }
        )
        return super_dict


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

        for i, frame in enumerate(stack.frames):
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
        if len(stack.frames) > i + 1:
            previous_frame = stack.frames[i + 1]
            if isinstance(previous_frame, CollectInformationPatternFlowStackFrame):
                previous_frame.step_id = ContinueFlowStep.continue_step_for_id(END_STEP)

        events: List[Event] = [SlotSet(DIALOGUE_STACK_SLOT, stack.as_dict())]

        events.extend([SlotSet(k, v) for k, v in top.corrected_slots.items()])

        return events
