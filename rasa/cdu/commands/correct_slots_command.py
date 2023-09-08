from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

from rasa.cdu.commands import Command
from rasa.cdu.patterns.correction import (
    FLOW_PATTERN_CORRECTION_ID,
    CorrectionPatternFlowStackFrame,
)
from rasa.cdu.stack.dialogue_stack import DialogueStack
from rasa.cdu.stack.frames.flow_frame import BaseFlowStackFrame, UserFlowStackFrame
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.flow import END_STEP, ContinueFlowStep, FlowStep, FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.cdu.stack.utils import top_flow_frame, top_user_flow_frame

structlogger = structlog.get_logger()


def _find_earliest_updated_collect_info(
    current_user_flow_frame: Optional[UserFlowStackFrame],
    updated_slots: List[str],
    all_flows: FlowsList,
) -> Optional[FlowStep]:
    """Find the earliest collect information step that fills one of the slots.

    When we update slots, we need to reset a flow to the question when the slot
    was asked. This function finds the earliest collect information step that
    fills one of the slots - with the idea being that we afterwards go through
    the other updated slots.

    Args:
        current_user_flow_frame: The current user flow frame.
        updated_slots: The slots that were updated.
        all_flows: All flows.

    Returns:
    The earliest collect information step that fills one of the slots.
    """
    if not current_user_flow_frame:
        return None
    flow = current_user_flow_frame.flow(all_flows)
    step = current_user_flow_frame.step(all_flows)
    asked_collect_info_steps = flow.previously_asked_collect_information(step.id)

    for collect_info_step in reversed(asked_collect_info_steps):
        if collect_info_step.collect_information in updated_slots:
            return collect_info_step
    return None


@dataclass
class CorrectedSlot:
    """A slot that was corrected."""

    name: str
    value: Any


@dataclass
class CorrectSlotsCommand(Command):
    """A command to correct the value of a slot."""

    corrected_slots: List[CorrectedSlot]

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return "correct slot"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CorrectSlotsCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        return CorrectSlotsCommand(
            corrected_slots=[
                CorrectedSlot(s["name"], value=s["value"])
                for s in data["corrected_slots"]
            ]
        )

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
        dialogue_stack = DialogueStack.from_tracker(tracker)
        current_user_frame = top_user_flow_frame(dialogue_stack)

        top_non_collect_info_frame = top_flow_frame(dialogue_stack)
        if not top_non_collect_info_frame:
            # we shouldn't end up here as a correction shouldn't be triggered
            # if there is nothing to correct. but just in case we do, we
            # just skip the command.
            structlogger.warning(
                "command_executor.correct_slots.no_active_flow", command=self
            )
            return []
        structlogger.debug("command_executor.correct_slots", command=self)
        proposed_slots = {}

        for corrected_slot in self.corrected_slots:
            if tracker.get_slot(corrected_slot.name) != corrected_slot.value:
                proposed_slots[corrected_slot.name] = corrected_slot.value
            else:
                structlogger.debug(
                    "command_executor.skip_correction.slot_already_set", command=self
                )

        # check if all corrected slots have ask_before_filling=True
        # if this is a case, we are not correcting a value but we
        # are resetting the slots and jumping back to the first question
        is_reset_only = all(
            collect_information_step.collect_information not in proposed_slots
            or collect_information_step.ask_before_filling
            for flow in all_flows.underlying_flows
            for collect_information_step in flow.get_collect_information_steps()
        )

        reset_step = _find_earliest_updated_collect_info(
            current_user_frame, proposed_slots, all_flows
        )
        correction_frame = CorrectionPatternFlowStackFrame(
            is_reset_only=is_reset_only,
            corrected_slots=proposed_slots,
            reset_flow_id=current_user_frame.flow_id if current_user_frame else None,
            reset_step_id=reset_step.id if reset_step else None,
        )

        if top_non_collect_info_frame.flow_id != FLOW_PATTERN_CORRECTION_ID:
            dialogue_stack.push(correction_frame)
        else:
            # wrap up the previous correction flow
            for i, frame in enumerate(reversed(dialogue_stack.frames)):
                if isinstance(frame, BaseFlowStackFrame):
                    frame.step_id = ContinueFlowStep.continue_step_for_id(END_STEP)
                    if frame.frame_id == top_non_collect_info_frame.frame_id:
                        break

            # push a new correction flow
            dialogue_stack.push(
                correction_frame,
                # we allow the previous correction to finish first before
                # starting the new one
                index=len(dialogue_stack.frames) - i - 1,
            )
        return [SlotSet(DIALOGUE_STACK_SLOT, dialogue_stack.as_dict())]
