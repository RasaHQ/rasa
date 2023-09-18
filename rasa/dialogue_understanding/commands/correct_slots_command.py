from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import structlog

from rasa.dialogue_understanding.commands import Command
from rasa.dialogue_understanding.patterns.correction import (
    FLOW_PATTERN_CORRECTION_ID,
    CorrectionPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    BaseFlowStackFrame,
    UserFlowStackFrame,
)
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.flows.flow import END_STEP, ContinueFlowStep, FlowStep, FlowsList
from rasa.shared.core.trackers import DialogueStateTracker
import rasa.dialogue_understanding.stack.utils as utils

structlogger = structlog.get_logger()


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
        try:
            return CorrectSlotsCommand(
                corrected_slots=[
                    CorrectedSlot(s["name"], value=s["value"])
                    for s in data["corrected_slots"]
                ]
            )
        except KeyError as e:
            raise ValueError(
                f"Missing key when parsing CorrectSlotsCommand: {e}"
            ) from e

    @staticmethod
    def are_all_slots_reset_only(
        proposed_slots: Dict[str, Any], all_flows: FlowsList
    ) -> bool:
        """Checks if all slots are reset only.

        A slot is reset only if the `collect_information` step it gets filled by
        has the `ask_before_filling` flag set to `True`. This means, the slot
        shouldn't be filled if the question isn't asked.

        If such a slot gets corrected, we don't want to correct the slot but
        instead reset the flow to the question where the slot was asked.

        Args:
            proposed_slots: The proposed slots.
            all_flows: All flows in the assistant.

        Returns:
            `True` if all slots are reset only, `False` otherwise.
        """
        return all(
            collect_information_step.collect_information not in proposed_slots
            or collect_information_step.ask_before_filling
            for flow in all_flows.underlying_flows
            for collect_information_step in flow.get_collect_information_steps()
        )

    @staticmethod
    def find_earliest_updated_collect_info(
        user_frame: UserFlowStackFrame,
        updated_slots: List[str],
        all_flows: FlowsList,
    ) -> Optional[FlowStep]:
        """Find the earliest collect information step that fills one of the slots.

        When we update slots, we need to reset a flow to the question when the slot
        was asked. This function finds the earliest collect information step that
        fills one of the slots - with the idea being that we afterwards go through
        the other updated slots.

        Args:
            user_frame: The current user flow frame.
            updated_slots: The slots that were updated.
            all_flows: All flows.

        Returns:
        The earliest collect information step that fills one of the slots.
        """
        flow = user_frame.flow(all_flows)
        step = user_frame.step(all_flows)
        # TODO: DM2 rethink the jumping back behaviour we use for corrections.
        #   Currently we move backwards from a given step id but this could
        #   technically result in wrong results in cases of branches merging
        #   again. If you call this method on a merge step or after it, you'll
        #   get all slots from both branches, although there was only one taken.
        #   You could get this in our verify account flow if you call this on the
        #   final confirm step. I think you'll get the income question even
        #   if you went the not based in ca route. Maybe it's not a problem.
        #   The way to get the exact set of slots would probably simulate the
        #   flow forwards from the starting step. Given the current slots you
        #   could chart the path to the current step id.
        asked_collect_info_steps = flow.previous_collect_information_steps(step.id)

        for collect_info_step in reversed(asked_collect_info_steps):
            if collect_info_step.collect_information in updated_slots:
                return collect_info_step
        return None

    def corrected_slots_dict(self, tracker: DialogueStateTracker) -> Dict[str, Any]:
        """Returns the slots that should be corrected.

        Filters out slots, that are already set to the correct value.

        Args:
            tracker: The tracker.

        Returns:
        A dict with the slots and their values that should be corrected.
        """
        proposed_slots = {}
        for corrected_slot in self.corrected_slots:
            if tracker.get_slot(corrected_slot.name) != corrected_slot.value:
                proposed_slots[corrected_slot.name] = corrected_slot.value
            else:
                structlogger.debug(
                    "command_executor.skip_correction.slot_already_set", command=self
                )
        return proposed_slots

    @staticmethod
    def index_for_correction_frame(
        top_flow_frame: BaseFlowStackFrame, stack: DialogueStack
    ) -> int:
        """Returns the index for the correction frame.

        Args:
            top_flow_frame: The top flow frame.
            stack: The stack.

        Returns:
            The index for the correction frame.
        """
        if top_flow_frame.flow_id != FLOW_PATTERN_CORRECTION_ID:
            # we are not in a correction flow, so we can just push the correction
            # frame on top of the stack
            return len(stack.frames)
        else:
            # we allow the previous correction to finish first before
            # starting the new one. that's why we insert the new correction below
            # the previous one.
            for i, frame in enumerate(stack.frames):
                if frame.frame_id == top_flow_frame.frame_id:
                    return i
            else:
                # we should never get here as we should always find the previous
                # correction frame
                raise ValueError(
                    f"Could not find the previous correction frame "
                    f"{top_flow_frame.frame_id} on the stack {stack}."
                )

    @staticmethod
    def end_previous_correction(
        top_flow_frame: BaseFlowStackFrame, stack: DialogueStack
    ) -> None:
        """Ends the previous correction.

        If the top flow frame is already a correction, we wrap up the previous
        correction before starting the new one. All frames that were added
        after that correction and the correction itself will be set to continue
        at the END step.

        Args:
            top_flow_frame: The top flow frame.
            stack: The stack.
        """
        if top_flow_frame.flow_id != FLOW_PATTERN_CORRECTION_ID:
            # only need to end something if we are already in a correction
            return

        for frame in reversed(stack.frames):
            if isinstance(frame, BaseFlowStackFrame):
                frame.step_id = ContinueFlowStep.continue_step_for_id(END_STEP)
                if frame.frame_id == top_flow_frame.frame_id:
                    break

    @classmethod
    def create_correction_frame(
        cls,
        user_frame: Optional[BaseFlowStackFrame],
        proposed_slots: Dict[str, Any],
        all_flows: FlowsList,
    ) -> CorrectionPatternFlowStackFrame:
        """Creates a correction frame.

        Args:
            user_frame: The user frame.
            proposed_slots: The proposed slots.
            all_flows: All flows in the assistant.

        Returns:
            The correction frame.
        """
        if user_frame:
            # check if all corrected slots have ask_before_filling=True
            # if this is a case, we are not correcting a value but we
            # are resetting the slots and jumping back to the first question
            is_reset_only = cls.are_all_slots_reset_only(proposed_slots, all_flows)

            reset_step = cls.find_earliest_updated_collect_info(
                user_frame, proposed_slots, all_flows
            )
            return CorrectionPatternFlowStackFrame(
                is_reset_only=is_reset_only,
                corrected_slots=proposed_slots,
                reset_flow_id=user_frame.flow_id,
                reset_step_id=reset_step.id if reset_step else None,
            )
        else:
            return CorrectionPatternFlowStackFrame(
                corrected_slots=proposed_slots,
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
        stack = DialogueStack.from_tracker(tracker)
        user_frame = utils.top_user_flow_frame(stack)

        top_flow_frame = utils.top_flow_frame(stack)
        if not top_flow_frame:
            # we shouldn't end up here as a correction shouldn't be triggered
            # if we are not in any flow. but just in case we do, we
            # just skip the command.
            structlogger.warning(
                "command_executor.correct_slots.no_active_flow", command=self
            )
            return []

        structlogger.debug("command_executor.correct_slots", command=self)
        proposed_slots = self.corrected_slots_dict(tracker)

        correction_frame = self.create_correction_frame(
            user_frame, proposed_slots, all_flows
        )
        insertion_index = self.index_for_correction_frame(top_flow_frame, stack)
        self.end_previous_correction(top_flow_frame, stack)

        stack.push(correction_frame, index=insertion_index)
        return [SlotSet(DIALOGUE_STACK_SLOT, stack.as_dict())]
