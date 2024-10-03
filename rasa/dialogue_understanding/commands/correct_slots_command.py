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
)
from rasa.shared.core.events import Event
from rasa.shared.core.flows.flow_step import FlowStepWithFlowReference
from rasa.shared.core.flows.steps.constants import END_STEP
from rasa.shared.core.flows.steps.continuation import ContinueFlowStep
from rasa.shared.core.flows import FlowsList
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

        A slot is reset only if the `collect` step it gets filled by
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
            collect_step.collect not in proposed_slots
            or collect_step.ask_before_filling
            for flow in all_flows.underlying_flows
            for collect_step in flow.get_collect_steps()
        )

    @staticmethod
    def find_earliest_updated_collect_info(
        updated_slots: Dict[str, Any],
        all_flows: FlowsList,
        tracker: DialogueStateTracker,
    ) -> Optional[FlowStepWithFlowReference]:
        """Find the earliest collect information step that fills one of the slots.

        When we update slots, we need to reset a flow to the question when the slot
        was asked. This function finds the earliest collect information step that
        fills one of the slots - with the idea being that we afterwards go through
        the other updated slots.

        Args:
            updated_slots: The slots that were updated.
            all_flows: All flows.
            tracker: The dialogue state tracker.

        Returns:
        The earliest collect information step that fills one of the slots and
        the  flow id of that step.
        """

        collect_steps = utils.previous_collect_steps_for_active_flow(tracker, all_flows)

        for collect_step, flow_id in collect_steps:
            if collect_step.collect in updated_slots:
                return FlowStepWithFlowReference(collect_step, flow_id)

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
    def determine_index_for_new_correction_frame(
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
        proposed_slots: Dict[str, Any],
        all_flows: FlowsList,
        tracker: DialogueStateTracker,
    ) -> Optional[CorrectionPatternFlowStackFrame]:
        """Creates a correction frame.

        Args:
            proposed_slots: The proposed slots.
            all_flows: All flows in the assistant.
            tracker: The dialogue state tracker.

        Returns:
            The correction frame.
        """
        # check if all corrected slots have ask_before_filling=True
        # if this is a case, we are not correcting a value but we
        # are resetting the slots and jumping back to the first question
        is_reset_only = cls.are_all_slots_reset_only(proposed_slots, all_flows)

        earliest_collect = cls.find_earliest_updated_collect_info(
            proposed_slots, all_flows, tracker
        )

        if not earliest_collect and not is_reset_only:
            # if we could not find any step in the flow, where the slots were
            # previously set, and we also don't want to reset the slots, do
            # not correct the slots.
            structlogger.debug(
                "command_executor.skip_correction",
                is_reset_only=is_reset_only,
            )
            return None

        return CorrectionPatternFlowStackFrame(
            is_reset_only=is_reset_only,
            corrected_slots=proposed_slots,
            reset_flow_id=earliest_collect.flow_id if earliest_collect else None,
            reset_step_id=earliest_collect.step.id if earliest_collect else None,
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
        stack = tracker.stack

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
            proposed_slots, all_flows, tracker
        )
        if not correction_frame:
            return []

        insertion_index = self.determine_index_for_new_correction_frame(
            top_flow_frame, stack
        )
        self.end_previous_correction(top_flow_frame, stack)

        stack.push(correction_frame, index=insertion_index)
        return tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        return hash(self.command())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CorrectSlotsCommand):
            return False

        return True
