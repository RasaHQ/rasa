from typing import Any, Dict, Optional, Text, List

import structlog
from rasa.core.actions import action
from rasa.core.channels import OutputChannel
from rasa.cdu.flow_stack import FlowStack, FlowStackFrame, StackFrameType
from rasa.shared.constants import FLOW_PREFIX

from rasa.shared.core.constants import (
    ACTION_CANCEL_FLOW,
    ACTION_CORRECT_FLOW_SLOT,
    FLOW_STACK_SLOT,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActiveLoop,
    Event,
    SlotSet,
)
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.core.flows.flow import END_STEP, START_STEP
from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger(__name__)


class FlowTriggerAction(action.Action):
    """Action which implements and executes the form logic."""

    def __init__(self, flow_action_name: Text) -> None:
        """Creates a `FlowTriggerAction`.

        Args:
            flow_action_name: Name of the flow.
        """
        super().__init__()
        self._flow_name = flow_action_name[len(FLOW_PREFIX) :]
        self._flow_action_name = flow_action_name

    def name(self) -> Text:
        """Return the flow name."""
        return self._flow_action_name

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Trigger the flow."""
        stack = FlowStack.from_tracker(tracker)
        if self._flow_name == "pattern_continue_interrupted":
            frame_type = StackFrameType.REMARK
        elif self._flow_name == "pattern_completed":
            frame_type = StackFrameType.REMARK
        elif tracker.active_loop_name and not stack.is_empty():
            frame_type = StackFrameType.INTERRUPT
        else:
            frame_type = StackFrameType.REGULAR

        stack.push(
            FlowStackFrame(
                flow_id=self._flow_name,
                frame_type=frame_type,
            )
        )

        slots_to_be_set = metadata.get("slots", {}) if metadata else {}
        slot_set_events: List[Event] = [
            SlotSet(key, value) for key, value in slots_to_be_set.items()
        ]

        events: List[Event] = [
            SlotSet(FLOW_STACK_SLOT, stack.as_dict())
        ] + slot_set_events
        if tracker.active_loop_name:
            events.append(ActiveLoop(None))

        return events


class ActionCancelFlow(action.Action):
    """Action which cancels a flow from the stack."""

    def __init__(self) -> None:
        """Creates a `ActionCancelFlow`."""
        super().__init__()

    def name(self) -> Text:
        """Return the flow name."""
        return ACTION_CANCEL_FLOW

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Cancel the flow."""
        stack = FlowStack.from_tracker(tracker)
        if stack.is_empty():
            structlogger.warning("action.cancel_flow.no_active_flow", stack=stack)
            return []

        context = stack.current_context()
        canceled_flow_frames = context.get("canceled_frames", [])

        for canceled_frame_id in canceled_flow_frames:
            for frame in stack.frames:
                if frame.frame_id == canceled_frame_id:
                    frame.step_id = END_STEP
                    break
            else:
                structlogger.warning(
                    "action.cancel_flow.frame_not_found",
                    stack=stack,
                    frame_id=canceled_frame_id,
                )

        return [SlotSet(FLOW_STACK_SLOT, stack.as_dict())]


class ActionCorrectFlowSlot(action.Action):
    """Action which corrects a slots value in a flow."""

    def __init__(self) -> None:
        """Creates a `ActionCancelFlow`."""
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
        stack = FlowStack.from_tracker(tracker)
        if stack.is_empty():
            structlogger.warning("action.correct_flow_slot.no_active_flow", stack=stack)
            return []

        context = stack.current_context()
        corrected_slots = context.get("corrected_slots", {})
        reset_point = context.get("corrected_reset_point", {})

        for frame in stack.frames:
            if frame.flow_id == reset_point.get("id"):
                frame.step_id = reset_point.get("step_id") or START_STEP
                break
        events: List[Event] = [SlotSet(FLOW_STACK_SLOT, stack.as_dict())]

        events.extend([SlotSet(k, v) for k, v in corrected_slots.items()])

        return events
