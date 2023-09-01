from typing import Any, Dict, Optional, Text, List

import structlog
from rasa.cdu.conversation_patterns import FLOW_PATTERN_COLLECT_INFORMATION
from rasa.core.actions import action
from rasa.core.channels import OutputChannel
from rasa.cdu.flow_stack import FlowStack, FlowStackFrame, StackFrameType
from rasa.shared.constants import FLOW_PREFIX

from rasa.shared.core.constants import (
    ACTION_CANCEL_FLOW,
    ACTION_CORRECT_FLOW_SLOT,
    FLOW_STACK_SLOT,
    ACTION_CLARIFY_FLOWS,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActiveLoop,
    Event,
    SlotSet,
)
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.core.flows.flow import END_STEP, START_STEP, ContinueFlowStep
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
                    # Setting the stack frame to the end step so it is properly
                    # wrapped up by the flow policy
                    frame.step_id = ContinueFlowStep.continue_step_for_id(END_STEP)
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

        for i, frame in enumerate(stack.frames):
            if frame.flow_id == reset_point.get("id"):
                frame.step_id = (
                    ContinueFlowStep.continue_step_for_id(reset_point.get("step_id"))
                    if reset_point.get("step_id")
                    else START_STEP
                )
                break

        # also need to end any running collect information
        if (
            len(stack.frames) > i + 1
            and stack.frames[i + 1].flow_id == FLOW_PATTERN_COLLECT_INFORMATION
        ):
            stack.frames[i + 1].step_id = ContinueFlowStep.continue_step_for_id(
                END_STEP
            )

        events: List[Event] = [SlotSet(FLOW_STACK_SLOT, stack.as_dict())]

        events.extend([SlotSet(k, v) for k, v in corrected_slots.items()])

        return events


class ActionClarifyFlows(action.Action):
    """Action which clarifies which flow to start."""

    def name(self) -> Text:
        """Return the flow name."""
        return ACTION_CLARIFY_FLOWS

    @staticmethod
    def assemble_options_string(names: List[str]) -> str:
        """Concatenate options to a human-readable string."""
        clarification_message = ""
        for i, name in enumerate(names):
            if i == 0:
                clarification_message += name
            elif i == len(names) - 1:
                clarification_message += f" or {name}"
            else:
                clarification_message += f", {name}"
        return clarification_message

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
            structlogger.warning("action.clarify_flows.no_active_flow", stack=stack)
            return []

        top_frame = stack.top()
        if top_frame is None or top_frame.flow_id != "pattern_clarification":
            structlogger.warning(
                "action.clarify_flows.clarify_frame_not_on_top", stack=stack
            )
            return []

        context = top_frame.context
        if context is None:
            return []
        names = context.get("names", [])
        options_string = self.assemble_options_string(names)
        context["clarification_options"] = options_string
        return [SlotSet(FLOW_STACK_SLOT, stack.as_dict())]
