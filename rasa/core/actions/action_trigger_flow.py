from typing import Any, Dict, Optional, Text, List

import structlog
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.utils import top_user_flow_frame
from rasa.core.actions import action
from rasa.core.channels import OutputChannel
from rasa.shared.constants import FLOW_PREFIX

from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    ActiveLoop,
    Event,
    FlowInterrupted,
    SlotSet,
)
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger(__name__)


class ActionTriggerFlow(action.Action):
    """Action which triggers a flow by putting it on the dialogue stack."""

    def __init__(self, flow_action_name: Text) -> None:
        """Creates a `ActionTriggerFlow`.

        Args:
            flow_action_name: Name of the flow.
        """
        super().__init__()

        if not flow_action_name.startswith(FLOW_PREFIX):
            raise ValueError(
                f"Flow action name '{flow_action_name}' needs to start with "
                f"'{FLOW_PREFIX}'."
            )

        self._flow_name = flow_action_name[len(FLOW_PREFIX) :]
        self._flow_action_name = flow_action_name

    def name(self) -> Text:
        """Return the flow name."""
        return self._flow_action_name

    def create_events_to_start_flow(self, tracker: DialogueStateTracker) -> List[Event]:
        """Create events to start the flow.

        Args:
            tracker: The tracker to start the flow on.

        Returns:
            The event to start the flow."""
        stack = tracker.stack
        events: List[Event] = []

        frame_type = FlowStackFrameType.REGULAR

        if not stack.is_empty():
            frame_type = FlowStackFrameType.INTERRUPT
            top_user_frame = top_user_flow_frame(stack)
            if top_user_frame is not None:
                events.append(
                    FlowInterrupted(top_user_frame.flow_id, top_user_frame.step_id)
                )

        stack.push(
            UserFlowStackFrame(
                flow_id=self._flow_name,
                frame_type=frame_type,
            )
        )
        return events + tracker.create_stack_updated_events(stack)

    def create_events_to_set_flow_slots(self, metadata: Dict[str, Any]) -> List[Event]:
        """Create events to set the flow slots.

        Set additional slots to prefill information for the flow.

        Args:
            metadata: The metadata to set the slots from.

        Returns:
            The events to set the flow slots.
        """
        slots_to_be_set = metadata.get("slots", {}) if metadata else {}
        return [SlotSet(key, value) for key, value in slots_to_be_set.items()]

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Trigger the flow."""
        events: List[Event] = self.create_events_to_start_flow(tracker)
        events.extend(self.create_events_to_set_flow_slots(metadata))

        if tracker.active_loop_name:
            # end any active loop to ensure we are progressing the started flow
            events.append(ActiveLoop(None))

        return events
