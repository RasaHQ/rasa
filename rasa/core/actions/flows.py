from typing import Text, List
import logging
from rasa.core.actions import action
from rasa.core.channels import OutputChannel
from rasa.core.policies.flow_policy import FlowState
from rasa.shared.constants import FLOW_PREFIX

from rasa.shared.core.constants import (
    FLOW_STACK_SLOT,
    FLOW_STATE_SLOT,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import (
    Event,
    SlotSet,
)
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


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
    ) -> List[Event]:
        """Trigger the flow."""
        return [
            SlotSet(FLOW_STACK_SLOT, None),
            SlotSet(FLOW_STATE_SLOT, FlowState(flow_id=self._flow_name).as_dict()),
        ]
