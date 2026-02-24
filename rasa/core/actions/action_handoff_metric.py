from typing import Any, Dict, List, Optional

from rasa.core.actions.action import Action
from rasa.core.channels import OutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.core.constants import ACTION_HANDOFF_METRIC
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event
from rasa.shared.core.trackers import DialogueStateTracker


class ActionHandoffMetric(Action):
    """System action used by Studio to track handoff events for containment metrics.

    This action is a no-op and returns no events. Its purpose is to emit a Kafka
    event when called, which Studio uses to calculate containment. Any conversation
    that includes a call to this action is counted as "not contained".
    """

    def name(self) -> str:
        """Return the name of the action."""
        return ACTION_HANDOFF_METRIC

    async def run(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Event]:
        """Execute the action. This is a no-op that returns no events.

        The action execution itself is tracked via ActionExecuted event which is
        automatically added by the dialogue manager.
        """
        return []
