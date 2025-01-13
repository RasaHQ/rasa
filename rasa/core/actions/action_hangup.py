from typing import Any, Dict, List, Optional

from rasa.core.actions.action import Action
from rasa.core.channels import OutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.shared.core.constants import ACTION_HANGUP
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event, SessionEnded
from rasa.shared.core.trackers import DialogueStateTracker


class ActionHangup(Action):
    """Action which hangs up the call."""

    def name(self) -> str:
        """Return the name of the action."""
        return ACTION_HANGUP

    async def run(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Event]:
        """Ask the output channel to hang up the call."""
        await output_channel.hangup(tracker.sender_id)
        return [SessionEnded(metadata={"_reason": "hangup"})]
