from typing import Optional, Dict, Any, List

from rasa.core.actions.action import Action
from rasa.core.channels import OutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.dialogue_understanding.stack.frames import ChitChatStackFrame
from rasa.shared.core.constants import ACTION_TRIGGER_CHITCHAT
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event
from rasa.shared.core.trackers import DialogueStateTracker


class ActionTriggerChitchat(Action):
    """Action which triggers a chitchat answer."""

    def name(self) -> str:
        """Return the name of the action."""
        return ACTION_TRIGGER_CHITCHAT

    async def run(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Event]:
        """Run the predicate checks."""
        stack = tracker.stack
        stack.push(ChitChatStackFrame())
        return tracker.create_stack_updated_events(stack)
