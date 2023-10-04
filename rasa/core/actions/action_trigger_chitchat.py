from typing import Optional, Dict, Any, List

from rasa.core.actions.action import Action
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import ChitChatStackFrame
from rasa.shared.core.constants import DIALOGUE_STACK_SLOT, ACTION_TRIGGER_CHITCHAT
from rasa.shared.core.events import Event, SlotSet


class ActionTriggerChitchat(Action):
    """Action which triggers a chitchat answer."""

    def name(self) -> str:
        """Return the name of the action."""
        return ACTION_TRIGGER_CHITCHAT

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Event]:
        """Run the predicate checks."""
        dialogue_stack = DialogueStack.from_tracker(tracker)
        dialogue_stack.push(ChitChatStackFrame())
        return [SlotSet(DIALOGUE_STACK_SLOT, dialogue_stack.as_dict())]
