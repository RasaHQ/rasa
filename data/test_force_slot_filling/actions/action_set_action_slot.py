from typing import Any, Dict
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher


class ActionSetActionSlot(Action):
    def name(self) -> str:
        return "action_set_action_slot"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ):
        return [SlotSet("action_slot", 123)]
