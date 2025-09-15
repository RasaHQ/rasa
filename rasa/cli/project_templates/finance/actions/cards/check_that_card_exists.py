from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher


class ActionCheckThatCardExists(Action):
    def name(self) -> Text:
        return "check_that_card_exists"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # For example purposes, we assume the card exists; in a real situation,
        # fetch this info from a database or service.
        # card_number = tracker.get_slot("bank_card_number")
        return [SlotSet("return_value", True)]
