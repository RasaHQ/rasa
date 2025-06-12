from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher


class ActionValidateName(Action):

    def name(self) -> Text:
        return "validate_full_name"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        full_name = tracker.get_slot("full_name")
        if full_name and len(full_name.split()) >= 2:
            return [SlotSet("full_name", full_name.upper())]
        else:
            dispatcher.utter_message(text="Please provide a valid full name with at least two words.")
            return [SlotSet("full_name", None)]
