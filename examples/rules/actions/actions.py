from typing import Dict, Text, List

from rasa_sdk import Tracker
from rasa_sdk.events import EventType
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk import Action
from rasa_sdk.events import SlotSet


class ActionSwitchFAQ(Action):
    def name(self) -> Text:
        return "action_switch_faq"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        return [SlotSet("detailed_faq", not tracker.get_slot("detailed_faq"))]


class ValidateSlots(Action):
    def name(self) -> Text:
        """Unique identifier of the form"""

        return "action_validate_loop_q_form"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict
    ) -> List[EventType]:
        dispatcher.utter_message("validate_some_slot")
        return [SlotSet("some_slot", "sdk")]
