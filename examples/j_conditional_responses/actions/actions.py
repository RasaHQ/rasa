# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet, UserUttered

from datetime import datetime


def get_intents_from_tracker(tracker: Tracker) -> List[Text]:
    return [
        event.get("parse_data", {}).get("intent", {}).get("name")
        for event in tracker.events_after_latest_restart()
        if event.get("event", "") == "user"
    ]


class SlotMappingAction(Action):

    def name(self) -> Text:
        return "slot_mapping_action"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:

        current_time = datetime.now().strftime("%A, %d. %B %Y %I:%M%p")

        current_daytime = "noon"
        if datetime.now().hour < 10:
            current_daytime = "morning"
        elif datetime.now().hour > 17:
            current_daytime = "evening"

        num_ask_what_can_do_int = len([
            intent for intent in get_intents_from_tracker(tracker)
            if intent == "ask_what_can_do"
        ])
        num_ask_what_can_do = str(num_ask_what_can_do_int) if num_ask_what_can_do_int < 4 else "4+"

        return [
            SlotSet("current_time", current_time),
            SlotSet("current_daytime", current_daytime),
            SlotSet("num_ask_what_can_do", num_ask_what_can_do),
        ]
