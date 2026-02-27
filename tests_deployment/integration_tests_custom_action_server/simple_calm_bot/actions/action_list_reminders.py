from typing import Any, Dict

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionListReminders(Action):
    def name(self) -> str:
        return "action_list_reminders"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ):
        dispatcher.utter_message(
            text="You have 2 reminders: dentist appointment and team meeting."
        )
        return []
