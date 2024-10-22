from typing import Any, Dict

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionListContacts(Action):
    def name(self) -> str:
        return "action_list_contacts"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ):
        dispatcher.utter_message(text="Here are your contacts: John, Jack, Jane.")
        return []
