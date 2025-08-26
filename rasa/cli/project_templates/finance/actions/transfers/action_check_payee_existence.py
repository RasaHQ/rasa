from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.database import Database


class ActionCheckPayeeExistence(Action):
    def name(self) -> Text:
        return "action_check_payee_existence"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        name = tracker.get_slot("name")
        payee_name = tracker.get_slot("payee_name")

        db = Database()

        # Get user information
        user = db.get_user_by_name(name)
        if not user:
            dispatcher.utter_message(text="User not found.")
            return []

        # Check if payee exists
        payee = db.get_payee_by_name_and_user(payee_name, int(user["id"]))

        if payee:
            return [SlotSet("payee_exists", True)]

        dispatcher.utter_message(
            text=f"{payee_name} is not an authorised payee. Let's add them!"
        )
        return [SlotSet("payee_exists", False)]
