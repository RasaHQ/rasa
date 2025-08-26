from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.database import Database


class ActionRemovePayee(Action):
    def name(self) -> Text:
        return "action_remove_payee"

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
        if not payee:
            dispatcher.utter_message(
                text=f"I'm sorry, but I couldn't find a payee named '{payee_name}'"
            )
            return [SlotSet("payee_removed", False)]

        # Remove the payee
        success = db.remove_payee(payee_name, int(user["id"]))

        if success:
            dispatcher.utter_message(
                text=f"Payee {payee_name} has been removed successfully."
            )
            return [SlotSet("payee_removed", True)]
        else:
            dispatcher.utter_message(text="Failed to remove payee. Please try again.")
            return [SlotSet("payee_removed", False)]
