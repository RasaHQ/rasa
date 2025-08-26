from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.database import Database


class ActionAddPayee(Action):
    def name(self) -> Text:
        return "action_add_payee"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        name = tracker.get_slot("name")
        payee_name = tracker.get_slot("payee_name")
        account_number = tracker.get_slot("account_number")
        sort_code = tracker.get_slot("sort_code")
        payee_type = tracker.get_slot("payee_type")
        reference = tracker.get_slot("reference") or ""

        db = Database()

        # Get user information
        user = db.get_user_by_name(name)
        if not user:
            dispatcher.utter_message(text="User not found.")
            return []

        # Add the payee
        success = db.add_payee(
            int(user["id"]),
            payee_name,
            sort_code,
            account_number,
            payee_type,
            reference,
        )

        if success:
            dispatcher.utter_message(
                text=f"Payee {payee_name} has been added successfully."
            )
            return [SlotSet("payee_added", True)]
        else:
            dispatcher.utter_message(text="Failed to add payee. Please try again.")
            return [SlotSet("payee_added", False)]
