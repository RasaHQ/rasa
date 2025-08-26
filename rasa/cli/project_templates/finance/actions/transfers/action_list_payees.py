from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from actions.database import Database


class ActionListPayees(Action):
    def name(self) -> Text:
        return "action_list_payees"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        name = tracker.get_slot("name")

        db = Database()

        # Get user information
        user = db.get_user_by_name(name)
        if not user:
            dispatcher.utter_message(text="User not found.")
            return []

        # Get payees for the user
        user_id = int(user["id"])
        payees = db.get_payees_by_user(user_id)

        if not payees:
            dispatcher.utter_message(text="You have no payees set up.")
            return []

        payee_names = [payee["name"] for payee in payees]
        if len(payee_names) > 1:
            payees_list = ", ".join(payee_names[:-1]) + " and " + payee_names[-1]
        else:
            payees_list = payee_names[0]

        message = f"You are authorised to transfer money to: {payees_list}"
        dispatcher.utter_message(text=message)

        return []
