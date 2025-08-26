from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from actions.database import Database


class ActionCheckBalance(Action):
    def name(self) -> Text:
        return "action_check_balance"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        name = tracker.get_slot("name")
        account_number = tracker.get_slot("account")

        db = Database()

        # Get user information
        user = db.get_user_by_name(name)
        if not user:
            dispatcher.utter_message(text="User not found.")
            return []

        # Get account information
        account = db.get_account_by_user_and_number(int(user["id"]), account_number)
        if not account:
            dispatcher.utter_message(text="Account not found.")
            return []

        current_balance = float(account["balance"])

        message = f"The balance on that account is: ${current_balance}"
        dispatcher.utter_message(text=message)
        return []
