from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from actions.database import Database


class ActionAskAccount(Action):
    def name(self) -> Text:
        return "action_ask_account"

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

        # Get all accounts for the user
        user_id = int(user["id"])
        accounts = db.get_accounts_by_user(user_id)

        if not accounts:
            dispatcher.utter_message(text="No accounts found for this user.")
            return []

        buttons = [
            {
                "title": f"{account['number']} ({account['type'].title()})",
                "payload": account["number"],
            }
            for account in accounts
        ]
        message = "Which account would you like the balance for?"
        dispatcher.utter_message(text=message, buttons=buttons)

        return []
