from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.database import Database


class ActionCheckSufficientFunds(Action):
    def name(self) -> Text:
        return "action_check_sufficient_funds"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        name = tracker.get_slot("name")
        account_from = tracker.get_slot("account_from")
        amount = float(tracker.get_slot("amount"))

        db = Database()

        # Get user information
        user = db.get_user_by_name(name)
        if not user:
            dispatcher.utter_message(text="User not found.")
            return []

        # Check if user has sufficient funds
        has_sufficient_funds = db.check_sufficient_funds(
            int(user["id"]), account_from, amount
        )

        if has_sufficient_funds:
            return [SlotSet("sufficient_funds", True)]
        else:
            return [SlotSet("sufficient_funds", False)]
