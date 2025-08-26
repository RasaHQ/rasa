from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.database import Database


class ActionCheckCardExistence(Action):
    def name(self) -> str:
        return "action_check_card_existence"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:
        name = tracker.get_slot("name")
        card_number = tracker.get_slot("card_number")

        db = Database()

        # Get user information
        user = db.get_user_by_name(name)
        if not user:
            dispatcher.utter_message(text="User not found.")
            return []

        # Get all cards for the user
        user_id = int(user["id"])
        cards = db.get_cards_by_user(user_id)
        card_numbers = [card["number"] for card in cards]

        if card_number in card_numbers:
            return [SlotSet("card_found", True)]
        else:
            return [SlotSet("card_found", False)]
