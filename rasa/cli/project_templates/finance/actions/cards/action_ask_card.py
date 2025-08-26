from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from actions.database import Database


class ActionAskCard(Action):
    def name(self) -> Text:
        return "action_ask_card"

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

        # Get all cards for the user
        user_id = int(user["id"])
        cards = db.get_cards_by_user(user_id)

        if not cards:
            dispatcher.utter_message(text="No cards found for this user.")
            return []

        buttons = [
            {
                "content_type": "text",
                "title": f"{i + 1}: x{card['number'][-4:]} ({card['type'].title()})",
                "payload": f"/SetSlots(card_selection={card['number']!s})",
            }
            for i, card in enumerate(cards)
        ]
        message = "Select the card you require assistance with:"
        dispatcher.utter_message(text=message, buttons=buttons)

        return []
