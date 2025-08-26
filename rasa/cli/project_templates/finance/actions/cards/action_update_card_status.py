from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.database import Database


class ActionUpdateCardStatus(Action):
    def name(self) -> str:
        return "action_update_card_status"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:
        name = tracker.get_slot("name")
        card_number = tracker.get_slot("card")
        new_status = "inactive"

        db = Database()

        # Get user information
        user = db.get_user_by_name(name)
        if not user:
            dispatcher.utter_message(text="User not found.")
            return []

        # Get card information to verify it belongs to the user
        card = db.get_card_by_number(card_number)
        if not card:
            dispatcher.utter_message(text="I can't find the card within your account.")
            return []

        # Check if card belongs to the user
        try:
            if int(card["user_id"]) != int(user["id"]):
                dispatcher.utter_message(
                    text="That card is not associated with your account."
                )
                return []
        except (ValueError, TypeError, KeyError):
            dispatcher.utter_message(text="Error verifying card ownership.")
            return []

        # Update card status
        success = db.update_card_status(card_number, new_status)

        if success:
            return [SlotSet("card_status", new_status)]
        else:
            # Log the error but don't show technical error to user
            print(f"Error: Failed to update card status for card {card_number}")
            return []
