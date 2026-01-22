from typing import Any, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.db import get_contacts


class ValidatePayee(Action):
    def name(self) -> str:
        return "validate_payee"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        recipient_account = tracker.get_slot("recipient_account")

        if not recipient_account:
            return [SlotSet("is_valid_payee", False)]

        contacts = get_contacts(tracker.sender_id)

        recipient_normalized = recipient_account.lower().strip()
        if recipient_normalized.startswith("@"):
            recipient_normalized = recipient_normalized[1:]

        is_valid = False
        for contact in contacts:
            if contact.name.lower().strip() == recipient_normalized:
                is_valid = True
                break
            contact_handle = contact.handle.lower().strip()
            if contact_handle.startswith("@"):
                contact_handle = contact_handle[1:]
            if contact_handle == recipient_normalized:
                is_valid = True
                break

        return [SlotSet("is_valid_payee", is_valid)]
