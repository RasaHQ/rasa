from typing import Any, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from actions.db import get_contacts


class ListPayees(Action):
    def name(self) -> str:
        return "list_payees"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        contacts = get_contacts(tracker.sender_id)

        if not contacts:
            dispatcher.utter_message(text="You have no payees set up.")
            return []

        payee_names = [contact.name for contact in contacts]
        if len(payee_names) > 1:
            payees_list = ", ".join(payee_names[:-1]) + " and " + payee_names[-1]
        else:
            payees_list = payee_names[0]

        message = f"You are authorised to transfer money to: {payees_list}"
        dispatcher.utter_message(text=message)

        return []
