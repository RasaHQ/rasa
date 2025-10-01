from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.db import get_contacts, write_contacts


class RemoveContact(Action):
    def name(self) -> str:
        return "remove_contact"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:
        contacts = get_contacts(tracker.sender_id)
        handle = tracker.get_slot("remove_contact_handle")

        if handle is not None:
            contact_indices_with_handle = [
                i for i, c in enumerate(contacts) if c.handle == handle
            ]
            if len(contact_indices_with_handle) == 0:
                return [SlotSet("return_value", "not_found")]
            else:
                removed_contact = contacts.pop(contact_indices_with_handle[0])
                write_contacts(tracker.sender_id, contacts)
                return [
                    SlotSet("return_value", "success"),
                    SlotSet("remove_contact_name", removed_contact.name),
                ]

        else:
            return [SlotSet("return_value", "missing_handle")]
