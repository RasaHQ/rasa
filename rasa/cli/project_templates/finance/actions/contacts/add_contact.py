from typing import Any, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.db import Contact, add_contact, get_contacts


class AddContact(Action):
    def name(self) -> str:
        return "add_contact"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        contacts = get_contacts(tracker.sender_id)
        name = tracker.get_slot("add_contact_name")
        handle = tracker.get_slot("add_contact_handle")

        if name is None or handle is None:
            return [SlotSet("return_value", "data_not_present")]

        existing_handles = {c.handle for c in contacts}
        if handle in existing_handles:
            return [SlotSet("return_value", "already_exists")]

        new_contact = Contact(name=name, handle=handle)
        add_contact(tracker.sender_id, new_contact)
        return [SlotSet("return_value", "success")]
