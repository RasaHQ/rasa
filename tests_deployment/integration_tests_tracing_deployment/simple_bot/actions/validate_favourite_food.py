from typing import Any, Dict, Text
from rasa_sdk import Tracker, ValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ValidatePredefinedSlots(ValidationAction):
    def validate_favourite_food(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate favourite_food value."""
        if isinstance(slot_value, str):
            capitalised_slot_value = slot_value.capitalize()
            dispatcher.utter_message("Nice! I've always wanted to try out ", capitalised_slot_value)
            return {"favourite_food": capitalised_slot_value}
        else:
            return {"favourite_food": None}
