from typing import Any, Dict, Text

from rasa_sdk import Tracker, ValidationAction
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ValidateMood(ValidationAction):

    def validate_mood(
        self,
        slot_value: Any,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> Dict[Text, Any]:
        """Validate the mood slot value."""
        latest_message_intent = tracker.get_intent_of_latest_message()

        if latest_message_intent == "positive_mood":
            return {"mood": "positive"}
        elif latest_message_intent == "negative_mood":
            return {"mood": "negative"}
        else:
            return {"mood": None}
