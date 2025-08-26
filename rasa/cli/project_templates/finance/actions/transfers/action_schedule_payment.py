from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher


class ActionSchedulePayment(Action):
    def name(self) -> Text:
        return "action_schedule_payment"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        # TODO: Should this add the payment to a scheduled_payment table?
        return [SlotSet("payment_scheduled", True)]
