from datetime import datetime
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher


class ActionValidatePaymentDate(Action):
    def name(self) -> Text:
        return "action_validate_payment_date"

    def is_future_date(self, payment_date_str: str, current_date_str: str) -> bool:
        current_date = datetime.strptime(current_date_str, "%d/%m/%Y").date()
        payment_date = datetime.strptime(payment_date_str, "%d/%m/%Y").date()
        return payment_date > current_date

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        current_date = tracker.get_slot("current_date")
        payment_date = tracker.get_slot("payment_date")

        if self.is_future_date(payment_date, current_date):
            return [
                SlotSet("valid_payment_date", True),
                SlotSet("future_payment_date", True),
            ]
        else:
            return [
                SlotSet("valid_payment_date", True),
                SlotSet("future_payment_date", False),
            ]
