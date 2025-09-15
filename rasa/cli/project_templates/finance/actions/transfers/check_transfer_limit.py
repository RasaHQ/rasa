from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.db import get_account


class CheckTransferLimit(Action):
    def name(self) -> Text:
        return "check_transfer_limit"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        limit_type = tracker.get_slot("transfer_limit_type")

        account = get_account(tracker.sender_id)
        daily_limit = account.daily_limit
        monthly_limit = account.monthly_limit

        transfer_limit = None

        if limit_type == "daily":
            transfer_limit = daily_limit
        elif limit_type == "monthly":
            transfer_limit = monthly_limit

        return [
            SlotSet("transfer_limit", transfer_limit),
            SlotSet("return_value", transfer_limit is not None),
        ]
