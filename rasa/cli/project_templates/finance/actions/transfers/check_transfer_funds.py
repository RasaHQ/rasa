import re
from typing import Any, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.db import get_account


class CheckTransferFunds(Action):
    def name(self) -> str:
        return "check_transfer_funds"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        account = get_account(tracker.sender_id)
        amount_of_money = tracker.get_slot("amount_of_money")
        if not amount_of_money:
            has_sufficient_funds = False
        else:
            if isinstance(amount_of_money, str):
                # Remove any non-numeric characters (except for the decimal point)
                amount_of_money = float(re.sub(r"[^0-9.]", "", amount_of_money))
            has_sufficient_funds = account.funds >= amount_of_money
        return [SlotSet("has_sufficient_funds", has_sufficient_funds)]
