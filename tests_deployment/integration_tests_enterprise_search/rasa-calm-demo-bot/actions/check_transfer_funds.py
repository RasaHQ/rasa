from typing import Any, Dict
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from .db import get_account
import re


class CheckTransferFunds(Action):

    def name(self) -> str:
        return "check_transfer_funds"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker, domain: Dict[str, Any]):
        account = get_account(tracker.sender_id)
        amount_of_money = tracker.get_slot("transfer_money_amount_of_money")
        if not amount_of_money:
            has_sufficient_funds = False
        else:
            amount_of_money_value = float(re.sub(r"[^0-9.]", "", amount_of_money))
            has_sufficient_funds = account.funds >= amount_of_money_value
        return [SlotSet("transfer_money_has_sufficient_funds", has_sufficient_funds)]
