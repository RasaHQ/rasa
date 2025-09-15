import re
from datetime import datetime
from typing import Any, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.db import Transaction, add_transaction, get_account, write_account

AMOUNT_OF_MONEY_REGEX = re.compile(r"\d*[.,]*\d+")


class ExecuteTransfer(Action):
    def name(self) -> str:
        return "execute_transfer"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        account = get_account(tracker.sender_id)

        recipient = tracker.get_slot("recipient_account")
        amount_of_money = tracker.get_slot("amount_of_money")

        if recipient is None or amount_of_money is None:
            return [SlotSet("transfer_successful", False)]

        # just as a play case
        if recipient == "Jack":
            return [SlotSet("transfer_successful", False)]

        if isinstance(amount_of_money, str):
            amount_of_money = float(re.sub(r"[^0-9.]", "", amount_of_money))
        account.funds -= amount_of_money
        new_transaction = Transaction(
            datetime=datetime.now().isoformat(),
            recipient=recipient or "unknown",
            sender="self",
            amount=str(amount_of_money),
            description="",
        )
        add_transaction(tracker.sender_id, new_transaction)
        write_account(tracker.sender_id, account)
        return [SlotSet("transfer_successful", True)]
