from typing import Any, Dict
from datetime import datetime
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from .db import get_account, write_account, add_transaction, Transaction


class ExecuteTransfer(Action):

    def name(self) -> str:
        return "execute_transfer"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker, domain: Dict[str, Any]):
        account = get_account(tracker.sender_id)

        recipient = tracker.get_slot("transfer_money_recipient")
        amount_of_money = tracker.get_slot("transfer_money_amount_of_money")

        if recipient is None or amount_of_money is None:
            return [SlotSet("transfer_money_transfer_successful", False)]

        # just as a play case
        if recipient == "Jack":
            return [SlotSet("transfer_money_transfer_successful", False)]

        amount_of_money_value = float(amount_of_money.replace("$", "").replace("USD", "").strip())
        account.funds -= amount_of_money_value
        new_transaction = \
            Transaction(datetime=datetime.now().isoformat(), recipient=recipient,
                        sender="self", amount=amount_of_money, description="")
        add_transaction(tracker.sender_id, new_transaction)
        write_account(tracker.sender_id, account)
        return [SlotSet("transfer_money_transfer_successful", True)]
