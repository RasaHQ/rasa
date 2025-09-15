from datetime import datetime
from typing import Any, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.db import get_transactions


class ListTransactions(Action):
    def name(self) -> str:
        return "list_transactions"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        full_transaction_list = tracker.get_slot("full_transaction_list")

        transactions = get_transactions(tracker.sender_id)

        # sort by date
        transactions.sort(
            key=lambda x: datetime.strptime(x.datetime, "%Y-%m-%dT%H:%M:%S.%f"),
            reverse=True,
        )

        if not full_transaction_list:
            transactions = transactions[:5]

        transactions_list = "\n".join([t.stringify() for t in transactions])
        return [SlotSet("transactions_list", transactions_list)]
