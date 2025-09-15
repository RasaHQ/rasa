from typing import Any, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.db import get_account


class CheckBalance(Action):
    def name(self) -> str:
        return "check_balance"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        account = get_account(tracker.sender_id)
        return [SlotSet("current_balance", account.funds)]
