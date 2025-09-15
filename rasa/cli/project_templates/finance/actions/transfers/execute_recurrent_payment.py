from typing import List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import EventType, SlotSet
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict


class ExecutePayment(Action):
    def name(self) -> str:
        return "action_execute_recurrent_payment"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: DomainDict,
    ) -> List[EventType]:
        # Set-up payment logic and additional checks here
        return [SlotSet("setup_recurrent_payment_successful", True)]
