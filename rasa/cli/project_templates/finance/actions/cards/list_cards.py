from typing import Any, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.db import get_cards


class ListCards(Action):
    def name(self) -> str:
        return "list_cards"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        cards = get_cards(tracker.sender_id)
        if len(cards) > 0:
            cards_list = "".join([f"- {c.name} ({c.handle}) \n" for c in cards])
            return [SlotSet("cards_list", cards_list)]
        else:
            return [SlotSet("cards_list", None)]
