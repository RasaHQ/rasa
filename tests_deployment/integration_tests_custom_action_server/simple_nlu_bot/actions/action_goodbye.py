from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionSayGoodbye(Action):
    def name(self) -> str:
        return "action_goodbye"

    def run(
        self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[str, Any]
    ) -> List[Dict[Text, Any]]:
        dispatcher.utter_message(text="Goodbye!")
        return []
