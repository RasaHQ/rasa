from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher


class ActionEndSession(Action):
    """Custom action that terminates the conversation by returning SessionEnded."""

    def name(self) -> str:
        return "action_end_session"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[str, Any],
    ) -> List[Dict[Text, Any]]:
        # TODO: replace with `return [SessionEnded()]` once the 
        # rasa-sdk release that supports SessionEnded event is made
        # remember to import SessionEnded from rasa_sdk.events
        return [{"event": "session_ended"}]
