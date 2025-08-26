from datetime import datetime
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import ActionExecuted, SlotSet
from rasa_sdk.executor import CollectingDispatcher

from actions.database import Database


class ActionSessionStart(Action):
    def name(self) -> Text:
        return "action_session_start"

    def set_current_date(self) -> List[Dict[Text, Any]]:
        current_date = datetime.now().strftime("%d/%m/%Y")
        return [SlotSet("current_date", current_date)]

    def set_user_profile(self, name: str) -> List[Dict[Text, Any]]:
        db = Database()

        # Get user information
        user = db.get_user_by_name(name)

        if user:
            return [
                SlotSet("name", user["name"]),
                SlotSet("username", user["username"]),
                SlotSet("segment", user["segment"]),
                SlotSet("email_address", user["email"]),
                SlotSet("physical_address", user["address"]),
            ]
        else:
            return []

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        current_date_events = self.set_current_date()

        # Advanced option: It is possible to enable the chatbot with multiple
        # user profiles.
        # This option influences the interactions with the assistant.
        # The bot randomly assigns a user profile at the start of each session.
        # With this advance option: The demo bot will include three user profiles
        # with different payees:
        # John Smith: Payees: Robert (friend), James (son), Food Market (groceries)
        # Mary Brown: Payees: Richard (business partner), Susan (friend),
        # Electric Company (utilities)
        # Dan Young: Payees: Amy (colleague), Fitness Gym (gym), William (friend)
        #
        # The default profile is John Smith
        #
        # Each profile also has its own accounts—Current (Checking), Savings,
        # and Business
        #
        # Intructions: Just toggle the comments of the following 2 lines.
        name = "John Smith"
        # username = random.choice(["John Smith", "Mary Brown", "Dan Young"])

        user_profile_events = self.set_user_profile(name)

        events = (
            current_date_events
            + user_profile_events
            + [
                ActionExecuted("action_listen"),
            ]
        )

        return events
