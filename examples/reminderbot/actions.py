# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/core/actions/#custom-actions/


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
import datetime

from rasa_sdk import Action, Tracker
from rasa_sdk.events import ReminderScheduled, ReminderCancelled
from rasa_sdk.executor import CollectingDispatcher


class ActionSetReminder(Action):
    """Schedules a reminder, supplied with the last message's entities."""

    def name(self) -> Text:
        return "action_set_reminder"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        dispatcher.utter_message("I will remind you in 5 seconds.")

        date = datetime.datetime.now() + datetime.timedelta(seconds=5)
        entities = tracker.latest_message.get("entities")

        reminder = ReminderScheduled(
            "EXTERNAL_reminder",
            trigger_date_time=date,
            entities=entities,
            name="my_reminder",
            kill_on_user_message=False,
        )

        return [reminder]


class ActionReactToReminder(Action):
    """Reminds the user to call someone."""

    def name(self) -> Text:
        return "action_react_to_reminder"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        name = next(tracker.get_latest_entity_values("name"), None) or ""
        dispatcher.utter_message(f"Remember to call {name}!")

        return []


class ActionTellID(Action):
    """Informs the user about the conversation ID."""

    def name(self) -> Text:
        return "action_tell_id"

    def run(
        self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        conversation_id = tracker.sender_id

        dispatcher.utter_message(
            f"The ID of this conversation is: " f"{conversation_id}."
        )

        dispatcher.utter_message(
            f"Trigger an intent with "
            f'curl -H "Content-Type: application/json" '
            f'-X POST -d \'{{"name": "EXTERNAL_pizza_ready", '
            f'"entities": {{"topping": "Hawaii"}}}}\' '
            f"http://localhost:5005/conversations/{conversation_id}/"
            f"trigger_intent"
        )

        return []


class ActionPizzaReady(Action):
    """Informs the user that pizza is ready."""
    def name(self) -> Text:
        return "action_pizza_ready"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        topping = next(tracker.get_latest_entity_values("topping"), None) or ""
        print(topping)
        dispatcher.utter_message(f"Your Pizza {topping} is ready!")

        return []


class ForgetReminders(Action):
    """Cancels all reminders."""

    def name(self) -> Text:
        return "action_forget_reminders"

    def run(
        self, dispatcher, tracker: Tracker, domain: Dict[Text, Any]
    ) -> List[Dict[Text, Any]]:

        # Cancel all reminders
        return [ReminderCancelled()]
