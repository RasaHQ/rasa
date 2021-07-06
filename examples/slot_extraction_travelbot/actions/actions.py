# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions

from typing import Any, Text, Dict, List

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet


class BookingStatus(Action):
    def name(self) -> Text:
        return "action_extract_booking_status"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        has_booked_slot = tracker.get_slot("has_booked")
        last_bot_utterance = tracker.get_last_event_for("bot")
        if not last_bot_utterance:
            return []

        bot_msg = last_bot_utterance.get("text")

        if has_booked_slot is None:
            has_booked = True if bot_msg == "Your booking has been completed successfully." \
                else False
            return [SlotSet("has_booked", has_booked)]

        if not has_booked_slot:
            if bot_msg == "Your booking has been completed successfully.":
                has_booked = True
                return [SlotSet("has_booked", has_booked)]

        return []


class SubscriptionStatus(Action):
    def name(self) -> Text:
        return "action_extract_subscription_status"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        has_subscribed_slot = tracker.get_slot("has_subscribed")
        last_bot_utterance = tracker.get_last_event_for("bot")
        if not last_bot_utterance:
            return []

        bot_msg = last_bot_utterance.get("text")

        if has_subscribed_slot is None:
            if bot_msg == "You are now subscribed to the travel newsletter.":
                has_subscribed = True
            else:
                has_subscribed = False
            return [SlotSet("has_subscribed", has_subscribed)]

        if not has_subscribed_slot:
            if bot_msg == "You are now subscribed to the travel newsletter.":
                has_subscribed = True
                return [SlotSet("has_subscribed", has_subscribed)]

        return []
