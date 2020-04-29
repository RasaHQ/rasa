from typing import Dict, Text, Any, List, Union, Optional

from rasa_sdk import Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction, REQUESTED_SLOT
from rasa_sdk import Action
from rasa_sdk.events import SlotSet, Form


class ActionActivateQForm(Action):
    def name(self) -> Text:
        """Unique identifier of the form"""

        return "action_activate_q_form"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message("action_activate_q_form")
        return [Form("q_form")]


class ActionLoopQForm(Action):
    def name(self) -> Text:
        """Unique identifier of the form"""

        return "action_loop_q_form"

    def run(self, dispatcher, tracker, domain):
        slot = "some_slot"
        dispatcher.utter_template(
            "utter_ask_{}".format(slot), tracker, silent_fail=False, **tracker.slots
        )
        return [SlotSet(REQUESTED_SLOT, slot)]


class ValidateSomeSlot(Action):
    def name(self) -> Text:
        """Unique identifier of the form"""

        return "validate_some_slot"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message("validate_some_slot")
        value = tracker.get_latest_entity_values("some_slot")
        return [SlotSet("some_slot", value)]


class ActionStopQForm(Action):
    def name(self) -> Text:
        """Unique identifier of the form"""

        return "action_stop_q_form"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message("action_stop_q_form")
        return [Form(None)]


class ActionSwitchFAQ(Action):
    def name(self) -> Text:
        """Unique identifier of the form"""

        return "action_switch_faq"

    def run(self, dispatcher, tracker, domain):
        dispatcher.utter_message("action_switch_faq")
        v = tracker.get_slot("detailed_faq")

        if v is None:
            v = True
        else:
            v = not v

        return [SlotSet("detailed_faq", v)]
