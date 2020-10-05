from typing import Dict, Text, Any, List, Union

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.forms import FormAction
from rasa_sdk.events import (
    SlotSet,
    EventType,
)


class ActionRestaurant(Action):
    """Example of a custom form action."""

    def name(self) -> Text:
        return "action_restaurant"

    async def run(self, dispatcher, tracker, domain) -> List[EventType]:
        """Define what the form has to do after all required slots are filled."""

        dispatcher.utter_message(template="utter_submit")
        dispatcher.utter_message(template="utter_slots_values")
        return []


class ValidateRestaurantForm(Action):
    def name(self) -> Text:
        return "validate_restaurant_form"

    async def run(self, dispatcher, tracker, domain) -> List[EventType]:
        extracted_slots = tracker.slots_to_validate()
        validation_events = []

        for slot_name, slot_value in extracted_slots.items():
            if slot_name == "cuisine":
                validation_events.append(validate_cuisine(slot_value, dispatcher))
            elif slot_name == "num_people":
                validation_events.append(validate_num_people(slot_value, dispatcher))
            elif slot_name == "outdoor_seating":
                validation_events.append(
                    validate_outdoor_seating(slot_value, dispatcher)
                )
            else:
                validation_events.append(SlotSet(slot_name, slot_value))

        return validation_events


def cuisine_db() -> List[Text]:
    """Database of supported cuisines."""

    return [
        "caribbean",
        "chinese",
        "french",
        "greek",
        "indian",
        "italian",
        "mexican",
    ]


def is_int(string: Text) -> bool:
    """Check if a string is an integer."""

    try:
        int(string)
        return True
    except ValueError:
        return False


def validate_cuisine(value: Text, dispatcher: CollectingDispatcher) -> Dict[Text, Any]:
    """Validate cuisine value."""

    if value.lower() in cuisine_db():
        # validation succeeded, set the value of the "cuisine" slot to value
        return SlotSet("cuisine", value)
    else:
        dispatcher.utter_message(template="utter_wrong_cuisine")
        # validation failed, set this slot to None, meaning the
        # user will be asked for the slot again
        return SlotSet("cuisine", None)


def validate_num_people(
    value: Text, dispatcher: CollectingDispatcher
) -> Dict[Text, Any]:
    """Validate num_people value."""

    if is_int(value) and int(value) > 0:
        return SlotSet("num_people", value)
    else:
        dispatcher.utter_message(template="utter_wrong_num_people")
        # validation failed, set slot to None
        return SlotSet("num_people", None)


def validate_outdoor_seating(
    value: Text, dispatcher: CollectingDispatcher
) -> Dict[Text, Any]:
    """Validate outdoor_seating value."""

    if isinstance(value, str):
        if "out" in value:
            # convert "out..." to True
            return SlotSet("outdoor_seating", True)
        elif "in" in value:
            # convert "in..." to False
            return SlotSet("outdoor_seating", False)
        else:
            dispatcher.utter_message(template="utter_wrong_outdoor_seating")
            # validation failed, set slot to None
            return SlotSet("outdoor_seating", None)

    else:
        # affirm/deny was picked up as True/False by the from_intent mapping
        return SlotSet("outdoor_seating", value)
