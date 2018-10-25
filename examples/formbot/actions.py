# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import typing
from typing import Dict, Text, Any, List, Union

from rasa_core_sdk import ActionExecutionRejection
from rasa_core_sdk.forms import FormAction, REQUESTED_SLOT
from rasa_core_sdk.events import SlotSet

if typing.TYPE_CHECKING:
    from rasa_core_sdk import Tracker
    from rasa_core_sdk.executor import CollectingDispatcher


class RestaurantForm(FormAction):
    """Example of a custom form action"""

    def name(self):
        # type: () -> Text
        """Unique identifier of the form"""

        return "restaurant_form"

    @staticmethod
    def required_slots(tracker):
        # type: (Tracker) -> List[Text]
        """A list of required slots that the form has to fill"""

        return ["cuisine", "num_people", "outdoor_seating",
                "preferences", "feedback"]

    def slot_mappings(self):
        # type: () -> Dict[Text: Union[Dict, List[Dict]]]
        """A dictionary to map required slots to
            - an extracted entity
            - intent: value pairs
            - a whole message
            or a list of them, where a first match will be picked"""

        return {"cuisine": self.from_entity(entity="cuisine",
                                            intent="inform"),
                "num_people": self.from_entity(entity="number"),
                "outdoor_seating": [self.from_entity(entity="seating"),
                                    self.from_intent(intent='affirm',
                                                     value=True),
                                    self.from_intent(intent='deny',
                                                     value=False)],
                "preferences": [self.from_text(intent='inform'),
                                self.from_intent(intent='deny',
                                                 value="no additional "
                                                       "preferences")],
                "feedback": [self.from_entity(entity="feedback"),
                             self.from_text()]}

    @staticmethod
    def cuisine_db():
        # type: () -> List[Text]
        """Database of supported cuisines"""
        return ["caribbean",
                "chinese",
                "french",
                "greek",
                "indian",
                "italian",
                "mexican"]

    @staticmethod
    def is_int(string):
        # type: (Text) -> bool
        """Check if a string is an integer"""
        try:
            int(string)
            return True
        except ValueError:
            return False

    def validate(self, dispatcher, tracker, domain):
        # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict]
        """"Validate extracted requested slot else raise an error"""
        slot_to_fill = tracker.get_slot(REQUESTED_SLOT)

        # extract requested slot from a user input by using `slot_mapping`
        extracted_value = self.extract(dispatcher, tracker, domain)
        if extracted_value is None:
            # raise an error if nothing was extracted
            raise ActionExecutionRejection(self.name(),
                                           "Failed to validate slot {0} "
                                           "with action {1}"
                                           "".format(slot_to_fill,
                                                     self.name()))

        if slot_to_fill == 'cuisine':
            if extracted_value.lower() not in self.cuisine_db():
                dispatcher.utter_template('utter_wrong_cuisine', tracker)
                # validation failed, set this slot to None
                return [SlotSet(slot_to_fill, None)]

        elif slot_to_fill == 'num_people':
            if not self.is_int(extracted_value) or int(extracted_value) <= 0:
                dispatcher.utter_template('utter_wrong_num_people',
                                          tracker)
                # validation failed, set this slot to None
                return [SlotSet(slot_to_fill, None)]

        elif slot_to_fill == 'outdoor_seating':
            if isinstance(extracted_value, str):
                if 'out' in extracted_value:
                    # convert "out..." to True
                    return [SlotSet(slot_to_fill, True)]
                elif 'in' in extracted_value:
                    # convert "in..." to False
                    return [SlotSet(slot_to_fill, False)]

        return [SlotSet(slot_to_fill, extracted_value)]

    def submit(self, dispatcher, tracker, domain):
        # type: (CollectingDispatcher, Tracker, Dict[Text, Any]) -> List[Dict]
        """Define what the form has to do
            after all required slots are filled"""

        # utter submit template
        dispatcher.utter_template('utter_submit', tracker)
        return []
