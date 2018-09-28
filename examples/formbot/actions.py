# -*- coding: utf-8 -*-
from rasa_core_sdk.forms import FormAction, REQUESTED_SLOT
from rasa_core_sdk.events import SlotSet


class RestaurantForm(FormAction):

    def name(self):
        return "restaurant_form"

    @staticmethod
    def required_slots():
        return ["cuisine", "num_people"]

    def slot_mapping(self):
        return {"cuisine": "cuisine", "num_people": "number"}

    cuisine_db = ["caribbean",
                  "chinese",
                  "french",
                  "greek",
                  "indian",
                  "italian",
                  "mexican"]

    @staticmethod
    def is_int(string):
        try:
            int(string)
            return True
        except ValueError:
            return False

    def validate(self, dispatcher, tracker, domain):

        slot_to_fill = tracker.slots[REQUESTED_SLOT]

        events = super(RestaurantForm,
                       self).validate(dispatcher, tracker, domain)

        entity = events[0]['value']

        if slot_to_fill == 'cuisine':
            if entity.lower() not in self.cuisine_db:
                dispatcher.utter_template('utter_wrong_cuisine', tracker)
                events = [SlotSet(slot_to_fill, None)]
        elif slot_to_fill == 'num_people':
            if not self.is_int(entity) or int(entity) <= 0:
                dispatcher.utter_template('utter_wrong_num_people', tracker)
                events = [SlotSet(slot_to_fill, None)]

        return events

    def submit(self, dispatcher, tracker, domain):
        dispatcher.utter_template('utter_submit', tracker)
        return []
