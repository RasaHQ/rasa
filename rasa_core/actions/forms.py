# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from rasa_core.actions.action import Action
from rasa_core.events import SlotSet

import logging
import random

logger = logging.getLogger(__name__)


class FormField(object):

    def validate(self, value):
        """Check if extracted value for a requested slot is valid.

        Users should override this to implement custom validation logic,
        returning None indicates a negative validation result, and the slot
        will not be set.
        """
        return value


class EntityFormField(FormField):

    def __init__(self, entity_name, slot_name):
        self.entity_name = entity_name
        self.slot_name = slot_name

    def extract(self, tracker):
        value = next(tracker.get_latest_entity_values(self.entity_name), None)
        validated = self.validate(value)
        if validated is not None:
            return [SlotSet(self.slot_name, validated)]
        else:
            return []


class BooleanFormField(FormField):
    """A form field that prompts the user for a yes or no answer.
    The interpreter should map positive and negative answers to
    the intents ``affirm_intent`` and ``deny_intent``.
    """
    def __init__(self, slot_name, affirm_intent, deny_intent):
        self.slot_name = slot_name
        self.affirm_intent = affirm_intent
        self.deny_intent = deny_intent

    def extract(self, tracker):

        intent = tracker.latest_message.intent.get("name")
        if intent == self.affirm_intent:
            value = True
        elif intent == self.deny_intent:
            value = False
        else:
            return []

        return [SlotSet(self.slot_name, value)]


class FreeTextFormField(FormField):

    def __init__(self, slot_name):
        self.slot_name = slot_name

    def extract(self, tracker):
        validated = self.validate(tracker.latest_message.text)
        if validated is not None:
            return [SlotSet(self.slot_name, validated)]
        return []


class FormAction(Action):

    RANDOMIZE = True

    @staticmethod
    def required_fields():
        return []

    def should_request_slot(self, tracker, slot_name):
        existing_val = tracker.get_slot(slot_name)
        return existing_val is None

    def get_other_slots(self, tracker):
        requested_slot = tracker.get_slot("requested_slot")

        requested_entity = None
        for f in self.required_fields():
            if f.slot_name == requested_slot:
                requested_entity = getattr(f, 'entity_name', None)

        slot_events = []
        extracted_entities = {requested_entity}

        for f in self.required_fields():
            if isinstance(f, EntityFormField) and \
                not f.slot_name == requested_slot and \
                not f.entity_name in extracted_entities:
                slot_events.extend(f.extract(tracker))
                extracted_entities.add(f.entity_name)
        return slot_events

    def get_requested_slot(self, tracker):
        requested_slot = tracker.get_slot("requested_slot")

        required = self.required_fields()

        if self.RANDOMIZE:
            random.shuffle(required)

        if requested_slot is None:
            return []
        else:
            fields = [f for f in required if f.slot_name == requested_slot]
            if len(fields) == 1:
                return fields[0].extract(tracker)
            else:
                logger.debug("unable to extract value "
                             "for requested slot: {}".format(requested_slot))
                return []

    def run(self, dispatcher, tracker, domain):

        events = self.get_requested_slot(tracker) + self.get_other_slots(tracker)

        temp_tracker = tracker.copy()
        for e in events:
            temp_tracker.update(e)

        for field in self.required_fields():
            if self.should_request_slot(temp_tracker, field.slot_name):
                dispatcher.utter_template(
                    "utter_ask_{}".format(field.slot_name),
                    temp_tracker)

                events.append(SlotSet("requested_slot", field.slot_name))
                return events

        events_from_submit = self.submit(dispatcher, temp_tracker, domain) or []

        return events + events_from_submit

    def submit(self, dispatcher, tracker, domain):
        raise NotImplementedError("a form action must implement a submit method")
