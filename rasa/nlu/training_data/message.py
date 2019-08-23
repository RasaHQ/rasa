# -*- coding: utf-8 -*-

from rasa.nlu.utils import ordered

from rasa.nlu.constants import (
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_INTENT_ATTRIBUTE,
    MESSAGE_RESPONSE_ATTRIBUTE,
    MESSAGE_ENTITIES_ATTRIBUTE,
)


class Message(object):
    def __init__(self, text, data=None, output_properties=None, time=None):
        self.text = text
        self.time = time
        self.data = data if data else {}

        if output_properties:
            self.output_properties = output_properties
        else:
            self.output_properties = set()

    def set(self, prop, info, add_to_output=False):
        self.data[prop] = info
        if add_to_output:
            self.output_properties.add(prop)

    def get(self, prop, default=None):
        if prop == MESSAGE_TEXT_ATTRIBUTE:
            return self.text
        return self.data.get(prop, default)

    def as_dict(self, only_output_properties=False):
        if only_output_properties:
            d = {
                key: value
                for key, value in self.data.items()
                if key in self.output_properties
            }
        else:
            d = self.data

        # Filter all keys with None value. These could have come while building the Message object in markdown format
        d = {key: value for key, value in d.items() if value is not None}

        return dict(d, text=self.text)

    def __eq__(self, other):
        if not isinstance(other, Message):
            return False
        else:
            return (other.text, ordered(other.data)) == (self.text, ordered(self.data))

    def __hash__(self):
        return hash((self.text, str(ordered(self.data))))

    @classmethod
    def build(cls, text, intent=None, entities=None, response=None):
        data = {}
        if intent:
            data[MESSAGE_INTENT_ATTRIBUTE] = intent
        if entities:
            data[MESSAGE_ENTITIES_ATTRIBUTE] = entities
        if response:
            data[MESSAGE_RESPONSE_ATTRIBUTE] = response
        return cls(text, data)
