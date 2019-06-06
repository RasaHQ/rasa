# -*- coding: utf-8 -*-

from rasa.nlu.utils import ordered


class Message(object):
    def __init__(self, text, data=None, output_properties=None, time=None, params=None):
        self.text = text
        self.time = time
        self.data = data if data else {}
        self.params = params

        if output_properties:
            self.output_properties = output_properties
        else:
            self.output_properties = set()

    def set(self, prop, info, add_to_output=False):
        self.data[prop] = info
        if add_to_output:
            self.output_properties.add(prop)

    def get(self, prop, default=None):
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
        return dict(d, text=self.text)

    def __eq__(self, other):
        if not isinstance(other, Message):
            return False
        else:
            return (other.text, ordered(other.data)) == (self.text, ordered(self.data))

    def __hash__(self):
        return hash((self.text, str(ordered(self.data))))

    @classmethod
    def build(cls, text, intent=None, entities=None):
        data = {}
        if intent:
            data["intent"] = intent
        if entities:
            data["entities"] = entities
        return cls(text, data)
