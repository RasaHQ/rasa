from typing import Any, Optional, Tuple, Text

from rasa.nlu.constants import (
    ENTITIES,
    INTENT,
    RESPONSE,
    RESPONSE_KEY_ATTRIBUTE,
    TEXT,
    RESPONSE_IDENTIFIER_DELIMITER,
)
from rasa.nlu.utils import ordered


class Message:
    def __init__(
        self, text: Text, data=None, output_properties=None, time=None
    ) -> None:
        self.text = text
        self.time = time
        self.data = data if data else {}

        if output_properties:
            self.output_properties = output_properties
        else:
            self.output_properties = set()

    def set(self, prop, info, add_to_output=False) -> None:
        self.data[prop] = info
        if add_to_output:
            self.output_properties.add(prop)

    def get(self, prop, default=None) -> Any:
        if prop == TEXT:
            return self.text
        return self.data.get(prop, default)

    def as_dict_nlu(self) -> dict:
        """Get dict representation of message as it would appear in training data"""

        d = self.as_dict()
        if d.get(INTENT, None):
            d[INTENT] = self.get_combined_intent_response_key()
        d.pop(RESPONSE_KEY_ATTRIBUTE, None)
        d.pop(RESPONSE, None)
        return d

    def as_dict(self, only_output_properties=False) -> dict:
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

    def __eq__(self, other) -> bool:
        if not isinstance(other, Message):
            return False
        else:
            return (other.text, ordered(other.data)) == (self.text, ordered(self.data))

    def __hash__(self) -> int:
        return hash((self.text, str(ordered(self.data))))

    @classmethod
    def build(cls, text, intent=None, entities=None) -> "Message":
        data = {}
        if intent:
            split_intent, response_key = cls.separate_intent_response_key(intent)
            data[INTENT] = split_intent
            if response_key:
                data[RESPONSE_KEY_ATTRIBUTE] = response_key
        if entities:
            data[ENTITIES] = entities
        return cls(text, data)

    def get_combined_intent_response_key(self) -> Text:
        """Get intent as it appears in training data"""

        intent = self.get(INTENT)
        response_key = self.get(RESPONSE_KEY_ATTRIBUTE)
        response_key_suffix = (
            f"{RESPONSE_IDENTIFIER_DELIMITER}{response_key}" if response_key else ""
        )
        return f"{intent}{response_key_suffix}"

    @staticmethod
    def separate_intent_response_key(original_intent) -> Optional[Tuple[Any, Any]]:

        split_title = original_intent.split(RESPONSE_IDENTIFIER_DELIMITER)
        if len(split_title) == 2:
            return split_title[0], split_title[1]
        elif len(split_title) == 1:
            return split_title[0], None
