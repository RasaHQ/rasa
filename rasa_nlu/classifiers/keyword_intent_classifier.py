from typing import Any, Optional, Text

from rasa_nlu.components import Component
from rasa_nlu.training_data import Message


class KeywordIntentClassifier(Component):

    provides = ["intent"]

    his = ["hello", "hi", "hey"]

    byes = ["bye", "goodbye"]

    def process(self, message: Message, **kwargs: Any) -> None:

        intent = {"name": self.parse(message.text), "confidence": 1.0}
        message.set("intent", intent,
                    add_to_output=True)

    def parse(self, text: Text) -> Optional[Text]:

        _text = text.lower()

        def is_present(x):
            return x in _text

        if any(map(is_present, self.his)):
            return "greet"
        elif any(map(is_present, self.byes)):
            return "goodbye"
        else:
            return None
