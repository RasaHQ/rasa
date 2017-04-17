from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import map
from typing import Any
from typing import Dict
from typing import Text

from rasa_nlu.components import Component


class KeywordIntentClassifier(Component):

    name = "intent_classifier_keyword"

    context_provides = {
        "process": ["intent"],
    }

    output_provides = ["intent"]

    his = ["hello", "hi", "hey"]

    byes = ["bye", "goodbye"]

    def process(self, text):
        # type: (Text) -> Dict[Text, Any]

        return {
            "intent": {
                "name": self.parse(text),
                "confidence": 1.0,
            }
        }

    def parse(self, text):
        # type: (Text) -> Text

        _text = text.lower()

        def is_present(x): return x in _text

        if any(map(is_present, self.his)):
            return "greet"
        elif any(map(is_present, self.byes)):
            return "goodbye"
        else:
            return "None"
