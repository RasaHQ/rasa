from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from builtins import map
from typing import Any
from typing import Dict
from typing import Text
import re

import rasa_nlu.converters as converters
from rasa_nlu.training_data import TrainingData

from rasa_nlu.components import Component


class RegExIntentClassifier(Component):

    name = "intent_classifier_regex"

    context_provides = {
        "process": ["intent"],
    }

    output_provides = ["intent"]

    def __init__(self, clf=None, regex_dict=None):
        self.clf = clf
        self.regex_dict = {}

    def train(self, training_data):
    # build regex: intent dict from training data
        for example in training_data.regex_features:
            if ("intent" in example):
                self.regex_dict[example["pattern"]] = example["intent"]

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

        #def is_present(x): return x in _text

        for exp, intent in self.regex_dict.items():
            search = (re.search(exp, text) != None)
            if search:
                return intent
        return "None"
