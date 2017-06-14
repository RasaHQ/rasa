from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import io
import json
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

    def __init__(self, regex_dict={}):
        self.regex_dict = regex_dict

    def train(self, training_data):
        """ 
        Build regex: intent dictionary from training data 
        """
        for example in training_data.regex_features:
            if "intent" in example:
                self.regex_dict[example["pattern"]] = example["intent"]

    def process(self, text):
        # type: (Text) -> Dict[Text, Any]
        result = self.parse(text)
        if result is not None:
            return {
                "intent": {
                    "name": result,
                    "confidence": 1.0,
                }
            }

    def parse(self, text):
        # type: (Text) -> Text
        for exp, intent in self.regex_dict.items():
            if (re.search(exp, text) != None):
                return intent
        return None

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        file_name = self.name+".json"
        full_name = os.path.join(model_dir, file_name)
        with io.open(full_name, 'w') as f:
            f.write(unicode(json.dumps({"regex_dictionary": self.regex_dict})))
        return {"ner_regex_persisted": file_name}

    @classmethod
    def load(cls, model_dir, ner_regex_persisted):
        # type: (Text) -> RegExIntentClassifier
        persisted = os.path.join(model_dir, ner_regex_persisted)
        if os.path.isfile(persisted):
            with io.open(persisted, encoding='utf-8') as f:
                persisted_data = json.loads(f.read())
                return RegExIntentClassifier(persisted_data["regex_dictionary"])

