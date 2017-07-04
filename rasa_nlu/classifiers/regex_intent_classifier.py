from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import os
import re

from builtins import str
from typing import Any, Optional
from typing import Dict
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.training_data import TrainingData, Message


class RegExIntentClassifier(Component):
    name = "intent_classifier_regex"

    provides = ["intent"]

    def __init__(self, regex_dict=None):
        self.regex_dict = regex_dict if regex_dict else {}

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None
        """ Build regex: intent dictionary from training data """

        for example in training_data.regex_features:
            if "intent" in example:
                self.regex_dict[example["pattern"]] = example["intent"]

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Looks for the known regex pattern in the message and sets the intent if a pattern matches."""

        result = self.find_pattern_match(message.text)
        if result is not None:
            message.set("intent", {"name": result, "confidence": 1.0}, add_to_output=True)

    def find_pattern_match(self, text):
        # type: (Text) -> Optional[Text]

        for exp, intent in self.regex_dict.items():
            if re.search(exp, text) is not None:
                return intent
        return None

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory. Returns the metadata necessary to load the model again."""

        file_name = self.name + ".json"
        full_name = os.path.join(model_dir, file_name)
        with io.open(full_name, 'w') as f:
            f.write(str(json.dumps({"regex_dictionary": self.regex_dict})))
        return {"ner_regex_persisted": file_name}

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text) -> RegExIntentClassifier

        if model_dir and model_metadata.get("intent_classifier_sklearn"):
            persisted = os.path.join(model_dir, model_metadata.get("ner_regex_persisted"))
            if os.path.isfile(persisted):
                with io.open(persisted, encoding='utf-8') as f:
                    persisted_data = json.loads(f.read())
                    return RegExIntentClassifier(persisted_data["regex_dictionary"])
        else:
            return RegExIntentClassifier()
