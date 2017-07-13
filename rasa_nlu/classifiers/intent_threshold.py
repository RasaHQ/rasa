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


class IntentThreshold(Component):

    name = "intent_threshold"

    provides = ["intent"]

    requires = ["intent"]

    def __init__(self, intent_threshold=None):
        # 'threshold' should be an attribute
        self.intent_threshold = intent_threshold

    def train(self, training_data, config, **kwargs):
        self.intent_threshold = config["intent_threshold"]
        print("threshold: {}".format(self.intent_threshold))   

    def process(self, message, **kwargs):
        # type: (Text) -> Dict[Text, Any]
        # Explicitly set the threshold
        # self.intent_threshold = 0.55
        #print("intent: {}".format(message.get("intent")))
        #print("threshold: {}".format(self.intent_threshold))
        if message.get("intent").get("confidence", 1.0) < self.intent_threshold:
            message.set("intent", {
                    "name": "out_of_scope",
                    "confidence": 1.0,
                }, add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        file_name = self.name+".json"
        full_name = os.path.join(model_dir, file_name)
        with io.open(full_name, 'w') as f:
            f.write(unicode(json.dumps({"intent_threshold": self.intent_threshold})))
        return {"ner_regex_persisted": file_name}

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text) -> RegExIntentClassifier
        persisted = os.path.join(model_dir, model_metadata.get("ner_regex_persisted"))
        if os.path.isfile(persisted):
            with io.open(persisted, encoding='utf-8') as f:
                persisted_data = json.loads(f.read())
                return RegExIntentClassifier(persisted_data["intent_threshold"])
        return RegExIntentClassifier()


# threshold = IntentThreshold()
# intent = {
#             "intent": {
#                 "name": "restaurant_search",
#                 "confidence": 0.45,
#             }
#         }
# print(threshold.process(intent))
