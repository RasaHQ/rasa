from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import io
import json
import logging
from builtins import map
from typing import Any
from typing import Dict
from typing import Text
import re

import rasa_nlu.converters as converters
from rasa_nlu.training_data import TrainingData
from rasa_nlu.components import Component

logger = logging.getLogger(__name__)


class IntentOutOfScopeThreshold(Component):
    """ Out-of-scope intent detection component that classifies an utterence as oos if confidence levels are 
    below a specified threshold"""
    
    name = "intent_threshold"

    provides = ["intent"]

    requires = ["intent"]

    def __init__(self, intent_threshold=None):
        self.intent_threshold = intent_threshold

    def train(self, training_data, config, **kwargs):
        self.intent_threshold = config["intent_threshold"]
        logger.info("Intent Threshold: {}".format(self.intent_threshold))

    def process(self, message, **kwargs):
        # type: (Text, **Any) -> None
        if self.intent_threshold is None:
            raise Exception("Failed to train 'intent_threshold'. Missing a specified threshold")
        else: 
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
            f.write(str(json.dumps({"intent_threshold": self.intent_threshold})))
        return {"intent_threshold_persisted": file_name}

    @classmethod
    def load(cls, model_dir=None, model_metadata=None, cached_component=None, **kwargs):
        # type: (Text) -> IntentOutOfScopeThreashold
        persisted = os.path.join(model_dir, model_metadata.get("intent_threshold_persisted"))
        if os.path.isfile(persisted):
            with io.open(persisted, encoding='utf-8') as f:
                persisted_data = json.loads(f.read())
                return IntentOutOfScopeThreshold(persisted_data["intent_threshold"])
        return IntentOutOfScopeThreshold()