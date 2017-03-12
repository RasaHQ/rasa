# -*- coding: utf-8 -*-

import json
import os
import warnings
from itertools import groupby


class TrainingData(object):
    # Validation will ensure and warn if these lower limits are not met
    MIN_EXAMPLES_PER_INTENT = 2
    MIN_EXAMPLES_PER_ENTITY = 2

    def __init__(self, intent_examples=[], entity_examples=[], entity_synonyms={}):
        self.intent_examples = intent_examples
        self.entity_examples = entity_examples
        self.entity_synonyms = entity_synonyms

        self.validate()

    @property
    def num_entity_examples(self):
        return len([e for e in self.entity_examples if len(e["entities"]) > 0])

    def as_json(self, **kwargs):
        return json.dumps({
            "rasa_nlu_data": {
                "intent_examples": self.intent_examples,
                "entity_examples": self.entity_examples
            }
        }, **kwargs)

    def persist(self, dir_name):
        data_file = os.path.join(dir_name, "training_data.json")
        with open(data_file, 'w') as f:
            f.write(self.as_json(indent=2))

        return {
            "training_data": "training_data.json"
        }

    def sorted_entity_examples(self):
        return sorted([entity for ex in self.entity_examples for entity in ex["entities"]], key=lambda e: e["entity"])

    def sorted_intent_examples(self):
        return sorted(self.intent_examples, key=lambda e: e["intent"])

    def validate(self):
        examples = self.sorted_intent_examples()
        for intent, group in groupby(examples, lambda e: e["intent"]):
            size = len(list(group))
            if size < self.MIN_EXAMPLES_PER_INTENT:
                template = u"Intent '{0}' has only {1} training examples! minimum is {2}, training may fail."
                warnings.warn(template.format(intent, size, self.MIN_EXAMPLES_PER_INTENT))

        sorted_entity_examples = self.sorted_entity_examples()
        for entity, group in groupby(sorted_entity_examples, lambda e: e["entity"]):
            size = len(list(group))
            if size < self.MIN_EXAMPLES_PER_ENTITY:
                template = u"Entity '{0}' has only {1} training examples! minimum is {2}, training may fail."
                warnings.warn(template.format(entity, size, self.MIN_EXAMPLES_PER_ENTITY))
