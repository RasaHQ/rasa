# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from builtins import object, str, filter
import json
import os
import warnings
from itertools import groupby
import io


class TrainingData(object):
    """Holds loaded intent and entity training data."""

    # Validation will ensure and warn if these lower limits are not met
    MIN_EXAMPLES_PER_INTENT = 2
    MIN_EXAMPLES_PER_ENTITY = 2

    def __init__(self,
                 intent_examples_only=None,
                 entity_examples_only=None,
                 common_examples=None,
                 entity_synonyms=None):
        self.intent_examples_only = intent_examples_only if intent_examples_only else []
        self.entity_examples_only = entity_examples_only if entity_examples_only else []
        self.common_examples = common_examples if common_examples else []
        self.entity_synonyms = entity_synonyms if entity_synonyms else {}

        self.validate()

    @property
    def intent_examples(self):
        return list(filter(lambda e: "intent" in e,
                           self.intent_examples_only + self.common_examples))

    @property
    def entity_examples(self):
        return list(filter(lambda e: "entities" in e,
                           self.entity_examples_only + self.common_examples))

    @property
    def num_entity_examples(self):
        # type: () -> int
        """Returns the number of entity examples."""

        return len([e for e in self.entity_examples if len(e["entities"]) > 0])

    def as_json(self, **kwargs):
        # type: (dict) -> str
        """Represent this set of training examples as json adding the passed meta information."""

        return str(json.dumps({
            "rasa_nlu_data": {
                "common_examples": self.common_examples,
                "intent_examples": self.intent_examples_only,
                "entity_examples": self.entity_examples_only,
            }
        }, **kwargs))

    def persist(self, dir_name):
        # type: (str) -> dict
        """Persists this training data to disk and returns necessary information to load it again."""

        data_file = os.path.join(dir_name, "training_data.json")
        with io.open(data_file, 'w') as f:
            f.write(self.as_json(indent=2))

        return {
            "training_data": "training_data.json"
        }

    def sorted_entity_examples(self):
        # type: () -> [dict]
        """Sorts the entity examples by the annotated entity."""

        return sorted([entity for ex in self.entity_examples for entity in ex["entities"]], key=lambda e: e["entity"])

    def sorted_intent_examples(self):
        # type: () -> [dict]
        """Sorts the intent examples by the name of the intent."""

        return sorted(self.intent_examples, key=lambda e: e["intent"])

    def validate(self):
        # type: () -> None
        """Ensures that the loaded training data is valid, e.g. has a minimum of certain training examples."""

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
