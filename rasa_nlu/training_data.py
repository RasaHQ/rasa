# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import json
import logging
import os
import warnings
from itertools import groupby

from builtins import object, str
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.utils import lazyproperty
from rasa_nlu.utils import list_to_str

logger = logging.getLogger(__name__)

TrainingExample = Dict[Text, Any]
TrainingExamples = List[TrainingExample]


class TrainingData(object):
    """Holds loaded intent and entity training data."""

    # Validation will ensure and warn if these lower limits are not met
    MIN_EXAMPLES_PER_INTENT = 2
    MIN_EXAMPLES_PER_ENTITY = 2

    def __init__(self, training_examples=None, entity_synonyms=None):
        # type: (Optional[TrainingExamples], Optional[Dict[Text, List[Text]]]) -> None

        self.training_examples = self.sanitice_examples(training_examples) if training_examples else []
        self.entity_synonyms = entity_synonyms if entity_synonyms else {}

        self.validate()

    def sanitice_examples(self, examples):
        """Makes sure the training data is cleaned, e.q. removes trailing whitespaces from intent annotations."""

        for e in examples:
            if "intent" in e:
                e["intent"] = e["intent"].strip()
        return examples

    @lazyproperty
    def intent_examples(self):
        # type: () -> TrainingExamples
        return [e for e in self.training_examples if "intent" in e]

    @lazyproperty
    def entity_examples(self):
        # type: () -> TrainingExamples
        return [e for e in self.training_examples if "entities" in e]

    @lazyproperty
    def num_entity_examples(self):
        # type: () -> int
        """Returns the number of proper entity training examples (containing at least one annotated entity)."""

        return len([e for e in self.training_examples if len(e.get("entities", [])) > 0])

    @lazyproperty
    def num_intent_examples(self):
        # type: () -> int
        """Returns the number of intent examples."""

        return len(self.intent_examples)

    def as_json(self, **kwargs):
        # type: (**Any) -> str
        """Represent this set of training examples as json adding the passed meta information."""

        return str(json.dumps({
            "rasa_nlu_data": {
                "common_examples": self.training_examples,
            }
        }, **kwargs))

    def persist(self, dir_name):
        # type: (Text) -> Dict[Text, Any]
        """Persists this training data to disk and returns necessary information to load it again."""

        data_file = os.path.join(dir_name, "training_data.json")
        with io.open(data_file, 'w') as f:
            f.write(self.as_json(indent=2))

        return {
            "training_data": "training_data.json"
        }

    def sorted_entity_examples(self):
        # type: () -> List[TrainingExample]
        """Sorts the entity examples by the annotated entity."""

        return sorted([entity for ex in self.entity_examples for entity in ex["entities"]], key=lambda e: e["entity"])

    def sorted_intent_examples(self):
        # type: () -> List[TrainingExample]
        """Sorts the intent examples by the name of the intent."""

        return sorted(self.intent_examples, key=lambda e: e["intent"])

    def validate(self):
        # type: () -> None
        """Ensures that the loaded training data is valid, e.g. has a minimum of certain training examples."""

        logger.debug("Validating training data...")
        examples = self.sorted_intent_examples()
        different_intents = []
        for intent, group in groupby(examples, lambda e: e["intent"]):
            size = len(list(group))
            different_intents.append(intent)
            if size < self.MIN_EXAMPLES_PER_INTENT:
                template = "Intent '{}' has only {} training examples! minimum is {}, training may fail."
                warnings.warn(template.format(intent, size, self.MIN_EXAMPLES_PER_INTENT))

        sorted_entity_examples = self.sorted_entity_examples()
        different_entities = []
        for entity, group in groupby(sorted_entity_examples, lambda e: e["entity"]):
            size = len(list(group))
            different_entities.append(entity)
            if size < self.MIN_EXAMPLES_PER_ENTITY:
                template = "Entity '{}' has only {} training examples! minimum is {}, training may fail."
                warnings.warn(template.format(entity, size, self.MIN_EXAMPLES_PER_ENTITY))

        logger.info("Training data stats: \n" +
                    "\t- intent examples: {} ({} distinct intents)\n".format(
                            self.num_intent_examples, len(different_intents)) +
                    "\t- found intents: {}\n".format(list_to_str(different_intents)) +
                    "\t- entity examples: {} ({} distinct entities)\n".format(
                            self.num_entity_examples, len(different_entities)) +
                    "\t- found entities: {}\n".format(list_to_str(different_entities)))
