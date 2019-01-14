# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import warnings

from copy import deepcopy
from builtins import object, str
from rasa_nlu.training_data import Message

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from collections import Counter
import random

from rasa_nlu.utils import lazyproperty, write_to_file
from rasa_nlu.utils import list_to_str
from rasa_nlu.training_data.util import check_duplicate_synonym

logger = logging.getLogger(__name__)


class TrainingData(object):
    """Holds loaded intent and entity training data."""

    # Validation will ensure and warn if these lower limits are not met
    MIN_EXAMPLES_PER_INTENT = 2
    MIN_EXAMPLES_PER_ENTITY = 2

    def __init__(self,
                 training_examples=None,
                 entity_synonyms=None,
                 regex_features=None,
                 lookup_tables=None):
        # type: (Optional[List[Message]], Optional[Dict[Text, Text]]) -> None

        if training_examples:
            self.training_examples = self.sanitize_examples(training_examples)
        else:
            self.training_examples = []
        self.entity_synonyms = entity_synonyms if entity_synonyms else {}
        self.regex_features = regex_features if regex_features else []
        self.sort_regex_features()
        self.lookup_tables = lookup_tables if lookup_tables else []

        self.print_stats()

    def merge(self, *others):
        """Return merged instance of this data with other training data."""

        training_examples = deepcopy(self.training_examples)
        entity_synonyms = self.entity_synonyms.copy()
        regex_features = deepcopy(self.regex_features)
        lookup_tables = deepcopy(self.lookup_tables)

        for o in others:
            training_examples.extend(deepcopy(o.training_examples))
            regex_features.extend(deepcopy(o.regex_features))
            lookup_tables.extend(deepcopy(o.lookup_tables))

            for text, syn in o.entity_synonyms.items():
                check_duplicate_synonym(entity_synonyms, text, syn,
                                        "merging training data")

            entity_synonyms.update(o.entity_synonyms)

        return TrainingData(training_examples, entity_synonyms,
                            regex_features, lookup_tables)

    @staticmethod
    def sanitize_examples(examples):
        # type: (List[Message]) -> List[Message]
        """Makes sure the training data is clean.

        removes trailing whitespaces from intent annotations."""

        for ex in examples:
            if ex.get("intent"):
                ex.set("intent", ex.get("intent").strip())
        return examples

    @lazyproperty
    def intent_examples(self):
        # type: () -> List[Message]
        return [ex
                for ex in self.training_examples
                if ex.get("intent")]

    @lazyproperty
    def entity_examples(self):
        # type: () -> List[Message]
        return [ex
                for ex in self.training_examples
                if ex.get("entities")]

    @lazyproperty
    def intents(self):
        """Returns the set of intents in the training data."""
        return set([ex.get("intent") for ex in self.training_examples]) - {None}

    @lazyproperty
    def examples_per_intent(self):
        """Calculates the number of examples per intent."""
        intents = [ex.get("intent") for ex in self.training_examples]
        return dict(Counter(intents))

    @lazyproperty
    def entities(self):
        """Returns the set of entity types in the training data."""
        entity_types = [e.get("entity") for e in self.sorted_entities()]
        return set(entity_types)

    @lazyproperty
    def examples_per_entity(self):
        """Calculates the number of examples per entity."""
        entity_types = [e.get("entity") for e in self.sorted_entities()]
        return dict(Counter(entity_types))

    def sort_regex_features(self):
        """Sorts regex features lexicographically by name+pattern"""
        self.regex_features = sorted(self.regex_features,
                                     key=lambda e: "{}+{}".format(e['name'],
                                                                  e['pattern']))

    def as_json(self, **kwargs):
        # type: (**Any) -> str
        """Represent this set of training examples as json."""
        from rasa_nlu.training_data.formats import RasaWriter
        return RasaWriter().dumps(self)

    def as_markdown(self):
        # type: () -> str
        """Generates the markdown representation of the TrainingData."""
        from rasa_nlu.training_data.formats import MarkdownWriter
        return MarkdownWriter().dumps(self)

    def persist(self, dir_name):
        # type: (Text) -> Dict[Text, Any]
        """Persists this training data to disk and returns necessary
        information to load it again."""

        data_file = os.path.join(dir_name, "training_data.json")
        write_to_file(data_file, self.as_json(indent=2))

        return {
            "training_data": "training_data.json"
        }

    def sorted_entities(self):
        # type: () -> List[Any]
        """Extract all entities from examples and sorts them by entity type."""

        entity_examples = [entity
                           for ex in self.entity_examples
                           for entity in ex.get("entities")]
        return sorted(entity_examples, key=lambda e: e["entity"])

    def sorted_intent_examples(self):
        # type: () -> List[Message]
        """Sorts the intent examples by the name of the intent."""

        return sorted(self.intent_examples, key=lambda e: e.get("intent"))

    def validate(self):
        # type: () -> None
        """Ensures that the loaded training data is valid.

        Checks that the data has a minimum of certain training examples."""

        logger.debug("Validating training data...")
        if "" in self.intents:
            warnings.warn("Found empty intent, please check your "
                          "training data. This may result in wrong "
                          "intent predictions.")

        # emit warnings for intents with only a few training samples
        for intent, count in self.examples_per_intent.items():
            if count < self.MIN_EXAMPLES_PER_INTENT:
                warnings.warn("Intent '{}' has only {} training examples! "
                              "Minimum is {}, training may fail."
                              .format(intent, count,
                                      self.MIN_EXAMPLES_PER_INTENT))

        # emit warnings for entities with only a few training samples
        for entity_type, count in self.examples_per_entity.items():
            if count < self.MIN_EXAMPLES_PER_ENTITY:
                warnings.warn("Entity '{}' has only {} training examples! "
                              "minimum is {}, training may fail."
                              "".format(entity_type, count,
                                        self.MIN_EXAMPLES_PER_ENTITY))

    def train_test_split(self, train_frac=0.8):
        """Split into a training and test dataset, preserving the fraction of examples per intent."""
        train, test = [], []
        for intent, count in self.examples_per_intent.items():
            ex = [e for e in self.intent_examples if e.data["intent"] == intent]
            random.shuffle(ex)
            n_train = int(count * train_frac)
            train.extend(ex[:n_train])
            test.extend(ex[n_train:])

        data_train = TrainingData(
            train,
            entity_synonyms=self.entity_synonyms,
            regex_features=self.regex_features,
            lookup_tables=self.lookup_tables)
        data_test = TrainingData(
            test,
            entity_synonyms=self.entity_synonyms,
            regex_features=self.regex_features,
            lookup_tables=self.lookup_tables)
        return data_train, data_test

    def print_stats(self):
        logger.info("Training data stats: \n" +
                    "\t- intent examples: {} ({} distinct intents)\n".format(
                            len(self.intent_examples), len(self.intents)) +
                    "\t- Found intents: {}\n".format(
                            list_to_str(self.intents)) +
                    "\t- entity examples: {} ({} distinct entities)\n".format(
                            len(self.entity_examples), len(self.entities)) +
                    "\t- found entities: {}\n".format(
                            list_to_str(self.entities)))
