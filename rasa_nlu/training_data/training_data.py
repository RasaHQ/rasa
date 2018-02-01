# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import warnings
from itertools import groupby

from copy import deepcopy
from builtins import object, str
from collections import defaultdict

from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu.utils import lazyproperty, write_to_file
from rasa_nlu.utils import list_to_str
from rasa_nlu.utils import json_to_string
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
                 regex_features=None):
        # type: (Optional[List[Message]], Optional[Dict[Text, Text]]) -> None

        if training_examples:
            self.training_examples = self.sanitize_examples(training_examples)
        else:
            self.training_examples = []
        self.entity_synonyms = entity_synonyms if entity_synonyms else {}
        self.regex_features = regex_features if regex_features else []
        self.sort_regex_features()

        self.validate()

    def merge(self, *others):
        """Merges the TrainingData instance with others and creates a new one."""
        training_examples = deepcopy(self.training_examples)
        entity_synonyms = self.entity_synonyms.copy()
        regex_features = deepcopy(self.regex_features)

        for o in others:
            training_examples.extend(deepcopy(o.training_examples))
            regex_features.extend(deepcopy(o.regex_features))

            for text, syn in o.entity_synonyms.items():
                check_duplicate_synonym(entity_synonyms, text, syn, "merging training data")

            entity_synonyms.update(o.entity_synonyms)

        return TrainingData(training_examples, entity_synonyms, regex_features)

    def sanitize_examples(self, examples):
        # type: (List[Message]) -> List[Message]
        """Makes sure the training data is clean.

        removes trailing whitespaces from intent annotations."""

        for e in examples:
            if e.get("intent") is not None:
                e.set("intent", e.get("intent").strip())
        return examples

    @lazyproperty
    def intent_examples(self):
        # type: () -> List[Message]
        return [e
                for e in self.training_examples
                if e.get("intent") is not None]

    @lazyproperty
    def entity_examples(self):
        # type: () -> List[Message]
        return [e
                for e in self.training_examples
                if e.get("entities") is not None]

    @lazyproperty
    def num_entity_examples(self):
        # type: () -> int
        """Returns the number of proper entity training examples
        (containing at least one annotated entity)."""

        return len([e
                    for e in self.training_examples
                    if len(e.get("entities", [])) > 0])

    @lazyproperty
    def num_intent_examples(self):
        # type: () -> int
        """Returns the number of intent examples."""

        return len(self.intent_examples)

    def sort_regex_features(self):
        """Sorts regex features lexicographically by name+pattern"""
        self.regex_features = sorted(self.regex_features,
                                     key=lambda e: "{}+{}".format(e['name'], e['pattern']))

    #TODO: extract into RasaJson writer
    def as_json(self, **kwargs):
        # type: (**Any) -> str
        """Represent this set of training examples as json adding
        the passed meta information."""

        js_entity_synonyms = defaultdict(list)
        for k, v in self.entity_synonyms.items():
            if k != v:
                js_entity_synonyms[v].append(k)

        formatted_synonyms = [{'value': value, 'synonyms': syns}
                              for value, syns in js_entity_synonyms.items()]

        formatted_examples = [example.as_dict()
                              for example in self.training_examples]

        return str(json_to_string({
            "rasa_nlu_data": {
                "common_examples": formatted_examples,
                "regex_features": self.regex_features,
                "entity_synonyms": formatted_synonyms
            }
        }, **kwargs))

    def as_markdown(self):
        # type: () -> str
        """Generates the markdown representation of the TrainingData."""
        from rasa_nlu.training_data.formats import MarkdownWriter
        return self._as_format(MarkdownWriter)

    def _as_format(self, writer_clz):
        """Generates a string representation of the TrainingData given a writer class."""
        writer = writer_clz()
        return writer.dumps(self)

    def persist(self, dir_name):
        # type: (Text) -> Dict[Text, Any]
        """Persists this training data to disk and returns necessary
        information to load it again."""

        data_file = os.path.join(dir_name, "training_data.json")
        write_to_file(data_file, self.as_json(indent=2))

        return {
            "training_data": "training_data.json"
        }

    def sorted_entity_examples(self):
        # type: () -> List[Message]
        """Sorts the entity examples by the annotated entity."""

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
        """Ensures that the loaded training data is valid, e.g.

        has a minimum of certain training examples."""

        logger.debug("Validating training data...")
        examples = self.sorted_intent_examples()
        different_intents = []
        for intent, group in groupby(examples, lambda e: e.get("intent")):
            size = len(list(group))
            different_intents.append(intent)
            if intent == "":
                warnings.warn("Found empty intent, please check your "
                              "training data. This may result in wrong "
                              "intent predictions.")
            if size < self.MIN_EXAMPLES_PER_INTENT:
                template = ("Intent '{}' has only {} training examples! "
                            "minimum is {}, training may fail.")
                warnings.warn(template.format(repr(intent),
                                              size,
                                              self.MIN_EXAMPLES_PER_INTENT))

        different_entities = []
        for entity, group in groupby(self.sorted_entity_examples(),
                                     lambda e: e["entity"]):
            size = len(list(group))
            different_entities.append(entity)
            if size < self.MIN_EXAMPLES_PER_ENTITY:
                template = ("Entity '{}' has only {} training examples! "
                            "minimum is {}, training may fail.")
                warnings.warn(template.format(repr(entity), size,
                                              self.MIN_EXAMPLES_PER_ENTITY))

        logger.info("Training data stats: \n" +
                    "\t- intent examples: {} ({} distinct intents)\n".format(
                            self.num_intent_examples, len(different_intents)) +
                    "\t- found intents: {}\n".format(
                            list_to_str(different_intents)) +
                    "\t- entity examples: {} ({} distinct entities)\n".format(
                            self.num_entity_examples, len(different_entities)) +
                    "\t- found entities: {}\n".format(
                            list_to_str(different_entities)))
