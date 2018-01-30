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

    def merge(self, others):
        """Merges a TrainingData instance with others and creates a new one."""
        common_examples = deepcopy(self.training_examples)
        entity_synonyms = self.entity_synonyms.copy()
        regex_features = deepcopy(self.regex_features)

        def extend_with(training_data):
            common_examples.extend(deepcopy(training_data.training_examples))
            regex_features.extend(deepcopy(training_data.regex_features))

            for text, syn in training_data.entity_synonyms.items():
                if text in entity_synonyms and entity_synonyms[text] != syn:
                    logger.warning("Inconsistent entity synonyms, overwriting {0}->{1}"
                                   "with {0}->{2} during merge".format(text, entity_synonyms[text], syn))

            entity_synonyms.update(training_data.entity_synonyms)

        if isinstance(others, TrainingData):
            extend_with(others)
        elif isinstance(others, list) and all([isinstance(e, TrainingData) for e in others]):
            for o in others:
                extend_with(o)
        else:
            raise ValueError("Merging requires another TrainingData instance or a list of them")

        return TrainingData(common_examples, entity_synonyms, regex_features)

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
        """Represent this set of training examples as markdown adding
        the passed meta information."""
        from rasa_nlu.training_data import MarkdownWriter
        mdw = MarkdownWriter()
        return mdw.to_markdown(self)

    @staticmethod
    def from_markdown(markdown_file):
        """Creates a TrainingData object from a markdown file."""
        from rasa_nlu.training_data import MarkdownReader
        mdr = MarkdownReader()
        return mdr.read(markdown_file)

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
