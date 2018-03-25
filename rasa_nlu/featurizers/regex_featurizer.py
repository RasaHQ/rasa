from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import re
import warnings

import typing
from typing import Any, Dict, List, Optional, Text

from rasa_nlu import utils
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

import numpy as np

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_nlu.model import Metadata


REGEX_FEATURIZER_FILE_NAME = "regex_featurizer.json"


class RegexFeaturizer(Featurizer):
    name = "intent_entity_featurizer_regex"

    provides = ["text_features"]

    requires = ["tokens"]

    def __init__(self, component_config=None, known_patterns=None):
        super(RegexFeaturizer, self).__init__(component_config)

        self.known_patterns = known_patterns if known_patterns else []

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None

        for example in training_data.regex_features:
            self.known_patterns.append(example)

        for example in training_data.training_examples:
            updated = self._text_features_with_regex(example)
            example.set("text_features", updated)

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        updated = self._text_features_with_regex(message)
        message.set("text_features", updated)

    def _text_features_with_regex(self, message):
        if self.known_patterns is not None:
            extras = self.features_for_patterns(message)
            return self._combine_with_existing_text_features(message, extras)
        else:
            return message.get("text_features")

    def features_for_patterns(self, message):
        """Checks which known patterns match the message.

        Given a sentence, returns a vector of {1,0} values indicating which
        regexes did match. Furthermore, if the
        message is tokenized, the function will mark the matching regex on
        the tokens that are part of the match."""

        found = []
        for i, exp in enumerate(self.known_patterns):
            match = re.search(exp["pattern"], message.text)
            if match is not None:
                for t in message.get("tokens", []):
                    if t.offset < match.end() and t.end > match.start():
                        t.set("pattern", i)
                found.append(1.0)
            else:
                found.append(0.0)
        return np.array(found)

    @classmethod
    def load(cls,
             model_dir=None,   # type: Optional[Text]
             model_metadata=None,   # type: Optional[Metadata]
             cached_component=None,   # type: Optional[RegexFeaturizer]
             **kwargs  # type: **Any
             ):
        # type: (...) -> RegexFeaturizer

        meta = model_metadata.for_component(cls.name)
        file_name = meta.get("regex_file", REGEX_FEATURIZER_FILE_NAME)
        regex_file = os.path.join(model_dir, file_name)

        if os.path.exists(regex_file):
            known_patterns = utils.read_json_file(regex_file)
            return RegexFeaturizer(meta, known_patterns=known_patterns)
        else:
            return RegexFeaturizer(meta)

    def persist(self, model_dir):
        # type: (Text) -> Optional[Dict[Text, Any]]
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again."""

        if self.known_patterns:
            regex_file = os.path.join(model_dir, REGEX_FEATURIZER_FILE_NAME)
            utils.write_json_to_file(regex_file, self.known_patterns, indent=4)

        return {"regex_file": REGEX_FEATURIZER_FILE_NAME}
