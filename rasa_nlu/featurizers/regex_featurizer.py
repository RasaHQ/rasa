from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os
import re
import warnings

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Text

from rasa_nlu import utils
from rasa_nlu.config import RasaNLUConfig
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData

import numpy as np

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa_nlu.model import Metadata


class RegexFeaturizer(Featurizer):
    name = "intent_entity_featurizer_regex"

    provides = ["text_features"]

    requires = ["tokens"]

    def __init__(self, known_patterns=None):
        super(RegexFeaturizer, self).__init__()

        self.known_patterns = known_patterns if known_patterns else []

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["numpy"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData, RasaNLUConfig, **Any) -> None

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

        if model_dir and model_metadata.get("regex_featurizer"):
            regex_file = os.path.join(model_dir,
                                      model_metadata.get("regex_featurizer"))
            if os.path.isfile(regex_file):
                known_patterns = utils.read_json_file(regex_file)
                return RegexFeaturizer(known_patterns)
            else:
                warnings.warn("Failed to load regex pattern file "
                              "'{}'".format(regex_file))
        return RegexFeaturizer()

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again."""

        if self.known_patterns:
            regex_file = os.path.join(model_dir, "regex_featurizer.json")
            utils.write_json_to_file(regex_file, self.known_patterns, indent=4)
            return {"regex_featurizer": "regex_featurizer.json"}
        else:
            return {"regex_featurizer": None}
