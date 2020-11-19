import logging
import os
import re
from typing import Any, Dict, List, Optional, Text, Type, Tuple

import numpy as np
import scipy.sparse

import rasa.shared.utils.io
import rasa.utils.io
import rasa.nlu.utils.pattern_utils as pattern_utils
from rasa.nlu import utils
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.constants import TOKENS_NAMES, FEATURIZER_CLASS_ALIAS
from rasa.shared.nlu.constants import (
    TEXT,
    RESPONSE,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
    ACTION_TEXT,
)
from rasa.nlu.featurizers.featurizer import SparseFeaturizer
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.model import Metadata
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message

logger = logging.getLogger(__name__)


class RegexFeaturizer(SparseFeaturizer):
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    defaults = {
        # text will be processed with case sensitive as default
        "case_sensitive": True,
        # use lookup tables to generate features
        "use_lookup_tables": True,
        # use regexes to generate features
        "use_regexes": True,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        known_patterns: Optional[List[Dict[Text, Text]]] = None,
    ) -> None:

        super().__init__(component_config)

        self.known_patterns = known_patterns if known_patterns else []
        self.case_sensitive = self.component_config["case_sensitive"]

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        self.known_patterns = pattern_utils.extract_patterns(
            training_data,
            use_lookup_tables=self.component_config["use_lookup_tables"],
            use_regexes=self.component_config["use_regexes"],
        )

        for example in training_data.training_examples:
            for attribute in [TEXT, RESPONSE, ACTION_TEXT]:
                self._text_features_with_regex(example, attribute)

    def process(self, message: Message, **kwargs: Any) -> None:
        self._text_features_with_regex(message, TEXT)

    def _text_features_with_regex(self, message: Message, attribute: Text) -> None:
        if self.known_patterns:
            sequence_features, sentence_features = self._features_for_patterns(
                message, attribute
            )

            if sequence_features is not None:
                final_sequence_features = Features(
                    sequence_features,
                    FEATURE_TYPE_SEQUENCE,
                    attribute,
                    self.component_config[FEATURIZER_CLASS_ALIAS],
                )
                message.add_features(final_sequence_features)

            if sentence_features is not None:
                final_sentence_features = Features(
                    sentence_features,
                    FEATURE_TYPE_SENTENCE,
                    attribute,
                    self.component_config[FEATURIZER_CLASS_ALIAS],
                )
                message.add_features(final_sentence_features)

    def _features_for_patterns(
        self, message: Message, attribute: Text
    ) -> Tuple[Optional[scipy.sparse.coo_matrix], Optional[scipy.sparse.coo_matrix]]:
        """Checks which known patterns match the message.
        Given a sentence, returns a vector of {1,0} values indicating which
        regexes did match. Furthermore, if the
        message is tokenized, the function will mark all tokens with a dict
        relating the name of the regex to whether it was matched."""

        # Attribute not set (e.g. response not present)
        if not message.get(attribute):
            return None, None

        tokens = message.get(TOKENS_NAMES[attribute], [])

        if not tokens:
            # nothing to featurize
            return None, None

        flags = 0  # default flag
        if not self.case_sensitive:
            flags = re.IGNORECASE

        sequence_length = len(tokens)

        sequence_features = np.zeros([sequence_length, len(self.known_patterns)])
        sentence_features = np.zeros([1, len(self.known_patterns)])

        for pattern_index, pattern in enumerate(self.known_patterns):
            matches = re.finditer(pattern["pattern"], message.get(TEXT), flags=flags)
            matches = list(matches)

            for token_index, t in enumerate(tokens):
                patterns = t.get("pattern", default={})
                patterns[pattern["name"]] = False

                for match in matches:
                    if t.start < match.end() and t.end > match.start():
                        patterns[pattern["name"]] = True
                        sequence_features[token_index][pattern_index] = 1.0
                        if attribute in [RESPONSE, TEXT]:
                            # sentence vector should contain all patterns
                            sentence_features[0][pattern_index] = 1.0

                t.set("pattern", patterns)

        return (
            scipy.sparse.coo_matrix(sequence_features),
            scipy.sparse.coo_matrix(sentence_features),
        )

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["RegexFeaturizer"] = None,
        **kwargs: Any,
    ) -> "RegexFeaturizer":

        file_name = meta.get("file")
        regex_file = os.path.join(model_dir, file_name)

        if os.path.exists(regex_file):
            known_patterns = rasa.shared.utils.io.read_json_file(regex_file)
            return RegexFeaturizer(meta, known_patterns=known_patterns)
        else:
            return RegexFeaturizer(meta)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        file_name = file_name + ".pkl"
        regex_file = os.path.join(model_dir, file_name)
        utils.write_json_to_file(regex_file, self.known_patterns, indent=4)

        return {"file": file_name}
