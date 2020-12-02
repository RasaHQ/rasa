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
from rasa.shared.exceptions import RasaException

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
        # Additional number of patterns to consider
        # for incremental training
        "number_additional_patterns": None,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        known_patterns: Optional[List[Dict[Text, Text]]] = None,
        pattern_vocabulary_stats: Optional[Dict[Text, int]] = None,
        finetune_mode: bool = False,
    ) -> None:

        super().__init__(component_config)

        self.known_patterns = known_patterns if known_patterns else []
        self.case_sensitive = self.component_config["case_sensitive"]
        self.number_additional_patterns = self.component_config[
            "number_additional_patterns"
        ]
        self.finetune_mode = finetune_mode
        self.pattern_vocabulary_stats = pattern_vocabulary_stats

        if self.finetune_mode and not self.pattern_vocabulary_stats:
            # If the featurizer is instantiated in finetune mode,
            # the vocabulary stats for it should be known.
            raise RasaException(
                f"{self.__class__.__name__} was instantiated with"
                f" `finetune_mode=True` but `pattern_vocabulary_stats`"
                f" was left to `None`. This is invalid since the featurizer"
                f" needs vocabulary statistics to featurize in finetune mode."
            )

    def _get_vocabulary_stats(self) -> Dict[Text, int]:
        """

        Returns:

        """
        if not self.finetune_mode:
            max_number_patterns = (
                len(self.known_patterns) + self.number_additional_patterns
            )
            return {
                "pattern_slots_filled": len(self.known_patterns),
                "max_number_patterns": max_number_patterns,
            }
        else:
            return self.pattern_vocabulary_stats

    def _merge_with_existing_patterns(self, new_patterns: List[Dict[Text, Text]]):
        """

        Args:
            new_patterns:

        Returns:

        """
        max_number_patterns = self.pattern_vocabulary_stats["max_number_patterns"]
        pattern_name_index_map = {
            pattern["name"]: index for index, pattern in enumerate(self.known_patterns)
        }
        patterns_dropped = False

        for extra_pattern in new_patterns:
            new_pattern_name = extra_pattern["name"]

            # Some patterns may have just new examples added
            # to them. These do not count as additional pattern.
            if new_pattern_name in pattern_name_index_map:
                self.known_patterns[pattern_name_index_map[new_pattern_name]][
                    "pattern"
                ] = extra_pattern["pattern"]
            else:
                if len(self.known_patterns) == max_number_patterns:
                    patterns_dropped = True
                    continue
                self.known_patterns.extend(extra_pattern)
        if patterns_dropped:
            logger.warning(
                f"The originally trained model was configured to "
                f"handle a maximum number of {max_number_patterns} patterns. "
                f"The current training data exceeds this number as "
                f"there are {len(new_patterns)} patterns in total. "
                f"Some patterns will be dropped and not used for "
                f"featurization. It is advisable to re-train the "
                f"model from scratch."
            )

    def _compute_num_additional_slots(self) -> None:
        if self.number_additional_patterns is None:
            self.number_additional_patterns = max(10, len(self.known_patterns) * 2)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        patterns_from_data = pattern_utils.extract_patterns(
            training_data,
            use_lookup_tables=self.component_config["use_lookup_tables"],
            use_regexes=self.component_config["use_regexes"],
        )
        if self.finetune_mode:
            # Merge patterns extracted from data with known patterns
            self._merge_with_existing_patterns(patterns_from_data)
        else:
            self.known_patterns = patterns_from_data
            self._compute_num_additional_slots()

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

        max_number_patterns = self._get_vocabulary_stats()["max_number_patterns"]

        sequence_features = np.zeros([sequence_length, max_number_patterns])
        sentence_features = np.zeros([1, max_number_patterns])

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

        finetune_mode = kwargs.pop("finetune_mode", False)

        file_name = meta.get("file")

        patterns_file_name = file_name + ".patterns.pkl"
        regex_file = os.path.join(model_dir, patterns_file_name)

        vocabulary_stats_file_name = file_name + ".vocabulary_stats.pkl"
        vocabulary_file = os.path.join(model_dir, vocabulary_stats_file_name)

        known_patterns = None
        vocabulary_stats = None
        if os.path.exists(regex_file):
            known_patterns = rasa.shared.utils.io.read_json_file(regex_file)
        if os.path.exists(vocabulary_file):
            vocabulary_stats = rasa.shared.utils.io.read_json_file(vocabulary_file)

        return RegexFeaturizer(
            meta,
            known_patterns=known_patterns,
            pattern_vocabulary_stats=vocabulary_stats,
            finetune_mode=finetune_mode,
        )

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        patterns_file_name = file_name + ".patterns.pkl"
        regex_file = os.path.join(model_dir, patterns_file_name)
        utils.write_json_to_file(regex_file, self.known_patterns, indent=4)
        vocabulary_stats_file_name = file_name + ".vocabulary_stats.pkl"
        vocabulary_file = os.path.join(model_dir, vocabulary_stats_file_name)
        utils.write_json_to_file(
            vocabulary_file, self._get_vocabulary_stats(), indent=4
        )

        return {"file": file_name}
