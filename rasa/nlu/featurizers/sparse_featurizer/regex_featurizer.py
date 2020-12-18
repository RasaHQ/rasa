import logging
import re
from typing import Any, Dict, List, Optional, Text, Type, Tuple
from pathlib import Path
import numpy as np
import scipy.sparse

import rasa.shared.utils.io
import rasa.utils.io
import rasa.nlu.utils.pattern_utils as pattern_utils
from rasa.nlu import utils
from rasa.nlu.components import Component
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.constants import (
    TOKENS_NAMES,
    FEATURIZER_CLASS_ALIAS,
    MIN_ADDITIONAL_REGEX_PATTERNS,
)
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
from rasa.shared.utils.common import lazy_property

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
        # use match word boundaries for lookup table
        "use_word_boundaries": True,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        known_patterns: Optional[List[Dict[Text, Text]]] = None,
        pattern_vocabulary_stats: Optional[Dict[Text, int]] = None,
        finetune_mode: bool = False,
    ) -> None:
        """Constructs new features for regexes and lookup table using regex expressions.

        Args:
            component_config: Configuration for the component
            known_patterns: Regex Patterns the component should pre-load itself with.
            pattern_vocabulary_stats: Statistics about number of pattern slots filled and total number available.
            finetune_mode: Load component in finetune mode.
        """
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
            raise rasa.shared.exceptions.InvalidParameterException(
                f"{self.__class__.__name__} was instantiated with"
                f" `finetune_mode=True` but `pattern_vocabulary_stats`"
                f" was left to `None`. This is invalid since the featurizer"
                f" needs vocabulary statistics to featurize in finetune mode."
            )

    @lazy_property
    def vocabulary_stats(self) -> Dict[Text, int]:
        """Computes total vocabulary size and how much of it is consumed.

        Returns:
            Computed vocabulary size and number of filled vocabulary slots.
        """
        if not self.finetune_mode:
            max_number_patterns = (
                len(self.known_patterns) + self._get_num_additional_slots()
            )
            return {
                "pattern_slots_filled": len(self.known_patterns),
                "max_number_patterns": max_number_patterns,
            }
        else:
            self.pattern_vocabulary_stats["pattern_slots_filled"] = len(
                self.known_patterns
            )
            return self.pattern_vocabulary_stats

    def _merge_new_patterns(self, new_patterns: List[Dict[Text, Text]]) -> None:
        """Updates already known patterns with new patterns extracted from data.

        Args:
            new_patterns: Patterns extracted from training data and to be merged with known patterns.
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
                self.known_patterns.append(extra_pattern)
        if patterns_dropped:
            rasa.shared.utils.io.raise_warning(
                f"The originally trained model was configured to "
                f"handle a maximum number of {max_number_patterns} patterns. "
                f"The current training data exceeds this number as "
                f"there are {len(new_patterns)} patterns in total. "
                f"Some patterns will be dropped and not used for "
                f"featurization. It is advisable to re-train the "
                f"model from scratch."
            )

    def _get_num_additional_slots(self) -> int:
        """Computes number of additional pattern slots available in vocabulary on top of known patterns."""
        if self.number_additional_patterns is None:
            # We take twice the number of currently defined
            # regex patterns as the number of additional
            # vocabulary slots to support if this parameter
            # is not configured by the user. Also, to avoid having
            # to retrain from scratch very often, the default number
            # of additional slots is kept to MIN_ADDITIONAL_SLOTS.
            # This is an empirically tuned number.
            self.number_additional_patterns = max(
                MIN_ADDITIONAL_REGEX_PATTERNS, len(self.known_patterns) * 2
            )
        return self.number_additional_patterns

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Trains the component with all patterns extracted from training data.

        Args:
            training_data: Training data consisting of training examples and patterns available.
            config: NLU Pipeline config
            **kwargs: Any other arguments
        """
        patterns_from_data = pattern_utils.extract_patterns(
            training_data,
            use_lookup_tables=self.component_config["use_lookup_tables"],
            use_regexes=self.component_config["use_regexes"],
            use_word_boundaries=self.component_config["use_word_boundaries"],
        )
        if self.finetune_mode:
            # Merge patterns extracted from data with known patterns
            self._merge_new_patterns(patterns_from_data)
        else:
            self.known_patterns = patterns_from_data

        for example in training_data.training_examples:
            for attribute in [TEXT, RESPONSE, ACTION_TEXT]:
                self._text_features_with_regex(example, attribute)

    def process(self, message: Message, **kwargs: Any) -> None:
        self._text_features_with_regex(message, TEXT)

    def _text_features_with_regex(self, message: Message, attribute: Text) -> None:
        """Helper method to extract features and set them appropriately in the message object.

        Args:
            message: Message to be featurized.
            attribute: Attribute of message to be featurized.
        """
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
        relating the name of the regex to whether it was matched.

        Args:
            message: Message to be featurized.
            attribute: Attribute of message to be featurized.

        Returns:
           Token and sentence level features of message attribute.
        """
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

        max_number_patterns = self.vocabulary_stats["max_number_patterns"]

        sequence_features = np.zeros([sequence_length, max_number_patterns])
        sentence_features = np.zeros([1, max_number_patterns])

        for pattern_index, pattern in enumerate(self.known_patterns):
            matches = re.finditer(
                pattern["pattern"], message.get(attribute), flags=flags
            )
            matches = list(matches)

            for token_index, t in enumerate(tokens):
                patterns = t.get("pattern", default={})
                patterns[pattern["name"]] = False

                for match in matches:
                    if t.start < match.end() and t.end > match.start():
                        patterns[pattern["name"]] = True
                        sequence_features[token_index][pattern_index] = 1.0
                        if attribute in [RESPONSE, TEXT, ACTION_TEXT]:
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
        should_finetune: bool = False,
        **kwargs: Any,
    ) -> "RegexFeaturizer":
        """Loads a previously trained component.

        Args:
            meta: Configuration of trained component.
            model_dir: Path where trained pipeline is stored.
            model_metadata: Metadata for the trained pipeline.
            cached_component: Previously cached component(if any).
            should_finetune: Indicates whether to load the component for further finetuning.
            **kwargs: Any other arguments.
        """
        file_name = meta.get("file")

        patterns_file_name = Path(model_dir) / (file_name + ".patterns.pkl")

        vocabulary_stats_file_name = Path(model_dir) / (
            file_name + ".vocabulary_stats.pkl"
        )

        known_patterns = None
        vocabulary_stats = None
        if patterns_file_name.exists():
            known_patterns = rasa.shared.utils.io.read_json_file(patterns_file_name)
        if vocabulary_stats_file_name.exists():
            vocabulary_stats = rasa.shared.utils.io.read_json_file(
                vocabulary_stats_file_name
            )

        return RegexFeaturizer(
            meta,
            known_patterns=known_patterns,
            pattern_vocabulary_stats=vocabulary_stats,
            finetune_mode=should_finetune,
        )

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.

        Args:
            file_name: Prefix to add to all files stored as part of this component.
            model_dir: Path where files should be stored.

        Returns:
            Metadata necessary to load the model again.
        """
        patterns_file_name = file_name + ".patterns.pkl"
        regex_file = Path(model_dir) / patterns_file_name
        utils.write_json_to_file(regex_file, self.known_patterns, indent=4)
        vocabulary_stats_file_name = file_name + ".vocabulary_stats.pkl"
        vocabulary_file = Path(model_dir) / vocabulary_stats_file_name
        utils.write_json_to_file(vocabulary_file, self.vocabulary_stats, indent=4)

        return {"file": file_name}
