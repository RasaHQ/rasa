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
        # use match word boundaries for lookup table
        "use_word_boundaries": True,
        # Additional number of patterns to consider
        # for incremental training
        "number_additional_patterns": None,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        known_patterns: Optional[List[Dict[Text, Text]]] = None,
        finetune_mode: bool = False,
    ) -> None:
        """Constructs new features for regexes and lookup table using regex expressions.

        Args:
            component_config: Configuration for the component
            known_patterns: Regex Patterns the component should pre-load itself with.
            finetune_mode: Load component in finetune mode.
        """
        super().__init__(component_config)

        self.known_patterns = known_patterns if known_patterns else []
        self.case_sensitive = self.component_config["case_sensitive"]
        self.finetune_mode = finetune_mode
        if self.component_config["number_additional_patterns"]:
            rasa.shared.utils.io.raise_deprecation_warning(
                "The parameter `number_additional_patterns` has been deprecated "
                "since the pipeline does not create an extra buffer for new vocabulary "
                "anymore. Any value assigned to this parameter will be ignored. "
                "You can omit specifying `number_additional_patterns` in future runs."
            )

    def _merge_new_patterns(self, new_patterns: List[Dict[Text, Text]]) -> None:
        """Updates already known patterns with new patterns extracted from data.

        New patterns should always be added to the end of the existing
        patterns and the order of the existing patterns should not be disturbed.

        Args:
            new_patterns: Patterns extracted from training data and to be merged with
                known patterns.
        """
        pattern_name_index_map = {
            pattern["name"]: index for index, pattern in enumerate(self.known_patterns)
        }
        for extra_pattern in new_patterns:
            new_pattern_name = extra_pattern["name"]

            # Some patterns may have just new examples added
            # to them. These do not count as additional pattern.
            if new_pattern_name in pattern_name_index_map:
                self.known_patterns[pattern_name_index_map[new_pattern_name]][
                    "pattern"
                ] = extra_pattern["pattern"]
            else:
                self.known_patterns.append(extra_pattern)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Trains the component with all patterns extracted from training data.

        Args:
            training_data: Training data consisting of training examples and patterns
                available.
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
        """Helper method to extract features and set them appropriately in the message.

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

        num_patterns = len(self.known_patterns)

        sequence_features = np.zeros([sequence_length, num_patterns])
        sentence_features = np.zeros([1, num_patterns])

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
        model_dir: Text,
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
            should_finetune: Indicates whether to load the component for further
                finetuning.
            **kwargs: Any other arguments.
        """
        file_name = meta.get("file")

        patterns_file_name = Path(model_dir) / (file_name + ".patterns.pkl")

        known_patterns = None
        if patterns_file_name.exists():
            known_patterns = rasa.shared.utils.io.read_json_file(patterns_file_name)

        return RegexFeaturizer(
            meta, known_patterns=known_patterns, finetune_mode=should_finetune,
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

        return {"file": file_name}
