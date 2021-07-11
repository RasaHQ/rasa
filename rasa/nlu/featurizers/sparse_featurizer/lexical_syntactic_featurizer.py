import logging
from collections import defaultdict, OrderedDict
from pathlib import Path

import numpy as np
from typing import Any, Dict, Optional, Text, List, Type, Union, Callable

from rasa.nlu.tokenizers.spacy_tokenizer import POS_TAG_KEY
from rasa.shared.constants import DOCS_URL_COMPONENTS
from rasa.nlu.components import Component
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from rasa.nlu.featurizers.featurizer import SparseFeaturizer
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import TOKENS_NAMES, FEATURIZER_CLASS_ALIAS
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SEQUENCE

from rasa.nlu.model import Metadata
import rasa.utils.io as io_utils

logger = logging.getLogger(__name__)

END_OF_SENTENCE = "EOS"
BEGIN_OF_SENTENCE = "BOS"


class LexicalSyntacticFeaturizer(SparseFeaturizer):
    """Creates features for entity extraction.

    Moves with a sliding window over every token in the user message and creates
    features according to the configuration.
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Tokenizer]

    defaults = {
        # 'features' is [before, word, after] array with before, word,
        # after holding keys about which features to use for each word,
        # for example, 'title' in array before will have the feature
        # "is the preceding word in title case?"
        # POS features require 'SpacyTokenizer'.
        "features": [
            ["low", "title", "upper"],
            ["BOS", "EOS", "low", "upper", "title", "digit"],
            ["low", "title", "upper"],
        ]
    }

    function_dict: Dict[Text, Callable[[Token], Union[bool, Text, None]]] = {
        "low": lambda token: token.text.islower(),
        "title": lambda token: token.text.istitle(),
        "prefix5": lambda token: token.text[:5],
        "prefix2": lambda token: token.text[:2],
        "suffix5": lambda token: token.text[-5:],
        "suffix3": lambda token: token.text[-3:],
        "suffix2": lambda token: token.text[-2:],
        "suffix1": lambda token: token.text[-1:],
        "pos": lambda token: token.data.get(POS_TAG_KEY)
        if POS_TAG_KEY in token.data
        else None,
        "pos2": lambda token: token.data.get(POS_TAG_KEY, [])[:2]
        if "pos" in token.data
        else None,
        "upper": lambda token: token.text.isupper(),
        "digit": lambda token: token.text.isdigit(),
    }

    def __init__(
        self,
        component_config: Dict[Text, Any],
        feature_to_idx_dict: Optional[Dict[Text, Any]] = None,
    ):
        super().__init__(component_config)

        self.feature_to_idx_dict = feature_to_idx_dict or {}
        self.number_of_features = self._calculate_number_of_features()

    def _calculate_number_of_features(self) -> int:
        return sum(
            [
                len(feature_values.values())
                for feature_values in self.feature_to_idx_dict.values()
            ]
        )

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.feature_to_idx_dict = self._create_feature_to_idx_dict(training_data)
        self.number_of_features = self._calculate_number_of_features()

        for example in training_data.training_examples:
            self._create_sparse_features(example)

    def process(self, message: Message, **kwargs: Any) -> None:
        self._create_sparse_features(message)

    def _create_feature_to_idx_dict(
        self, training_data: TrainingData
    ) -> Dict[Text, Dict[Text, int]]:
        """Create dictionary of all feature values.

        Each feature key, defined in the component configuration, points to
        different feature values and their indices in the overall resulting
        feature vector.
        """

        # get all possible feature values
        all_features = []
        for example in training_data.training_examples:
            tokens = example.get(TOKENS_NAMES[TEXT])
            if tokens:
                all_features.append(self._tokens_to_features(tokens))

        # build vocabulary of features
        feature_vocabulary = self._build_feature_vocabulary(all_features)

        # assign a unique index to each feature value
        return self._map_features_to_indices(feature_vocabulary)

    @staticmethod
    def _map_features_to_indices(
        feature_vocabulary: Dict[Text, List[Text]]
    ) -> Dict[Text, Dict[Text, int]]:
        feature_to_idx_dict = {}
        offset = 0

        for feature_name, feature_values in feature_vocabulary.items():
            feature_to_idx_dict[feature_name] = {
                str(feature_value): feature_idx
                for feature_idx, feature_value in enumerate(
                    sorted(feature_values), start=offset
                )
            }
            offset += len(feature_values)

        return feature_to_idx_dict

    @staticmethod
    def _build_feature_vocabulary(
        features: List[List[Dict[Text, Any]]]
    ) -> Dict[Text, List[Text]]:
        feature_vocabulary = defaultdict(set)

        for sentence_features in features:
            for token_features in sentence_features:
                for feature_name, feature_value in token_features.items():
                    feature_vocabulary[feature_name].add(feature_value)

        # sort items to ensure same order every time (for tests)
        feature_vocabulary = OrderedDict(sorted(feature_vocabulary.items()))

        return feature_vocabulary

    def _create_sparse_features(self, message: Message) -> None:
        """Convert incoming messages into sparse features using the configured
        features."""
        import scipy.sparse

        tokens = message.get(TOKENS_NAMES[TEXT])
        # this check is required because there might be training data examples without
        # TEXT, e.g., `Message("", {action_name: "action_listen"})`
        if tokens:

            sentence_features = self._tokens_to_features(tokens)
            one_hot_seq_feature_vector = self._features_to_one_hot(sentence_features)

            sequence_features = scipy.sparse.coo_matrix(one_hot_seq_feature_vector)

            final_sequence_features = Features(
                sequence_features,
                FEATURE_TYPE_SEQUENCE,
                TEXT,
                self.component_config[FEATURIZER_CLASS_ALIAS],
            )
            message.add_features(final_sequence_features)

    def _tokens_to_features(self, tokens: List[Token]) -> List[Dict[Text, Any]]:
        """Convert words into discrete features."""

        configured_features = self.component_config["features"]
        sentence_features = []

        for token_idx in range(len(tokens)):
            # get the window size (e.g. before, word, after) of the configured features
            # in case of an even number we will look at one more word before,
            # e.g. window size 4 will result in a window range of
            # [-2, -1, 0, 1] (0 = current word in sentence)
            window_size = len(configured_features)
            half_window_size = window_size // 2
            window_range = range(-half_window_size, half_window_size + window_size % 2)

            prefixes = [str(i) for i in window_range]

            token_features = {}

            for pointer_position in window_range:
                current_idx = token_idx + pointer_position

                # skip, if current_idx is pointing to a non-existing token
                if current_idx < 0 or current_idx >= len(tokens):
                    continue

                token = tokens[token_idx + pointer_position]

                current_feature_idx = pointer_position + half_window_size
                prefix = prefixes[current_feature_idx]

                for feature in configured_features[current_feature_idx]:
                    token_features[f"{prefix}:{feature}"] = self._get_feature_value(
                        feature, token, token_idx, pointer_position, len(tokens)
                    )

            sentence_features.append(token_features)

        return sentence_features

    def _features_to_one_hot(
        self, sentence_features: List[Dict[Text, Any]]
    ) -> np.ndarray:
        """Convert the word features into a one-hot presentation using the indices
        in the feature-to-idx dictionary."""

        one_hot_seq_feature_vector = np.zeros(
            [len(sentence_features), self.number_of_features]
        )

        for token_idx, token_features in enumerate(sentence_features):
            for feature_name, feature_value in token_features.items():
                feature_value_str = str(feature_value)
                if (
                    feature_name in self.feature_to_idx_dict
                    and feature_value_str in self.feature_to_idx_dict[feature_name]
                ):
                    feature_idx = self.feature_to_idx_dict[feature_name][
                        feature_value_str
                    ]
                    one_hot_seq_feature_vector[token_idx][feature_idx] = 1

        return one_hot_seq_feature_vector

    def _get_feature_value(
        self,
        feature: Text,
        token: Token,
        token_idx: int,
        pointer_position: int,
        token_length: int,
    ) -> Union[bool, int, Text]:
        if feature == END_OF_SENTENCE:
            return token_idx + pointer_position == token_length - 1

        if feature == BEGIN_OF_SENTENCE:
            return token_idx + pointer_position == 0

        if feature not in self.function_dict:
            raise ValueError(
                f"Configured feature '{feature}' not valid. Please check "
                f"'{DOCS_URL_COMPONENTS}' for valid configuration parameters."
            )

        value = self.function_dict[feature](token)
        if value is None:
            logger.debug(
                f"Invalid value '{value}' for feature '{feature}'."
                f" Feature is ignored."
            )
        return value

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text,
        model_metadata: Optional[Metadata] = None,
        cached_component: Optional["LexicalSyntacticFeaturizer"] = None,
        **kwargs: Any,
    ) -> "LexicalSyntacticFeaturizer":
        """Loads trained component (see parent class for full docstring)."""
        file_name = meta.get("file")

        feature_to_idx_file = Path(model_dir) / f"{file_name}.feature_to_idx_dict.pkl"
        feature_to_idx_dict = io_utils.json_unpickle(feature_to_idx_file)

        return LexicalSyntacticFeaturizer(meta, feature_to_idx_dict=feature_to_idx_dict)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""

        feature_to_idx_file = Path(model_dir) / f"{file_name}.feature_to_idx_dict.pkl"
        io_utils.json_pickle(feature_to_idx_file, self.feature_to_idx_dict)

        return {"file": file_name}
