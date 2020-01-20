import logging
from collections import defaultdict, OrderedDict

import numpy as np
import os
import pickle
import typing
import scipy.sparse
from typing import Any, Dict, Optional, Text, List

from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import TOKENS_NAMES, TEXT_ATTRIBUTE, SPARSE_FEATURE_NAMES

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata


class LexicalSyntacticFeaturizer(Featurizer):

    provides = [SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE]]

    requires = [TOKENS_NAMES[TEXT_ATTRIBUTE]]

    defaults = {
        # 'features' is [before, word, after] array with before, word,
        # after holding keys about which features to use for each word,
        # for example, 'title' in array before will have the feature
        # "is the preceding word in title case?"
        # POS features require 'SpacyTokenizer'.
        "features": [
            ["low", "title", "upper"],
            [
                "BOS",
                "EOS",
                "low",
                "prefix5",
                "prefix2",
                "suffix5",
                "suffix3",
                "suffix2",
                "upper",
                "title",
                "digit",
            ],
            ["low", "title", "upper"],
        ]
    }

    function_dict = {
        "low": lambda token: token.text.islower(),
        "title": lambda token: token.text.istitle(),
        "prefix5": lambda token: token.text[:5],
        "prefix2": lambda token: token.text[:2],
        "suffix5": lambda token: token.text[-5:],
        "suffix3": lambda token: token.text[-3:],
        "suffix2": lambda token: token.text[-2:],
        "suffix1": lambda token: token.text[-1:],
        "pos": lambda token: token.data.get("pos") if "pos" in token.data else None,
        "pos2": lambda token: token.data.get("pos")[:2]
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

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        self.feature_to_idx_dict = self._create_feature_to_idx_dict(training_data)

        for example in training_data.training_examples:
            self._create_text_features(example)

    def process(self, message: Message, **kwargs: Any) -> None:
        self._create_text_features(message)

    def _create_text_features(self, message: Message) -> None:
        """Convert incoming messages into sparse features using the configured
        features."""

        # [:-1] to remove CLS token
        tokens = message.get(TOKENS_NAMES[TEXT_ATTRIBUTE])[:-1]

        features = self._tokens_to_features(tokens)
        features = self._features_to_one_hot(features)
        features = self._combine_with_existing_sparse_features(
            message, features, feature_name=SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE]
        )
        message.set(SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE], features)

    def _features_to_one_hot(
        self, features: List[Dict[Text, Any]]
    ) -> scipy.sparse.spmatrix:
        """Convert the word features into a one-hot presentation using the indices
        in the feature-to-idx dictionary."""

        vec = self._initialize_feature_vector(len(features))

        for word_idx, features in enumerate(features):
            for feature_key, feature_value in features.items():
                if (
                    feature_key in self.feature_to_idx_dict
                    and str(feature_value) in self.feature_to_idx_dict[feature_key]
                ):
                    feature_idx = self.feature_to_idx_dict[feature_key][
                        str(feature_value)
                    ]
                    vec[word_idx][feature_idx] = 1

        # set vector of CLS token to sum of everything
        vec[-1] = np.sum(vec, axis=0)

        return scipy.sparse.coo_matrix(vec)

    def _initialize_feature_vector(self, number_of_tokens: int) -> np.ndarray:
        """Initialize a feature vector of size number-of-tokens x number-of-features
        with zeros."""

        number_of_features = sum(
            [
                len(feature_values.values())
                for feature_values in self.feature_to_idx_dict.values()
            ]
        )
        # +1 for the CLS token
        return np.zeros([number_of_tokens + 1, number_of_features])

    def _create_feature_to_idx_dict(
        self, training_data: TrainingData
    ) -> Dict[Text, Dict[Text, int]]:
        """Create dictionary of all feature values.

        Each feature key, defined in the component configuration, points to
        different feature values and their indices in the overall resulting
        feature vector.
        """

        # get all possible feature values
        features = []
        for example in training_data.training_examples:
            # [:-1] to remove CLS token
            tokens = example.get(TOKENS_NAMES[TEXT_ATTRIBUTE])[:-1]
            features.append(self._tokens_to_features(tokens))

        # build vocabulary of features
        feature_vocabulary = self._build_feature_vocabulary(features)

        # assign a unique index to each feature value
        return self._map_features_to_indices(feature_vocabulary)

    @staticmethod
    def _map_features_to_indices(
        feature_vocabulary: Dict[Text, List[Text]]
    ) -> Dict[Text, Dict[Text:int]]:
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

        for sent_features in features:
            for word_features in sent_features:
                for feature_name, feature_value in word_features.items():
                    feature_vocabulary[feature_name].add(feature_value)

        # sort items to ensure same order every time (for tests)
        feature_vocabulary = OrderedDict(sorted(feature_vocabulary.items()))

        return feature_vocabulary

    def _tokens_to_features(self, tokens: List[Token]) -> List[Dict[Text, Any]]:
        """Convert words into discrete features."""

        configured_features = self.component_config["features"]
        features = []

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
                    token_features[prefix + ":" + feature] = self._get_feature_value(
                        feature, token, token_idx, pointer_position, len(tokens)
                    )

            features.append(token_features)

        return features

    def _get_feature_value(
        self,
        feature: Text,
        token: Token,
        token_idx: int,
        pointer_position: int,
        token_length: int,
    ):
        if feature == "EOS":
            return token_idx + pointer_position == token_length - 1

        if feature == "BOS":
            return token_idx + pointer_position == 0

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
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["LexicalSyntacticFeaturizer"] = None,
        **kwargs: Any,
    ) -> "LexicalSyntacticFeaturizer":

        file_name = meta.get("file")

        with open(
            os.path.join(model_dir, file_name + ".feature_to_idx_dict.pkl"), "rb"
        ) as f:
            feature_to_idx_dict = pickle.load(f)

        return LexicalSyntacticFeaturizer(meta, feature_to_idx_dict=feature_to_idx_dict)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        with open(
            os.path.join(model_dir, file_name + ".feature_to_idx_dict.pkl"), "wb"
        ) as f:
            pickle.dump(self.feature_to_idx_dict, f)

        return {"file": file_name}
