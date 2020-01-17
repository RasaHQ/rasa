import logging
from collections import defaultdict, OrderedDict

import numpy as np
import os
import pickle
import typing
import scipy.sparse
from typing import Any, Dict, Optional, Text, List

from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    TOKENS_NAMES,
    TEXT_ATTRIBUTE,
    SPARSE_FEATURE_NAMES,
    SPACY_DOCS,
)

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.nlu.model import Metadata

try:
    import spacy
except ImportError:
    spacy = None


class Word(typing.NamedTuple):
    text: Text
    pos_tag: Text


class LexicalSyntacticFeaturizer(Featurizer):

    provides = [SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE]]

    requires = [TOKENS_NAMES[TEXT_ATTRIBUTE]]

    defaults = {
        # 'features' is [before, word, after] array with before, word,
        # after holding keys about which features to use for each word,
        # for example, 'title' in array before will have the feature
        # "is the preceding word in title case?"
        # POS features require spaCy to be installed
        "features": [
            ["low", "title", "upper"],
            [
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
        "low": lambda word: word.text.islower(),
        "title": lambda word: word.text.istitle(),
        "prefix5": lambda word: word.text[:5],
        "prefix2": lambda word: word.text[:2],
        "suffix5": lambda word: word.text[-5:],
        "suffix3": lambda word: word.text[-3:],
        "suffix2": lambda word: word.text[-2:],
        "suffix1": lambda word: word.text[-1:],
        "pos": lambda word: word.pos_tag,
        "pos2": lambda word: word.pos_tag[:2],
        "upper": lambda word: word.text.isupper(),
        "digit": lambda word: word.text.isdigit(),
    }

    def __init__(
        self,
        component_config: Dict[Text, Any],
        feature_to_idx_dict: Optional[Dict[Text, Any]] = None,
    ):
        super().__init__(component_config)

        if feature_to_idx_dict is None:
            self.feature_to_idx_dict = {}
        else:
            self.feature_to_idx_dict = feature_to_idx_dict

        self._check_pos_features_and_spacy()

    def _check_pos_features_and_spacy(self):
        import itertools

        features = set(
            itertools.chain.from_iterable(self.component_config.get("features", []))
        )
        self.pos_features = "pos" in features or "pos2" in features

        if self.pos_features and spacy is None:
            raise ImportError(
                "Failed to import `spaCy`. `spaCy` is required for POS features. "
                "See https://spacy.io/usage/ for installation instructions."
            )

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

        words = self._convert_to_words(message)
        word_features = self._words_to_features(words)
        features = self._features_to_one_hot(word_features)
        features = self._combine_with_existing_sparse_features(
            message, features, feature_name=SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE]
        )
        message.set(SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE], features)

    def _features_to_one_hot(
        self, word_features: List[Dict[Text, Any]]
    ) -> scipy.sparse.spmatrix:
        """Convert the word features into a one-hot presentation using the indices
        in the feature-to-idx dictionary."""

        vec = self._initialize_feature_vector(len(word_features))

        for word_idx, word_features in enumerate(word_features):
            for feature_key, feature_value in word_features.items():
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
        feature vector."""

        # get all possible feature values
        features = []
        for example in training_data.training_examples:
            words = self._convert_to_words(example)
            features.append(self._words_to_features(words))

        # build vocabulary of features
        feature_vocabulary = defaultdict(set)
        for sent_features in features:
            for word_features in sent_features:
                for feature_name, feature_value in word_features.items():
                    feature_vocabulary[feature_name].add(feature_value)

        feature_vocabulary = OrderedDict(sorted(feature_vocabulary.items()))

        # assign a unique index to each feature value
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

    def _words_to_features(self, words: List[Word]) -> List[Dict[Text, Any]]:
        """Convert words into discrete features."""

        configured_features = self.component_config["features"]
        words_features = []

        for word_idx in range(len(words)):
            # get the window size (e.g. before, word, after) of the configured features
            # in case of an even number we will look at one more word before,
            # e.g. window size 4 will result in a window range of
            # [-2, -1, 0, 1] (0 = current word in sentence)
            window_size = len(configured_features)
            half_window_size = window_size // 2
            window_range = range(-half_window_size, half_window_size + window_size % 2)

            prefixes = [str(i) for i in window_range]

            word_features = {}

            for pointer_position in window_range:
                current_idx = word_idx + pointer_position

                # skip, if current_idx is pointing to a non-existing word
                if current_idx < 0 or current_idx >= len(words):
                    continue

                # check if we are at the start or at the end
                if word_idx == len(words) - 1 and pointer_position == 0:
                    word_features["EOS"] = True
                elif word_idx == 0 and pointer_position == 0:
                    word_features["BOS"] = True

                word = words[word_idx + pointer_position]

                current_feature_idx = pointer_position + half_window_size
                prefix = prefixes[current_feature_idx]
                features = configured_features[current_feature_idx]

                for feature in features:
                    # append each feature to a feature vector
                    value = self.function_dict[feature](word)
                    word_features[prefix + ":" + feature] = value

            words_features.append(word_features)

        return words_features

    def _convert_to_words(self, message: Message) -> List[Word]:
        """Takes a sentence and switches it to crfsuite format."""

        words = []
        if self.pos_features:
            tokens = message.get(SPACY_DOCS[TEXT_ATTRIBUTE])
            if not tokens:
                raise ValueError(
                    f"Missing '{SPACY_DOCS[TEXT_ATTRIBUTE]}'. "
                    f"Make sure to add 'SpacyNLP' to your pipeline."
                )
        else:
            tokens = message.get(TOKENS_NAMES[TEXT_ATTRIBUTE])
            # remove CLS token
            tokens = tokens[:-1]

        for i, token in enumerate(tokens):
            pos_tag = self._tag_of_token(token) if self.pos_features else None

            words.append(Word(token.text, pos_tag))

        return words

    @staticmethod
    def _tag_of_token(token):
        if spacy.about.__version__ > "2" and token._.has("tag"):
            return token._.get("tag")
        else:
            return token.tag_

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
