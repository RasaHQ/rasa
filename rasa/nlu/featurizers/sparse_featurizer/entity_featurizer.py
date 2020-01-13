import logging
from collections import defaultdict

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


class CRFToken(typing.NamedTuple):
    text: Text
    pos_tag: Text
    pattern: Dict[Text, Any]


class EntityFeaturizer(Featurizer):

    provides = [SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE]]

    requires = [TOKENS_NAMES[TEXT_ATTRIBUTE]]

    defaults = {
        # crf_features is [before, word, after] array with before, word,
        # after holding keys about which
        # features to use for each word, for example, 'title' in
        # array before will have the feature
        # "is the preceding word in title case?"
        # POS features require spaCy to be installed
        "features": [
            ["low", "title", "upper"],
            [
                "bias",
                "low",
                "prefix5",
                "prefix2",
                "suffix5",
                "suffix3",
                "suffix2",
                "upper",
                "title",
                "digit",
                "pattern",
            ],
            ["low", "title", "upper"],
        ]
    }

    function_dict = {
        "low": lambda crf_token: crf_token.text.lower(),  # pytype: disable=attribute-error
        "title": lambda crf_token: crf_token.text.istitle(),  # pytype: disable=attribute-error
        "prefix5": lambda crf_token: crf_token.text[:5],
        "prefix2": lambda crf_token: crf_token.text[:2],
        "suffix5": lambda crf_token: crf_token.text[-5:],
        "suffix3": lambda crf_token: crf_token.text[-3:],
        "suffix2": lambda crf_token: crf_token.text[-2:],
        "suffix1": lambda crf_token: crf_token.text[-1:],
        "pos": lambda crf_token: crf_token.pos_tag,
        "pos2": lambda crf_token: crf_token.pos_tag[:2],
        "bias": lambda crf_token: "bias",
        "upper": lambda crf_token: crf_token.text.isupper(),  # pytype: disable=attribute-error
        "digit": lambda crf_token: crf_token.text.isdigit(),  # pytype: disable=attribute-error
        "pattern": lambda crf_token: crf_token.pattern,
    }

    def __init__(
        self,
        component_config: Dict[Text, Any],
        feature_id_dict: Optional[Dict[Text, Dict[Text, int]]] = None,
    ):
        super().__init__(component_config)

        self.feature_id_dict = feature_id_dict
        self._check_pos_features_and_spacy()

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:
        self.feature_id_dict = self._create_feature_id_dict(training_data)

        for example in training_data.training_examples:
            self._text_features_for_entities(example)

    def process(self, message: Message, **kwargs: Any) -> None:
        self._text_features_for_entities(message)

    def _text_features_for_entities(self, message: Message) -> None:
        tokens = self._from_text_to_crf(message)
        features = self._sentence_to_features(tokens)

        num_features = sum(
            [
                len(feature_vals.values())
                for feature_vals in self.feature_id_dict.values()
            ]
        )

        vec = np.zeros([len(tokens), num_features])

        # convert features into one-hot
        for token_idx, token in enumerate(features):
            for k, v in token.items():
                if k in self.feature_id_dict and str(v) in self.feature_id_dict[k]:
                    feature_idx = self.feature_id_dict[k][str(v)]
                    vec[token_idx][feature_idx] = 1

        entity_features = scipy.sparse.coo_matrix(vec)

        # set features
        features = self._combine_with_existing_sparse_features(
            message, entity_features, feature_name=SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE]
        )
        message.set(SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE], features)

    def _create_feature_id_dict(
        self, training_data: TrainingData
    ) -> Dict[Text, Dict[Text, int]]:
        features = []
        for example in training_data.training_examples:
            tokens = self._from_text_to_crf(example)
            features.append(self._sentence_to_features(tokens))

        # build vocab of features
        vocab_x = defaultdict(set)
        for sent_features in features:
            for token_features in sent_features:
                for key, val in token_features.items():
                    vocab_x[key].add(val)

        feature_id_dict = {}
        offset = 0
        for key, val in vocab_x.items():
            feature_id_dict[key] = {
                str(feature_val): idx
                for idx, feature_val in enumerate(sorted(val), offset)
            }
            offset += len(val)

        return feature_id_dict

    def _sentence_to_features(self, sentence: List[CRFToken]) -> List[Dict[Text, Any]]:
        """Convert a word into discrete features in self.crf_features,
        including word before and word after."""

        configured_features = self.component_config["features"]
        sentence_features = []

        for word_idx in range(len(sentence)):
            # word before(-1), current word(0), next word(+1)
            feature_span = len(configured_features)
            half_span = feature_span // 2
            feature_range = range(-half_span, half_span + 1)
            prefixes = [str(i) for i in feature_range]
            word_features = {}
            for f_i in feature_range:
                if word_idx + f_i >= len(sentence):
                    word_features["EOS"] = True
                    # End Of Sentence
                elif word_idx + f_i < 0:
                    word_features["BOS"] = True
                    # Beginning Of Sentence
                else:
                    word = sentence[word_idx + f_i]
                    f_i_from_zero = f_i + half_span
                    prefix = prefixes[f_i_from_zero]
                    features = configured_features[f_i_from_zero]
                    for feature in features:
                        if feature == "pattern":
                            # add all regexes as a feature
                            regex_patterns = self.function_dict[feature](word)
                            # pytype: disable=attribute-error
                            for p_name, matched in regex_patterns.items():
                                feature_name = prefix + ":" + feature + ":" + p_name
                                word_features[feature_name] = matched
                            # pytype: enable=attribute-error
                        else:
                            # append each feature to a feature vector
                            value = self.function_dict[feature](word)
                            word_features[prefix + ":" + feature] = value
            sentence_features.append(word_features)
        return sentence_features

    def _from_text_to_crf(self, message: Message) -> List[CRFToken]:
        """Takes a sentence and switches it to crfsuite format."""

        crf_format = []
        if self.pos_features:
            tokens = message.get(SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE])
        else:
            tokens = message.get(TOKENS_NAMES[TEXT_ATTRIBUTE])

        for i, token in enumerate(tokens):
            pattern = self.__pattern_of_token(message, i)
            pos_tag = self.__tag_of_token(token) if self.pos_features else None

            crf_format.append(CRFToken(token.text, pos_tag, pattern))

        return crf_format

    @staticmethod
    def __pattern_of_token(message, i):
        if message.get(TOKENS_NAMES[TEXT_ATTRIBUTE]) is not None:
            return message.get(TOKENS_NAMES[TEXT_ATTRIBUTE])[i].get("pattern", {})
        else:
            return {}

    @staticmethod
    def __tag_of_token(token):
        if spacy.about.__version__ > "2" and token._.has("tag"):
            return token._.get("tag")
        else:
            return token.tag_

    def _check_pos_features_and_spacy(self):
        import itertools

        features = self.component_config.get("features", [])
        fts = set(itertools.chain.from_iterable(features))
        self.pos_features = "pos" in fts or "pos2" in fts
        if self.pos_features:
            self._check_spacy()

    @staticmethod
    def _check_spacy():
        if spacy is None:
            raise ImportError(
                "Failed to import `spaCy`. "
                "`spaCy` is required for POS features "
                "See https://spacy.io/usage/ for installation"
                "instructions."
            )

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Optional[Text] = None,
        model_metadata: Optional["Metadata"] = None,
        cached_component: Optional["EntityFeaturizer"] = None,
        **kwargs: Any,
    ) -> "EntityFeaturizer":

        file_name = meta.get("file")

        with open(
            os.path.join(model_dir, file_name + ".feature_id_dict.pkl"), "rb"
        ) as f:
            feature_id_dict = pickle.load(f)

        return EntityFeaturizer(meta, feature_id_dict=feature_id_dict)

    def persist(self, file_name: Text, model_dir: Text) -> Optional[Dict[Text, Any]]:
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        with open(
            os.path.join(model_dir, file_name + ".feature_id_dict.pkl"), "wb"
        ) as f:
            pickle.dump(self.feature_id_dict, f)

        return {"file": file_name}
