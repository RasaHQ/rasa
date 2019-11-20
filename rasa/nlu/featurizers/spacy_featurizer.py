import numpy as np
import typing
from typing import Any

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers import Featurizer
from rasa.nlu.training_data import Message, TrainingData

if typing.TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc

from rasa.nlu.constants import (
    RESPONSE_ATTRIBUTE,
    INTENT_ATTRIBUTE,
    TEXT_ATTRIBUTE,
    TOKEN_NAMES,
    ATTRIBUTES,
    SPACY_FEATURE_NAMES,
    FEATURE_NAMES,
    ENTITIES_ATTRIBUTE,
    NER_FEATURES_ATTRIBUTE,
    DENSE_FEATURIZABLE_ATTRIBUTES,
)


def ndim(spacy_nlp: "Language") -> int:
    """Number of features used to represent a document / sentence."""
    return spacy_nlp.vocab.vectors_length


def features_for_doc(doc: "Doc") -> np.ndarray:
    """Feature vector for a single document / sentence."""
    return doc.vector


class SpacyFeaturizer(Featurizer):

    provides = [
        FEATURE_NAMES[attribute] for attribute in DENSE_FEATURIZABLE_ATTRIBUTES
    ] + [NER_FEATURES_ATTRIBUTE]

    requires = [
        SPACY_FEATURE_NAMES[attribute] for attribute in DENSE_FEATURIZABLE_ATTRIBUTES
    ]

    defaults = {"ner_feature_vectors": False}

    def __init__(self, component_config=None, known_patterns=None, lookup_tables=None):

        super().__init__(component_config)

        self.ner_feature_vectors = self.component_config["ner_feature_vectors"]

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        for example in training_data.intent_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self._set_spacy_features(example, attribute)
            self._set_spacy_ner_features(example)

    def get_doc(self, message, attribute):

        return message.get(SPACY_FEATURE_NAMES[attribute])

    def process(self, message: Message, **kwargs: Any) -> None:

        self._set_spacy_features(message)
        self._set_spacy_ner_features(message)

    def _set_spacy_ner_features(self, message: Message):
        """If we want to use spacy as an NER featurizer, set token vectors"""
        doc = message.get(SPACY_FEATURE_NAMES[TEXT_ATTRIBUTE])
        if self.ner_feature_vectors:
            ner_features = np.array([t.vector for t in doc])
        else:
            ner_features = np.array([[] for t in doc])
        combined_features = self._combine_with_existing_features(
            message, ner_features, FEATURE_NAMES[ENTITIES_ATTRIBUTE]
        )
        message.set(FEATURE_NAMES[ENTITIES_ATTRIBUTE], combined_features)

    def _set_spacy_features(self, message, attribute=TEXT_ATTRIBUTE):
        """Adds the spacy word vectors to the messages features."""

        message_attribute_doc = self.get_doc(message, attribute)
        if message_attribute_doc is not None:
            fs = features_for_doc(message_attribute_doc)
            features = self._combine_with_existing_features(
                message, fs, FEATURE_NAMES[attribute]
            )
            message.set(FEATURE_NAMES[attribute], features)
