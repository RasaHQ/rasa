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
    MESSAGE_ATTRIBUTES,
    MESSAGE_INTENT_ATTRIBUTE,
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_RESPONSE_ATTRIBUTE,
)


def ndim(spacy_nlp: "Language") -> int:
    """Number of features used to represent a document / sentence."""
    return spacy_nlp.vocab.vectors_length


def features_for_doc(doc: "Doc") -> np.ndarray:
    """Feature vector for a single document / sentence."""
    return doc.vector


class SpacyFeaturizer(Featurizer):

    provides = ["text_features", "intent_features", "response_features"]

    requires = ["spacy_doc", "intent_spacy_doc", "response_spacy_doc"]

    def train(
        self, training_data: TrainingData, config: RasaNLUModelConfig, **kwargs: Any
    ) -> None:

        for example in training_data.intent_examples:
            for attribute in MESSAGE_ATTRIBUTES:
                self._set_spacy_features(example, attribute)

    def get_doc(self, message, attribute):

        attribute = "" if attribute == MESSAGE_TEXT_ATTRIBUTE else attribute + "_"
        return message.get("{0}spacy_doc".format(attribute))

    def process(self, message: Message, **kwargs: Any) -> None:

        self._set_spacy_features(message)

    def _set_spacy_features(self, message, attribute=MESSAGE_TEXT_ATTRIBUTE):
        """Adds the spacy word vectors to the messages features."""

        message_attribute_doc = self.get_doc(message, attribute)
        if message_attribute_doc:
            fs = features_for_doc(message_attribute_doc)
            features = self._combine_with_existing_features(message, fs, attribute)
            message.set("{0}_features".format(attribute), features)
