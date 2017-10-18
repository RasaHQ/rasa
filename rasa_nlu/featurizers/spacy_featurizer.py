from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.featurizers import Featurizer
from rasa_nlu.components import Component
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData


if typing.TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc
    import numpy as np


class SpacyFeaturizer(Featurizer):
    name = "intent_featurizer_spacy"

    provides = ["text_features"]

    requires = ["spacy_doc"]

    def train(self, training_data, config, **kwargs):
        # type: (TrainingData) -> None

        for example in training_data.intent_examples:
            features = self.features_for_doc(example.get("spacy_doc"))
            example.set("text_features", self._combine_with_existing_text_features(example, features))

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None

        features = self.features_for_doc(message.get("spacy_doc"))
        message.set("text_features", self._combine_with_existing_text_features(message, features))

    def features_for_doc(self, doc):
        # type: (Doc) -> np.ndarray

        return doc.vector   # this will return the sentence embedding as calculated by spacy (currently averaged words)
