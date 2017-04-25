from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import

import typing
from typing import Any
from typing import Dict
from typing import List
from typing import Text

from rasa_nlu.components import Component
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.training_data import TrainingData

if typing.TYPE_CHECKING:
    import mitie
    import numpy as np


class MitieFeaturizer(Featurizer, Component):
    name = "intent_featurizer_mitie"

    context_provides = {
        "train": ["intent_features"],
        "process": ["intent_features"],
    }

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["mitie", "numpy"]

    def ndim(self, feature_extractor):
        # type: (mitie.total_word_feature_extractor) -> int

        return feature_extractor.num_dimensions

    def train(self, training_data, mitie_feature_extractor):
        # type: (TrainingData, mitie.total_word_feature_extractor) -> Dict[Text, Any]

        sentences = [e["text"] for e in training_data.intent_examples]
        features = self.features_for_sentences(sentences, mitie_feature_extractor)
        return {
            "intent_features": features
        }

    def process(self, tokens, mitie_feature_extractor):
        # type: (List[Text], mitie.total_word_feature_extractor) -> Dict[Text, Any]

        features = self.features_for_tokens(tokens, mitie_feature_extractor)
        return {
            "intent_features": features
        }

    def features_for_tokens(self, tokens, feature_extractor):
        # type: (List[Text], mitie.total_word_feature_extractor) -> np.ndarray
        import numpy as np

        vec = np.zeros(self.ndim(feature_extractor))
        for token in tokens:
            vec += feature_extractor.get_feature_vector(token)
        if tokens:
            return vec / len(tokens)
        else:
            return vec

    def features_for_sentences(self, sentences, feature_extractor):
        # type: (List[Text], mitie.total_word_feature_extractor) -> np.ndarray
        import mitie
        import numpy as np

        X = np.zeros((len(sentences), self.ndim(feature_extractor)))

        for idx, sentence in enumerate(sentences):
            tokens = mitie.tokenize(sentence)
            X[idx, :] = self.features_for_tokens(tokens, feature_extractor)
        return X
