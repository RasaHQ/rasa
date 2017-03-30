from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from rasa_nlu.components import Component
from rasa_nlu.featurizers import Featurizer
from rasa_nlu.training_data import TrainingData


class MitieFeaturizer(Featurizer, Component):
    name = "intent_featurizer_mitie"

    context_provides = {
        "train": ["intent_features"],
        "process": ["intent_features"],
    }

    def ndim(self, feature_extractor):
        # type: (mitie.total_word_feature_extractor) -> int
        import mitie

        return feature_extractor.num_dimensions

    def train(self, training_data, mitie_feature_extractor):
        # type: (TrainingData, mitie.total_word_feature_extractor) -> dict
        import mitie

        sentences = [e["text"] for e in training_data.intent_examples]
        features = self.features_for_sentences(sentences, mitie_feature_extractor)
        return {
            "intent_features": features
        }

    def process(self, tokens, mitie_feature_extractor):
        # type: ([str], mitie.total_word_feature_extractor) -> dict
        import mitie

        features = self.features_for_tokens(tokens, mitie_feature_extractor)
        return {
            "intent_features": features
        }

    def features_for_tokens(self, tokens, feature_extractor):
        # type: ([str], mitie.total_word_feature_extractor) -> np.ndarray
        import numpy as np
        import mitie

        vec = np.zeros(self.ndim(feature_extractor))
        for token in tokens:
            vec += feature_extractor.get_feature_vector(token)
        if tokens:
            return vec / len(tokens)
        else:
            return vec

    def features_for_sentences(self, sentences, feature_extractor):
        # type: ([str], mitie.total_word_feature_extractor) -> np.ndarray
        import mitie
        import numpy as np

        X = np.zeros((len(sentences), self.ndim(feature_extractor)))

        for idx, sentence in enumerate(sentences):
            tokens = mitie.tokenize(sentence)
            X[idx, :] = self.features_for_tokens(tokens, feature_extractor)
        return X
