from mitie import *
import numpy as np

from rasa_nlu.featurizers import Featurizer


class MITIEFeaturizer(Featurizer):
    @staticmethod
    def load(path):
        if path:
            extractor = total_word_feature_extractor(path)
            ndim = extractor.num_dimensions
            return MITIEFeaturizer(extractor, ndim)
        else:
            return None

    def __init__(self, extractor, ndim):
        self.feature_extractor = extractor
        self.ndim = ndim

    def features_for_tokens(self, tokens):
        vec = np.zeros(self.ndim)
        for token in tokens:
            vec += self.feature_extractor.get_feature_vector(token)
        if tokens:
            return vec / len(tokens)
        else:
            return vec

    def features_for_sentences(self, sentences):
        X = np.zeros((len(sentences), self.ndim))

        for idx, sentence in enumerate(sentences):
            tokens = tokenize(sentence)
            X[idx, :] = self.features_for_tokens(tokens)
        return X
