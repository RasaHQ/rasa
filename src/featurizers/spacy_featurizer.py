import numpy as np

from rasa_nlu.featurizers import Featurizer


class SpacyFeaturizer(Featurizer):
    def __init__(self, nlp):
        self.nlp = nlp
        self.ndim = 300

    def features_for_doc(self, doc):
        vec = []
        for token in doc:
            if token.has_vector:
                vec.append(token.vector)
        if vec:
            return np.sum(vec, axis=0) / len(vec)
        else:
            return np.zeros(self.ndim)

    def features_for_sentences(self, sentences):
        X = np.zeros((len(sentences), self.ndim))
        for idx, sentence in enumerate(sentences):
            doc = self.nlp(sentence)
            X[idx, :] = self.features_for_doc(doc)
        return X
