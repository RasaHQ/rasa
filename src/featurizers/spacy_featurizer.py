import numpy as np


class SpacyFeaturizer(object):
    def __init__(self):
        self.ndim = 300

    def create_bow_vecs(self, sentences, nlp=None):
        X = np.zeros((len(sentences), self.ndim))
        for idx, sentence in enumerate(sentences):
            doc = nlp(sentence)
            vec = np.zeros(self.ndim)
            for token in doc:
                vec += token.vector
            X[idx, :] = vec / len(doc)
        return X
