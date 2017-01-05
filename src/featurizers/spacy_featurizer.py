import spacy
import numpy as np


class SpacyFeaturizer(object):
    def __init__(self, nlp):
        self.nlp = nlp
        self.ndim = 300

    def create_bow_vecs(self, sentences):
        X = np.zeros((len(sentences), self.ndim))
        for idx, sentence in enumerate(sentences):
            doc = self.nlp(sentence)
            vec = []
            for token in doc:
                if token.has_vector:
                    vec.append(token.vector)
            if vec:
                X[idx, :] = np.sum(vec, axis=0) / len(vec)
        return X
