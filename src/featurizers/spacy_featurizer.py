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
            vec = np.zeros(self.ndim)
            for token in doc:
                vec += token.vector
            X[idx, :] = vec / len(doc)
        return X
