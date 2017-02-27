from mitie import *


class MITIEFeaturizer(object):
    def __init__(self, fe_file):
        self.feature_extractor = total_word_feature_extractor(fe_file)
        self.ndim = self.feature_extractor.num_dimensions

    def create_bow_vecs(self, sentences):
        import numpy as np
        X = np.zeros((len(sentences), self.ndim))

        for idx, sent in enumerate(sentences):
            tokens = tokenize(sent)
            vec = np.zeros(self.ndim)
            for token in tokens:
                vec += self.feature_extractor.get_feature_vector(token)
            X[idx, :] = vec / len(tokens)
        return X
