# How many labels are at max put into the output
# ranking, everything else will be cut off
LABEL_RANKING_LENGTH = 10

import scipy.sparse


# TODO should be removed in next PR
def convert_sparse_back(sparse_features: scipy.sparse.csr_matrix):
    import numpy as np

    if sparse_features is not None:
        return np.sum(sparse_features.toarray(), axis=0)
    return None


# TODO should be removed in next PR
def convert_dense_back(dense_features: scipy.sparse.csr_matrix):
    import numpy as np

    if dense_features is not None:
        return np.sum(dense_features, axis=0)
    return None
