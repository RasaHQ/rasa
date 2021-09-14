from abc import ABC
import scipy.sparse
from rasa.nlu.featurizers.featurizer import Featurizer2


class SparseFeaturizer2(Featurizer2[scipy.sparse.spmatrix], ABC):
    """Base class for all sparse featurizers."""

    pass
