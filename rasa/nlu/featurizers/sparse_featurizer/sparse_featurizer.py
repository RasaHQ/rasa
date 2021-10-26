from abc import ABC
import scipy.sparse
from rasa.nlu.featurizers.featurizer import Featurizer


class SparseFeaturizer(Featurizer[scipy.sparse.spmatrix], ABC):
    """Base class for all sparse featurizers."""

    pass
