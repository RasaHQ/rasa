import numpy as np
from typing import Text, Optional, Dict, Any

from rasa.architecture_prototype.graph import ComponentPersistor
from rasa.nlu.constants import FEATURIZER_CLASS_ALIAS
from rasa.nlu.components import Component
from rasa.utils.tensorflow.constants import MEAN_POOLING, MAX_POOLING


class Featurizer(Component):
    def __init__(
        self, persistor: Optional[ComponentPersistor] = None, **kwargs: Any,
    ) -> None:
        super().__init__(persistor=persistor, **kwargs)

        # makes sure the alias name is set
        # Necessary for `unique_name` to be defined
        self.component_config.setdefault(FEATURIZER_CLASS_ALIAS, self.unique_name)


class DenseFeaturizer(Featurizer):
    @staticmethod
    def _calculate_sentence_features(
        features: np.ndarray, pooling_operation: Text
    ) -> np.ndarray:
        # take only non zeros feature vectors into account
        non_zero_features = np.array([f for f in features if f.any()])

        # if features are all zero just return a vector with all zeros
        if non_zero_features.size == 0:
            return np.zeros([1, features.shape[-1]])

        if pooling_operation == MEAN_POOLING:
            return np.mean(non_zero_features, axis=0, keepdims=True)

        if pooling_operation == MAX_POOLING:
            return np.max(non_zero_features, axis=0, keepdims=True)

        raise ValueError(
            f"Invalid pooling operation specified. Available operations are "
            f"'{MEAN_POOLING}' or '{MAX_POOLING}', but provided value is "
            f"'{pooling_operation}'."
        )


class SparseFeaturizer(Featurizer):
    pass
