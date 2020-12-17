import numpy as np
from typing import Text, Optional, Dict, Any

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.constants import FEATURIZER_CLASS_ALIAS
from rasa.nlu.components import Component
from rasa.utils.tensorflow.constants import MEAN_POOLING, MAX_POOLING
from rasa.shared.nlu.training_data.training_data import TrainingDataChunk, TrainingData


class Featurizer(Component):
    """Abstract featurizer component."""

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:
        """Initializes the featurizer.

        Args:
            component_config: The component configuration.
        """
        if not component_config:
            component_config = {}

        # makes sure the alias name is set
        component_config.setdefault(FEATURIZER_CLASS_ALIAS, self.name)

        super().__init__(component_config)

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train this component."""
        self.prepare_partial_training(training_data, config, **kwargs)
        training_data_chunk = TrainingDataChunk(
            training_examples=training_data.training_examples,
            responses=training_data.responses,
        )
        self.train_chunk(training_data_chunk, config, **kwargs)

    def train_chunk(
        self,
        training_data_chunk: TrainingDataChunk,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train this component on the given chunk."""
        pass


class DenseFeaturizer(Featurizer):
    """Abstract dense featurizer component."""

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
    """Abstract sparse featurizer component."""

    pass
