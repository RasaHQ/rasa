import numpy as np
import scipy.sparse
from typing import Any, Text, List, Union, Optional
from rasa.nlu.training_data import Message
from rasa.nlu.components import Component
from rasa.nlu.constants import (
    MESSAGE_VECTOR_SPARSE_FEATURE_NAMES,
    MESSAGE_VECTOR_DENSE_FEATURE_NAMES,
    MESSAGE_TEXT_ATTRIBUTE,
)


def sequence_to_sentence_embedding(
    features: Union[List[List[float]], scipy.sparse.spmatrix], method: Text = "mean"
) -> Optional[np.ndarray]:
    if features is None:
        return None

    if isinstance(features, scipy.sparse.spmatrix):
        features = features.toarray()

    if method == "mean" or method == "avg":
        return np.mean(features, axis=0)
    if method == "sum":
        return np.sum(features, axis=0)


class Featurizer(Component):
    @staticmethod
    def _combine_with_existing_dense_features(
        message: Message,
        additional_features: Any,
        feature_name: Text = MESSAGE_VECTOR_DENSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE],
    ) -> Any:
        if message.get(feature_name) is not None:
            return np.concatenate(
                (message.get(feature_name), additional_features), axis=-1
            )
        else:
            return additional_features

    @staticmethod
    def _combine_with_existing_sparse_features(
        message: Message,
        additional_features: Any,
        feature_name: Text = MESSAGE_VECTOR_SPARSE_FEATURE_NAMES[
            MESSAGE_TEXT_ATTRIBUTE
        ],
    ) -> Any:
        if message.get(feature_name) is not None:
            from scipy.sparse import hstack

            return hstack([message.get(feature_name), additional_features])
        else:
            return additional_features
