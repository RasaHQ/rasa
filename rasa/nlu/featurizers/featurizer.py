import numpy as np
import scipy.sparse
from typing import Any, Text, Union, Optional
from rasa.nlu.training_data import Message
from rasa.nlu.components import Component
from rasa.nlu.constants import SPARSE_FEATURE_NAMES, DENSE_FEATURE_NAMES, TEXT_ATTRIBUTE


def sequence_to_sentence_features(
    features: Union[np.ndarray, scipy.sparse.spmatrix]
) -> Optional[Union[np.ndarray, scipy.sparse.spmatrix]]:
    """Extract the CLS token vector as sentence features.

    Features is a sequence. The last token is the CLS token. The feature vector of
    this token contains the sentence features."""

    if features is None:
        return None

    if isinstance(features, scipy.sparse.spmatrix):
        return scipy.sparse.coo_matrix(features.tocsr()[-1])

    return np.expand_dims(features[-1], axis=0)


class Featurizer(Component):
    @staticmethod
    def _combine_with_existing_dense_features(
        message: Message,
        additional_features: Any,
        feature_name: Text = DENSE_FEATURE_NAMES[TEXT_ATTRIBUTE],
    ) -> Any:
        if message.get(feature_name) is not None:

            if len(message.get(feature_name)) != len(additional_features):
                raise ValueError(
                    f"Cannot concatenate dense features as sequence dimension does not "
                    f"match: {len(message.get(feature_name))} != "
                    f"{len(additional_features)}. "
                    f"Make sure to set 'return_sequence' to the same value for all your "
                    f"featurizers."
                )

            return np.concatenate(
                (message.get(feature_name), additional_features), axis=-1
            )
        else:
            return additional_features

    @staticmethod
    def _combine_with_existing_sparse_features(
        message: Message,
        additional_features: Any,
        feature_name: Text = SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE],
    ) -> Any:
        if message.get(feature_name) is not None:
            from scipy.sparse import hstack

            if message.get(feature_name).shape[0] != additional_features.shape[0]:
                raise ValueError(
                    f"Cannot concatenate sparse features as sequence dimension does not "
                    f"match: {message.get(feature_name).shape[0]} != "
                    f"{additional_features.shape[0]}. "
                    f"Make sure to set 'return_sequence' to the same value for all your "
                    f"featurizers."
                )
            return hstack([message.get(feature_name), additional_features])
        else:
            return additional_features
