import logging
import tensorflow as tf
import numpy as np
from typing import Optional, Text, Dict, Any

from rasa.utils.tensorflow.constants import SIMILARITY_TYPE, LOSS_TYPE

logger = logging.getLogger(__name__)


def load_tf_config(config: Dict[Text, Any]) -> Optional[tf.compat.v1.ConfigProto]:
    """Prepare `tf.compat.v1.ConfigProto` for training"""

    if config.get("tf_config") is not None:
        return tf.compat.v1.ConfigProto(**config.pop("tf_config"))
    else:
        return None


def normalize(values: np.ndarray, ranking_length: Optional[int] = 0) -> np.ndarray:
    """Normalizes an array of positive numbers over the top `ranking_length` values.
    Other values will be set to 0.
    """

    new_values = values.copy()  # prevent mutation of the input
    if 0 < ranking_length < len(new_values):
        ranked = sorted(new_values, reverse=True)
        new_values[new_values < ranked[ranking_length - 1]] = 0

    if np.sum(new_values) > 0:
        new_values = new_values / np.sum(new_values)

    return new_values


def update_similarity_type(config: Dict[Text, Any]) -> Dict[Text, Any]:
    if config[SIMILARITY_TYPE] == "auto":
        if config[LOSS_TYPE] == "softmax":
            config[SIMILARITY_TYPE] = "inner"
        elif config[LOSS_TYPE] == "margin":
            config[SIMILARITY_TYPE] = "cosine"

    return config
