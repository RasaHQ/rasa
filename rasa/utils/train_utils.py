import logging
import tensorflow as tf
import numpy as np
from typing import Optional, Text, Dict, Any

from rasa.utils.tensorflow.constants import (
    HIDDEN_LAYERS_SIZES_TEXT,
    HIDDEN_LAYERS_SIZES_LABEL,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    MAX_SEQ_LENGTH,
    DENSE_DIM,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    C2,
    USE_MAX_SIM_NEG,
    MU_NEG,
    MU_POS,
    EMBED_DIM,
    HIDDEN_LAYERS_SIZES_DIALOGUE,
    DROPRATE_DIALOGUE,
    DROPRATE_LABEL,
)


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
    if config.get(SIMILARITY_TYPE) == "auto":
        if config[LOSS_TYPE] == "softmax":
            config[SIMILARITY_TYPE] = "inner"
        elif config[LOSS_TYPE] == "margin":
            config[SIMILARITY_TYPE] = "cosine"

    return config


def _replace_deprecated_option(
    old_option: Text, new_option: Text, config: Dict[Text, Any]
) -> Dict[Text, Any]:
    if old_option in config:
        logger.warning(
            f"Option '{old_option}' got renamed to '{new_option}'. "
            f"Please update your configuration file."
        )
        config[new_option] = config[old_option]

    return config


def check_deprecated_options(config: Dict[Text, Any]) -> Dict[Text, Any]:

    config = _replace_deprecated_option(
        "hidden_layers_sizes_pre_dial", HIDDEN_LAYERS_SIZES_DIALOGUE, config
    )
    config = _replace_deprecated_option(
        "hidden_layers_sizes_bot", HIDDEN_LAYERS_SIZES_LABEL, config
    )
    config = _replace_deprecated_option("droprate_a", DROPRATE_DIALOGUE, config)
    config = _replace_deprecated_option("droprate_b", DROPRATE_LABEL, config)
    config = _replace_deprecated_option(
        "hidden_layers_sizes_a", HIDDEN_LAYERS_SIZES_TEXT, config
    )
    config = _replace_deprecated_option(
        "hidden_layers_sizes_b", HIDDEN_LAYERS_SIZES_LABEL, config
    )
    config = _replace_deprecated_option(
        "num_transformer_layers", NUM_TRANSFORMER_LAYERS, config
    )
    config = _replace_deprecated_option("num_heads", NUM_HEADS, config)
    config = _replace_deprecated_option("max_seq_length", MAX_SEQ_LENGTH, config)
    config = _replace_deprecated_option("dense_dim", DENSE_DIM, config)
    config = _replace_deprecated_option("embed_dim", EMBED_DIM, config)
    config = _replace_deprecated_option("num_neg", NUM_NEG, config)
    config = _replace_deprecated_option("mu_pos", MU_POS, config)
    config = _replace_deprecated_option("mu_neg", MU_NEG, config)
    config = _replace_deprecated_option("use_max_sim_neg", USE_MAX_SIM_NEG, config)
    config = _replace_deprecated_option("C2", C2, config)
    config = _replace_deprecated_option(
        "evaluate_every_num_epochs", EVAL_NUM_EPOCHS, config
    )
    config = _replace_deprecated_option(
        "evaluate_on_num_examples", EVAL_NUM_EXAMPLES, config
    )

    return config
