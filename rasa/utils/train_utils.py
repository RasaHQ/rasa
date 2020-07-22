import numpy as np
import logging
from typing import Optional, Text, Dict, Any, Union, List, Tuple

from rasa.core.constants import DIALOGUE
from rasa.nlu.constants import TEXT, NUMBER_OF_SUB_TOKENS
from rasa.nlu.tokenizers.tokenizer import Token
import rasa.utils.io as io_utils
from rasa.utils.tensorflow.constants import (
    LABEL,
    HIDDEN_LAYERS_SIZES,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    DENSE_DIMENSION,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    REGULARIZATION_CONSTANT,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    DROP_RATE_DIALOGUE,
    DROP_RATE_LABEL,
    NEGATIVE_MARGIN_SCALE,
    DROP_RATE,
    EPOCHS,
    SOFTMAX,
    MARGIN,
    AUTO,
    INNER,
    COSINE,
)


logger = logging.getLogger(__name__)


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
    """
    If SIMILARITY_TYPE is set to 'auto', update the SIMILARITY_TYPE depending
    on the LOSS_TYPE.
    Args:
        config: model configuration

    Returns: updated model configuration
    """
    if config.get(SIMILARITY_TYPE) == AUTO:
        if config[LOSS_TYPE] == SOFTMAX:
            config[SIMILARITY_TYPE] = INNER
        elif config[LOSS_TYPE] == MARGIN:
            config[SIMILARITY_TYPE] = COSINE

    return config


def align_token_features(
    list_of_tokens: List[List[Token]],
    in_token_features: np.ndarray,
    shape: Optional[Tuple] = None,
) -> np.ndarray:
    """Align token features to match tokens.

    ConveRTTokenizer, LanguageModelTokenizers might split up tokens into sub-tokens.
    We need to take the mean of the sub-token vectors and take that as token vector.

    Args:
        list_of_tokens: tokens for examples
        in_token_features: token features from ConveRT
        shape: shape of feature matrix

    Returns:
        Token features.
    """
    if shape is None:
        shape = in_token_features.shape
    out_token_features = np.zeros(shape)

    for example_idx, example_tokens in enumerate(list_of_tokens):
        offset = 0
        for token_idx, token in enumerate(example_tokens):
            number_sub_words = token.get(NUMBER_OF_SUB_TOKENS, 1)

            if number_sub_words > 1:
                token_start_idx = token_idx + offset
                token_end_idx = token_idx + offset + number_sub_words

                mean_vec = np.mean(
                    in_token_features[example_idx][token_start_idx:token_end_idx],
                    axis=0,
                )

                offset += number_sub_words - 1

                out_token_features[example_idx][token_idx] = mean_vec
            else:
                out_token_features[example_idx][token_idx] = in_token_features[
                    example_idx
                ][token_idx + offset]

    return out_token_features


def update_evaluation_parameters(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """
    If EVAL_NUM_EPOCHS is set to -1, evaluate at the end of the training.

    Args:
        config: model configuration

    Returns: updated model configuration
    """

    if config[EVAL_NUM_EPOCHS] == -1:
        config[EVAL_NUM_EPOCHS] = config[EPOCHS]
    elif config[EVAL_NUM_EPOCHS] < 1:
        raise ValueError(
            f"'{EVAL_NUM_EXAMPLES}' is set to "
            f"'{config[EVAL_NUM_EPOCHS]}'. "
            f"Only values > 1 are allowed for this configuration value."
        )

    return config


def load_tf_hub_model(model_url: Text) -> Any:
    """Load model from cache if possible, otherwise from TFHub"""

    import tensorflow_hub as tfhub

    # needed to load the ConveRT model
    # noinspection PyUnresolvedReferences
    import tensorflow_text
    import os

    # required to take care of cases when other files are already
    # stored in the default TFHUB_CACHE_DIR
    try:
        return tfhub.load(model_url)
    except OSError:
        directory = io_utils.create_temporary_directory()
        os.environ["TFHUB_CACHE_DIR"] = directory
        return tfhub.load(model_url)


def _replace_deprecated_option(
    old_option: Text, new_option: Union[Text, List[Text]], config: Dict[Text, Any]
) -> Dict[Text, Any]:
    if old_option in config:
        if isinstance(new_option, str):
            logger.warning(
                f"Option '{old_option}' got renamed to '{new_option}'. "
                f"Please update your configuration file."
            )
            config[new_option] = config[old_option]
        else:
            logger.warning(
                f"Option '{old_option}' got renamed to "
                f"a dictionary '{new_option[0]}' with a key '{new_option[1]}'. "
                f"Please update your configuration file."
            )
            option_dict = config.get(new_option[0], {})
            option_dict[new_option[1]] = config[old_option]
            config[new_option[0]] = option_dict

    return config


def check_deprecated_options(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """
    If old model configuration parameters are present in the provided config, replace
    them with the new parameters and log a warning.
    Args:
        config: model configuration

    Returns: updated model configuration
    """

    config = _replace_deprecated_option(
        "hidden_layers_sizes_pre_dial", [HIDDEN_LAYERS_SIZES, DIALOGUE], config
    )
    config = _replace_deprecated_option(
        "hidden_layers_sizes_bot", [HIDDEN_LAYERS_SIZES, LABEL], config
    )
    config = _replace_deprecated_option("droprate", DROP_RATE, config)
    config = _replace_deprecated_option("droprate_a", DROP_RATE_DIALOGUE, config)
    config = _replace_deprecated_option("droprate_b", DROP_RATE_LABEL, config)
    config = _replace_deprecated_option(
        "hidden_layers_sizes_a", [HIDDEN_LAYERS_SIZES, TEXT], config
    )
    config = _replace_deprecated_option(
        "hidden_layers_sizes_b", [HIDDEN_LAYERS_SIZES, LABEL], config
    )
    config = _replace_deprecated_option(
        "num_transformer_layers", NUM_TRANSFORMER_LAYERS, config
    )
    config = _replace_deprecated_option("num_heads", NUM_HEADS, config)
    config = _replace_deprecated_option("dense_dim", DENSE_DIMENSION, config)
    config = _replace_deprecated_option("embed_dim", EMBEDDING_DIMENSION, config)
    config = _replace_deprecated_option("num_neg", NUM_NEG, config)
    config = _replace_deprecated_option("mu_pos", MAX_POS_SIM, config)
    config = _replace_deprecated_option("mu_neg", MAX_NEG_SIM, config)
    config = _replace_deprecated_option("use_max_sim_neg", USE_MAX_NEG_SIM, config)
    config = _replace_deprecated_option("C2", REGULARIZATION_CONSTANT, config)
    config = _replace_deprecated_option("C_emb", NEGATIVE_MARGIN_SCALE, config)
    config = _replace_deprecated_option(
        "evaluate_every_num_epochs", EVAL_NUM_EPOCHS, config
    )
    config = _replace_deprecated_option(
        "evaluate_on_num_examples", EVAL_NUM_EXAMPLES, config
    )

    return config
