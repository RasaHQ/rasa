from typing import Optional, Text, Dict, Any, Union, List, Tuple, TYPE_CHECKING
import copy
import numpy as np

import rasa.shared.utils.common
import rasa.shared.utils.io
import rasa.nlu.utils.bilou_utils
from rasa.shared.constants import NEXT_MAJOR_VERSION_FOR_DEPRECATIONS
from rasa.nlu.constants import NUMBER_OF_SUB_TOKENS
import rasa.utils.io as io_utils
from rasa.utils.tensorflow.constants import (
    LOSS_TYPE,
    SIMILARITY_TYPE,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    EPOCHS,
    SOFTMAX,
    MARGIN,
    AUTO,
    INNER,
    COSINE,
    TRANSFORMER_SIZE,
    NUM_TRANSFORMER_LAYERS,
    DENSE_DIMENSION,
)
from rasa.shared.nlu.constants import ACTION_NAME, INTENT, ENTITIES
from rasa.shared.core.constants import ACTIVE_LOOP, SLOTS
from rasa.core.constants import DIALOGUE

if TYPE_CHECKING:
    from rasa.nlu.classifiers.diet_classifier import EntityTagSpec
    from rasa.nlu.tokenizers.tokenizer import Token


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
    list_of_tokens: List[List["Token"]],
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
    old_option: Text,
    new_option: Union[Text, List[Text]],
    config: Dict[Text, Any],
    warn_until_version: Text = NEXT_MAJOR_VERSION_FOR_DEPRECATIONS,
) -> Dict[Text, Any]:
    if old_option not in config:
        return {}

    if isinstance(new_option, str):
        rasa.shared.utils.io.raise_deprecation_warning(
            f"Option '{old_option}' got renamed to '{new_option}'. "
            f"Please update your configuration file.",
            warn_until_version=warn_until_version,
        )
        return {new_option: config[old_option]}

    rasa.shared.utils.io.raise_deprecation_warning(
        f"Option '{old_option}' got renamed to "
        f"a dictionary '{new_option[0]}' with a key '{new_option[1]}'. "
        f"Please update your configuration file.",
        warn_until_version=warn_until_version,
    )
    return {new_option[0]: {new_option[1]: config[old_option]}}


def check_deprecated_options(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """Update the config according to changed config params.

    If old model configuration parameters are present in the provided config, replace
    them with the new parameters and log a warning.

    Args:
        config: model configuration

    Returns: updated model configuration
    """
    # note: call _replace_deprecated_option() here when there are options to deprecate

    return config


def check_core_deprecated_options(config: Dict[Text, Any]) -> Dict[Text, Any]:
    """Update the core config according to changed config params.

    If old model configuration parameters are present in the provided config, replace
    them with the new parameters and log a warning.

    Args:
        config: model configuration

    Returns: updated model configuration
    """
    # note: call _replace_deprecated_option() here when there are options to deprecate
    new_config = {}
    if isinstance(config.get(TRANSFORMER_SIZE), int):
        new_config = override_defaults(
            new_config,
            _replace_deprecated_option(
                TRANSFORMER_SIZE, [TRANSFORMER_SIZE, DIALOGUE], config
            ),
        )

    if isinstance(config.get(NUM_TRANSFORMER_LAYERS), int):
        new_config = override_defaults(
            new_config,
            _replace_deprecated_option(
                NUM_TRANSFORMER_LAYERS, [NUM_TRANSFORMER_LAYERS, DIALOGUE], config
            ),
        )

    if isinstance(config.get(DENSE_DIMENSION), int):
        new_config = override_defaults(
            new_config,
            _replace_deprecated_option(
                DENSE_DIMENSION, [DENSE_DIMENSION, INTENT], config
            ),
        )
        new_config = override_defaults(
            new_config,
            _replace_deprecated_option(
                DENSE_DIMENSION, [DENSE_DIMENSION, ACTION_NAME], config
            ),
        )
        new_config = override_defaults(
            new_config,
            _replace_deprecated_option(
                DENSE_DIMENSION, [DENSE_DIMENSION, ENTITIES], config
            ),
        )
        new_config = override_defaults(
            new_config,
            _replace_deprecated_option(
                DENSE_DIMENSION, [DENSE_DIMENSION, SLOTS], config
            ),
        )
        new_config = override_defaults(
            new_config,
            _replace_deprecated_option(
                DENSE_DIMENSION, [DENSE_DIMENSION, ACTIVE_LOOP], config
            ),
        )

    config.update(new_config)
    return config


def entity_label_to_tags(
    model_predictions: Dict[Text, Any],
    entity_tag_specs: List["EntityTagSpec"],
    bilou_flag: bool = False,
    prediction_index: int = 0,
) -> Tuple[Dict[Text, List[Text]], Dict[Text, List[float]]]:
    """Convert the output predictions for entities to the actual entity tags.

    Args:
        model_predictions: the output predictions using the entity tag indices
        entity_tag_specs: the entity tag specifications
        bilou_flag: if 'True', the BILOU tagging schema was used
        prediction_index: the index in the batch of predictions
            to use for entity extraction

    Returns:
        A map of entity tag type, e.g. entity, role, group, to actual entity tags and
        confidences.
    """
    predicted_tags = {}
    confidence_values = {}

    for tag_spec in entity_tag_specs:
        predictions = model_predictions[f"e_{tag_spec.tag_name}_ids"].numpy()
        confidences = model_predictions[f"e_{tag_spec.tag_name}_scores"].numpy()

        if not np.any(predictions):
            continue

        confidences = [float(c) for c in confidences[prediction_index]]
        tags = [tag_spec.ids_to_tags[p] for p in predictions[prediction_index]]

        if bilou_flag:
            (
                tags,
                confidences,
            ) = rasa.nlu.utils.bilou_utils.ensure_consistent_bilou_tagging(
                tags, confidences
            )

        predicted_tags[tag_spec.tag_name] = tags
        confidence_values[tag_spec.tag_name] = confidences

    return predicted_tags, confidence_values


def override_defaults(
    defaults: Optional[Dict[Text, Any]], custom: Optional[Dict[Text, Any]]
) -> Dict[Text, Any]:
    """Override default config with the given config.

    We cannot use `dict.update` method because configs contain nested dicts.

    Args:
        defaults: default config
        custom: user config containing new parameters

    Returns:
        updated config
    """
    if defaults:
        config = copy.deepcopy(defaults)
    else:
        config = {}

    if custom:
        for key in custom.keys():
            if isinstance(config.get(key), dict):
                config[key].update(custom[key])
            else:
                config[key] = custom[key]

    return config
