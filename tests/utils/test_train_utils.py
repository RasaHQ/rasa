from typing import Any, Dict

import numpy as np
import pytest
from typing import Text

import rasa.utils.train_utils as train_utils
from rasa.nlu.constants import NUMBER_OF_SUB_TOKENS
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.shared.nlu.constants import (
    SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
    SPLIT_ENTITIES_BY_COMMA,
)
from rasa.utils.tensorflow.constants import (
    MODEL_CONFIDENCE,
    SIMILARITY_TYPE,
    LOSS_TYPE,
    COSINE,
    SOFTMAX,
    INNER,
    CROSS_ENTROPY,
    MARGIN,
    AUTO,
    LINEAR_NORM,
)
from rasa.shared.exceptions import InvalidConfigException


def test_align_token_features():
    tokens = [
        Token("This", 0, data={NUMBER_OF_SUB_TOKENS: 1}),
        Token("is", 5, data={NUMBER_OF_SUB_TOKENS: 1}),
        Token("a", 8, data={NUMBER_OF_SUB_TOKENS: 1}),
        Token("sentence", 10, data={NUMBER_OF_SUB_TOKENS: 2}),
        Token("embedding", 19, data={NUMBER_OF_SUB_TOKENS: 4}),
    ]

    seq_dim = sum(t.get(NUMBER_OF_SUB_TOKENS) for t in tokens)
    token_features = np.random.rand(1, seq_dim, 64)

    actual_features = train_utils.align_token_features([tokens], token_features)

    assert np.all(actual_features[0][0] == token_features[0][0])
    assert np.all(actual_features[0][1] == token_features[0][1])
    assert np.all(actual_features[0][2] == token_features[0][2])
    # sentence is split into 2 sub-tokens
    assert np.all(actual_features[0][3] == np.mean(token_features[0][3:5], axis=0))
    # embedding is split into 4 sub-tokens
    assert np.all(actual_features[0][4] == np.mean(token_features[0][5:10], axis=0))


@pytest.mark.parametrize(
    "input_values, ranking_length, output_values",
    [
        ([0.2, 0.7, 0.1], 2, [0.2222222, 0.77777778, 0.0]),
        ([0.1, 0.7, 0.1], 5, [0.11111111, 0.77777778, 0.11111111]),
    ],
)
def test_normalize(input_values, ranking_length, output_values):
    normalized_values = train_utils.normalize(np.array(input_values), ranking_length)
    assert np.allclose(normalized_values, np.array(output_values), atol=1e-5)


@pytest.mark.parametrize(
    "split_entities_config, expected_initialized_config",
    [
        (
            SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
            {SPLIT_ENTITIES_BY_COMMA: SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE},
        ),
        (
            {"address": False, "ingredients": True},
            {
                "address": False,
                "ingredients": True,
                SPLIT_ENTITIES_BY_COMMA: SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
            },
        ),
    ],
)
def test_init_split_entities_config(
    split_entities_config: Any, expected_initialized_config: Dict[(str, bool)],
):
    assert (
        train_utils.init_split_entities(
            split_entities_config, SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE
        )
        == expected_initialized_config
    )


@pytest.mark.parametrize(
    "component_config, raises_exception",
    [
        ({MODEL_CONFIDENCE: SOFTMAX, LOSS_TYPE: MARGIN}, True),
        ({MODEL_CONFIDENCE: SOFTMAX, LOSS_TYPE: SOFTMAX}, False),
        ({MODEL_CONFIDENCE: SOFTMAX, LOSS_TYPE: CROSS_ENTROPY}, False),
        ({MODEL_CONFIDENCE: LINEAR_NORM, LOSS_TYPE: MARGIN}, False),
        ({MODEL_CONFIDENCE: LINEAR_NORM, LOSS_TYPE: SOFTMAX}, False),
        ({MODEL_CONFIDENCE: LINEAR_NORM, LOSS_TYPE: CROSS_ENTROPY}, False),
        ({MODEL_CONFIDENCE: INNER, LOSS_TYPE: MARGIN}, True),
        ({MODEL_CONFIDENCE: INNER, LOSS_TYPE: SOFTMAX}, True),
        ({MODEL_CONFIDENCE: INNER, LOSS_TYPE: CROSS_ENTROPY}, True),
        ({MODEL_CONFIDENCE: COSINE, LOSS_TYPE: MARGIN}, True),
        ({MODEL_CONFIDENCE: COSINE, LOSS_TYPE: SOFTMAX}, True),
        ({MODEL_CONFIDENCE: COSINE, LOSS_TYPE: CROSS_ENTROPY}, True),
    ],
)
def test_confidence_loss_settings(
    component_config: Dict[Text, Any], raises_exception: bool
):
    component_config[SIMILARITY_TYPE] = INNER
    if raises_exception:
        with pytest.raises(InvalidConfigException):
            train_utils._check_confidence_setting(component_config)
    else:
        train_utils._check_confidence_setting(component_config)


@pytest.mark.parametrize(
    "component_config, raises_exception",
    [
        ({MODEL_CONFIDENCE: SOFTMAX, SIMILARITY_TYPE: INNER}, False),
        ({MODEL_CONFIDENCE: SOFTMAX, SIMILARITY_TYPE: COSINE}, True),
        ({MODEL_CONFIDENCE: LINEAR_NORM, SIMILARITY_TYPE: INNER}, False),
        ({MODEL_CONFIDENCE: LINEAR_NORM, SIMILARITY_TYPE: COSINE}, False),
    ],
)
def test_confidence_similarity_settings(
    component_config: Dict[Text, Any], raises_exception: bool
):
    component_config[LOSS_TYPE] = SOFTMAX
    if raises_exception:
        with pytest.raises(InvalidConfigException):
            train_utils._check_confidence_setting(component_config)
    else:
        train_utils._check_confidence_setting(component_config)


@pytest.mark.parametrize(
    "component_config, model_confidence",
    [
        ({MODEL_CONFIDENCE: SOFTMAX, LOSS_TYPE: MARGIN}, AUTO),
        ({MODEL_CONFIDENCE: SOFTMAX, LOSS_TYPE: CROSS_ENTROPY}, SOFTMAX),
        ({MODEL_CONFIDENCE: LINEAR_NORM, LOSS_TYPE: CROSS_ENTROPY}, LINEAR_NORM,),
        ({MODEL_CONFIDENCE: LINEAR_NORM, LOSS_TYPE: MARGIN}, AUTO),
    ],
)
def test_update_confidence_type(
    component_config: Dict[Text, Text], model_confidence: Text
):
    component_config = train_utils.update_confidence_type(component_config)
    assert component_config[MODEL_CONFIDENCE] == model_confidence
