from typing import Any, Dict, List

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
    RANKING_LENGTH,
    RENORMALIZE_CONFIDENCES,
    SIMILARITY_TYPE,
    LOSS_TYPE,
    COSINE,
    SOFTMAX,
    INNER,
    CROSS_ENTROPY,
    MARGIN,
    AUTO,
    TOLERANCE,
    CHECKPOINT_MODEL,
    EVAL_NUM_EPOCHS,
    EVAL_NUM_EXAMPLES,
    EPOCHS,
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
    (
        "input_values, ranking_length, renormalize, possible_output_values, "
        " resulting_ranking_length"
    ),
    [
        # keep the top 2
        ([0.1, 0.4, 0.01], 2, False, [[0.1, 0.4, 0.0]], 2),
        # normalize top 2
        ([0.1, 0.4, 0.01], 2, True, [[0.2, 0.8, 0.0]], 2),
        # 2 possible values that could be excluded
        ([0.1, 0.4, 0.1], 2, True, [[0.0, 0.8, 0.2], [0.2, 0.8, 0.0]], 2),
        # ranking_length > num_confidences => ranking_length := num_confidences
        ([0.1, 0.3, 0.2], 5, False, [[0.1, 0.3, 0.2]], 3),
        # ranking_length > num_confidences  => ranking_length := num_confidences
        ([0.1, 0.3, 0.1], 5, True, [[0.1, 0.3, 0.1]], 3),
        # ranking_length == 0  => ranking_length := num_confidences
        ([0.1, 0.3, 0.1], 0, True, [[0.1, 0.3, 0.1]], 3),
    ],
)
def test_rank_and_mask(
    input_values: List[float],
    ranking_length: int,
    possible_output_values: List[List[float]],
    renormalize: bool,
    resulting_ranking_length: int,
):
    confidences = np.array(input_values)
    indices, modified_confidences = train_utils.rank_and_mask(
        confidences=confidences, ranking_length=ranking_length, renormalize=renormalize
    )
    assert any(
        np.allclose(modified_confidences, np.array(possible_output))
        for possible_output in possible_output_values
    )
    assert np.allclose(
        sorted(input_values, reverse=True)[:resulting_ranking_length],
        confidences[indices],
    )


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
    split_entities_config: Any, expected_initialized_config: Dict[(str, bool)]
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
        ({MODEL_CONFIDENCE: SOFTMAX, LOSS_TYPE: CROSS_ENTROPY}, False),
        ({MODEL_CONFIDENCE: INNER, LOSS_TYPE: MARGIN}, True),
        ({MODEL_CONFIDENCE: INNER, LOSS_TYPE: CROSS_ENTROPY}, True),
        ({MODEL_CONFIDENCE: COSINE, LOSS_TYPE: MARGIN}, True),
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
    ],
)
def test_confidence_similarity_settings(
    component_config: Dict[Text, Any], raises_exception: bool
):
    component_config[LOSS_TYPE] = CROSS_ENTROPY
    if raises_exception:
        with pytest.raises(InvalidConfigException):
            train_utils._check_confidence_setting(component_config)
    else:
        train_utils._check_confidence_setting(component_config)


@pytest.mark.parametrize(
    "component_config, raises_exception",
    [
        (
            {
                MODEL_CONFIDENCE: SOFTMAX,
                SIMILARITY_TYPE: INNER,
                RENORMALIZE_CONFIDENCES: True,
                RANKING_LENGTH: 10,
            },
            False,
        ),
        (
            {
                MODEL_CONFIDENCE: SOFTMAX,
                SIMILARITY_TYPE: INNER,
                RENORMALIZE_CONFIDENCES: False,
                RANKING_LENGTH: 10,
            },
            False,
        ),
        (
            {
                MODEL_CONFIDENCE: AUTO,
                SIMILARITY_TYPE: INNER,
                RENORMALIZE_CONFIDENCES: True,
                RANKING_LENGTH: 10,
            },
            True,
        ),
        (
            {
                MODEL_CONFIDENCE: AUTO,
                SIMILARITY_TYPE: INNER,
                RENORMALIZE_CONFIDENCES: False,
                RANKING_LENGTH: 10,
            },
            False,
        ),
    ],
)
def test_confidence_renormalization_settings(
    component_config: Dict[Text, Any], raises_exception: bool
):
    component_config[LOSS_TYPE] = CROSS_ENTROPY
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
    ],
)
def test_update_confidence_type(
    component_config: Dict[Text, Text], model_confidence: Text
):
    component_config = train_utils.update_confidence_type(component_config)
    assert component_config[MODEL_CONFIDENCE] == model_confidence


@pytest.mark.parametrize(
    "component_config, raises_exception",
    [
        ({TOLERANCE: 0.5}, False),
        ({TOLERANCE: 0.0}, False),
        ({TOLERANCE: 1.0}, False),
        ({TOLERANCE: -1.0}, True),
        ({TOLERANCE: 2.0}, True),
        ({}, False),
    ],
)
def test_tolerance_setting(component_config: Dict[Text, float], raises_exception: bool):
    if raises_exception:
        with pytest.raises(InvalidConfigException):
            train_utils._check_tolerance_setting(component_config)
    else:
        train_utils._check_tolerance_setting(component_config)


@pytest.mark.parametrize(
    "component_config",
    [
        (
            {
                CHECKPOINT_MODEL: True,
                EVAL_NUM_EPOCHS: -2,
                EVAL_NUM_EXAMPLES: 10,
                EPOCHS: 5,
            }
        ),
        (
            {
                CHECKPOINT_MODEL: True,
                EVAL_NUM_EPOCHS: 0,
                EVAL_NUM_EXAMPLES: 10,
                EPOCHS: 5,
            }
        ),
    ],
)
def test_warning_incorrect_eval_num_epochs(component_config: Dict[Text, Text]):
    with pytest.warns(UserWarning) as record:
        train_utils._check_evaluation_setting(component_config)
        assert len(record) == 1
        assert (
            f"'{EVAL_NUM_EPOCHS}' is not -1 or greater than 0. Training will fail"
            in record[0].message.args[0]
        )


@pytest.mark.parametrize(
    "component_config",
    [
        ({CHECKPOINT_MODEL: True, EVAL_NUM_EPOCHS: 10, EPOCHS: 5}),
        ({CHECKPOINT_MODEL: False, EVAL_NUM_EPOCHS: 10, EPOCHS: 5}),
    ],
)
def test_warning_eval_num_epochs_greater_than_epochs(
    component_config: Dict[Text, Text]
):
    warning = (
        f"'{EVAL_NUM_EPOCHS}={component_config[EVAL_NUM_EPOCHS]}' is "
        f"greater than '{EPOCHS}={component_config[EPOCHS]}'."
        f" No evaluation will occur."
    )
    with pytest.warns(UserWarning) as record:
        train_utils._check_evaluation_setting(component_config)
        assert len(record) == 1
        if component_config[CHECKPOINT_MODEL]:
            warning = (
                f"You have opted to save the best model, but {warning} "
                "No checkpoint model will be saved."
            )
        assert warning in record[0].message.args[0]


@pytest.mark.parametrize(
    "component_config",
    [
        ({CHECKPOINT_MODEL: True, EVAL_NUM_EPOCHS: 1, EVAL_NUM_EXAMPLES: 0, EPOCHS: 5}),
        (
            {
                CHECKPOINT_MODEL: True,
                EVAL_NUM_EPOCHS: 1,
                EVAL_NUM_EXAMPLES: -1,
                EPOCHS: 5,
            }
        ),
    ],
)
def test_warning_incorrect_eval_num_examples(component_config: Dict[Text, Text]):
    with pytest.warns(UserWarning) as record:
        train_utils._check_evaluation_setting(component_config)
        assert len(record) == 1
        assert (
            f"'{EVAL_NUM_EXAMPLES}' is not greater than 0. No checkpoint model "
            f"will be saved"
        ) in record[0].message.args[0]


@pytest.mark.parametrize(
    "component_config",
    [
        (
            {
                CHECKPOINT_MODEL: False,
                EVAL_NUM_EPOCHS: 0,
                EVAL_NUM_EXAMPLES: 0,
                EPOCHS: 5,
            }
        ),
        (
            {
                CHECKPOINT_MODEL: True,
                EVAL_NUM_EPOCHS: 1,
                EVAL_NUM_EXAMPLES: 10,
                EPOCHS: 5,
            }
        ),
    ],
)
def test_no_warning_correct_checkpoint_setting(component_config: Dict[Text, Text]):
    with pytest.warns(None) as record:
        train_utils._check_evaluation_setting(component_config)
        assert len(record) == 0
