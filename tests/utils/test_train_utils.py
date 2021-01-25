from typing import Any, Dict

import numpy as np
import pytest
from typing import List

import rasa.utils.train_utils as train_utils
from rasa.nlu.constants import NUMBER_OF_SUB_TOKENS
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.shared.nlu.constants import (
    SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
    SPLIT_ENTITIES_BY_COMMA,
)


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


def test_normalize():
    input_values = [0.7, 0.1, 0.1]
    normalized_values = train_utils.normalize(np.array(input_values))
    assert np.allclose(
        normalized_values, np.array([0.77777778, 0.11111111, 0.11111111]), atol=1e-5
    )


@pytest.mark.parametrize(
    "input_values, ranking_length, output_values",
    [([0.5, 0.8, 0.1], 2, [0.5, 0.8, 0.0]), ([0.5, 0.3, 0.9], 5, [0.5, 0.3, 0.9]),],
)
def test_sort_and_rank(
    input_values: List[float], ranking_length: int, output_values: List[float]
):
    ranked_values = train_utils.sort_and_rank(np.array(input_values), ranking_length)
    assert np.array_equal(ranked_values, output_values)


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
