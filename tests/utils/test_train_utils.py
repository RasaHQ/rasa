from typing import Any, Dict

import numpy as np
import pytest

import rasa.utils.train_utils as train_utils
from rasa.core.policies.ted_policy import TEDPolicy
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
