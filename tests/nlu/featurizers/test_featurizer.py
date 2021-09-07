from rasa.shared.exceptions import InvalidConfigException
from rasa.nlu.featurizers.featurizer import Featurizer2
from rasa.nlu.constants import FEATURIZER_CLASS_ALIAS
from typing import List, Dict, Any, Text

import numpy as np
import pytest

from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer2


@pytest.mark.parametrize(
    "pooling, features, only_non_zero_vectors, expected",
    [
        (
            "mean",
            np.array([[0.5, 3, 0.4, 0.1], [0, 0, 0, 0], [0.5, 3, 0.4, 0.1]]),
            True,
            np.array([[0.5, 3, 0.4, 0.1]]),
        ),
        (
            "max",
            np.array([[1.0, 3.0, 0.0, 2.0], [4.0, 3.0, 1.0, 0.0]]),
            True,
            np.array([[4.0, 3.0, 1.0, 2.0]]),
        ),
        (
            "max",
            np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
            True,
            np.array([[0.0, 0.0, 0.0, 0.0]]),
        ),
        ("max", np.array([[-1.0], [0.0]]), False, np.array([[0.0]]),),
        ("max", np.array([[-1.0], [0.0]]), True, np.array([[-1.0]]),),
    ],
)
def test_calculate_cls_vector(pooling, features, only_non_zero_vectors, expected):
    actual = DenseFeaturizer2.aggregate_sequence_features(
        features, pooling_operation=pooling, only_non_zero_vectors=only_non_zero_vectors
    )
    assert np.all(actual == expected)


@pytest.mark.parametrize(
    "featurizer_configs,passes",
    [
        (
            [
                {FEATURIZER_CLASS_ALIAS: "name-1", "same": "other-params"},
                {FEATURIZER_CLASS_ALIAS: "name-2", "same": "other-params"},
            ],
            True,
        ),
        ([{}, {}], True),
        (
            [
                {FEATURIZER_CLASS_ALIAS: "same-name", "something": "else"},
                {FEATURIZER_CLASS_ALIAS: "same-name"},
            ],
            False,
        ),
    ],
)
def test_validate_configs_compatible(
    featurizer_configs: List[Dict[Text, Any]], passes: bool
):
    if passes:
        Featurizer2.validate_configs_compatible(featurizer_configs)
    else:
        with pytest.raises(InvalidConfigException):
            Featurizer2.validate_configs_compatible(featurizer_configs)
