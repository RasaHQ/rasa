import pytest
import tensorflow as tf
import numpy as np
import rasa.utils.tensorflow.numpy
from typing import Optional, Dict, Any


@pytest.mark.parametrize(
    "value, expected_result",
    [
        ({}, {}),
        ({"a": 1}, {"a": 1}),
        ({"a": tf.zeros((2, 3))}, {"a": np.zeros((2, 3))}),
    ],
)
def test_values_to_numpy(
    value: Optional[Dict[Any, Any]], expected_result: Optional[Dict[Any, Any]]
):
    actual_result = rasa.utils.tensorflow.numpy.values_to_numpy(value)
    actual_result_value_types = [
        type(value) for value in sorted(actual_result.values())
    ]
    expected_result_value_types = [
        type(value) for value in sorted(actual_result.values())
    ]
    assert actual_result_value_types == expected_result_value_types
    np.testing.assert_equal(actual_result, expected_result)
