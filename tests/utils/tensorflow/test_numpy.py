import pytest
import tensorflow as tf
import numpy as np
import rasa.utils.tensorflow.numpy
import json
from typing import Optional, Dict, Any


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


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
    assert json.dumps(actual_result, sort_keys=True, cls=NumpyEncoder) == json.dumps(
        expected_result, sort_keys=True, cls=NumpyEncoder
    )
