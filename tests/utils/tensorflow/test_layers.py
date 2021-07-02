import pytest
import numpy as np
import tensorflow as tf
from rasa.utils.tensorflow.layers import RandomlyConnectedDense, DenseForSparse
from typing import Text, Union
from rasa.shared.nlu.constants import (
    TEXT,
    INTENT,
    ACTION_NAME,
    ACTION_TEXT,
    FEATURE_TYPE_SENTENCE,
    FEATURE_TYPE_SEQUENCE,
)
from rasa.utils.tensorflow.constants import LABEL
from rasa.core.constants import DIALOGUE


@pytest.mark.parametrize(
    "inputs, units, expected_output_shape",
    [
        (np.array([[1, 2], [4, 5], [7, 8]]), 4, (3, 4)),
        (np.array([[1, 2], [4, 5], [7, 8]]), 2, (3, 2)),
        (np.array([[1, 2], [4, 5], [7, 8]]), 5, (3, 5)),
        (np.array([[1, 2], [4, 5], [7, 8], [7, 8]]), 5, (4, 5)),
        (np.array([[[1, 2], [4, 5], [7, 8]]]), 4, (1, 3, 4)),
    ],
)
def test_randomly_connected_dense_shape(inputs, units, expected_output_shape):
    layer = RandomlyConnectedDense(units=units)
    y = layer(inputs)
    assert y.shape == expected_output_shape


@pytest.mark.parametrize(
    "inputs, units, expected_num_non_zero_outputs",
    [
        (np.array([[1, 2], [4, 5], [7, 8]]), 4, 12),
        (np.array([[1, 2], [4, 5], [7, 8]]), 2, 6),
        (np.array([[1, 2], [4, 5], [7, 8]]), 5, 15),
        (np.array([[1, 2], [4, 5], [7, 8], [7, 8]]), 5, 20),
        (np.array([[[1, 2], [4, 5], [7, 8]]]), 4, 12),
    ],
)
def test_randomly_connected_dense_output_always_dense(
    inputs: np.array, units: int, expected_num_non_zero_outputs: int
):
    layer = RandomlyConnectedDense(density=0.0, units=units, use_bias=False)
    y = layer(inputs)
    num_non_zero_outputs = tf.math.count_nonzero(y).numpy()
    assert num_non_zero_outputs == expected_num_non_zero_outputs


def test_randomly_connected_dense_all_inputs_connected():
    layer = RandomlyConnectedDense(density=0.0, units=2, use_bias=False)
    # Create a unit vector [1, 0, 0, 0, ...]
    x = np.zeros(10)
    x[0] = 1.0
    # For every standard basis vector
    for _ in range(10):
        x = np.roll(x, 1)
        y = layer(np.expand_dims(x, 0))
        assert tf.reduce_sum(y).numpy() != 0.0


@pytest.mark.parametrize(
    "layer_name, expected_feature_type",
    [
        (f"sparse_to_dense.{TEXT}_{FEATURE_TYPE_SENTENCE}", FEATURE_TYPE_SENTENCE),
        (
            f"sparse_to_dense.{LABEL}_{ACTION_TEXT}_{FEATURE_TYPE_SENTENCE}",
            FEATURE_TYPE_SENTENCE,
        ),
        (
            f"sparse_to_dense.{LABEL}_{ACTION_NAME}_{FEATURE_TYPE_SEQUENCE}",
            FEATURE_TYPE_SEQUENCE,
        ),
        (f"some_name.{DIALOGUE}_{FEATURE_TYPE_SEQUENCE}", FEATURE_TYPE_SEQUENCE),
        (f"some_name.{TEXT}_sentenc", None),
        (f"sparse_to_dense.{TEXT}_squence", None),
        ("some_name", None),
    ],
)
def test_dense_for_sparse_get_feature_type(
    layer_name: Text, expected_feature_type: Union[Text, None]
):
    layer = DenseForSparse(name=layer_name, units=10,)
    assert layer.get_feature_type() == expected_feature_type


@pytest.mark.parametrize(
    "layer_name, expected_attribute",
    [
        (f"sparse_to_dense.{TEXT}_{FEATURE_TYPE_SEQUENCE}", TEXT),
        (f"sparse_to_dense.{INTENT}_{FEATURE_TYPE_SENTENCE}", INTENT),
        (f"other_name.{LABEL}_{FEATURE_TYPE_SENTENCE}", LABEL),
        (f"other_name.{DIALOGUE}_{FEATURE_TYPE_SENTENCE}", DIALOGUE),
        (f"sparse_to_dense.{ACTION_NAME}_{FEATURE_TYPE_SEQUENCE}", ACTION_NAME),
        (f"other_name.{ACTION_TEXT}_{FEATURE_TYPE_SENTENCE}", ACTION_TEXT),
        (
            f"other_name.{LABEL}_{ACTION_NAME}_{FEATURE_TYPE_SENTENCE}",
            f"{LABEL}_{ACTION_NAME}",
        ),
        (
            f"sparse_to_dense.{LABEL}_{ACTION_TEXT}_{FEATURE_TYPE_SEQUENCE}",
            f"{LABEL}_{ACTION_TEXT}",
        ),
        ("some_name", None),
        ("sparse_to_dense", None),
        (f"sparse_to_dense.{TEXT}", None),
        (f"sparse_to_dense.labl_{FEATURE_TYPE_SEQUENCE}", None),
    ],
)
def test_dense_for_sparse_get_attribute(
    layer_name: Text, expected_attribute: Union[Text, None]
):
    layer = DenseForSparse(name=layer_name, units=10,)
    assert layer.get_attribute() == expected_attribute
