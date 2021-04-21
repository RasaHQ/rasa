import pytest
import numpy as np
import tensorflow as tf
from rasa.utils.tensorflow.layers import RandomlyConnectedDense


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
