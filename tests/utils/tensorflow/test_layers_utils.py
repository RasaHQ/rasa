import pytest
import tensorflow as tf
import numpy as np
from typing import List, Optional, Union
import rasa.utils.tensorflow.layers_utils as layers_utils


@pytest.mark.parametrize(
    "batch_size, n, n_max",
    [
        (10, 4, 100),
        (10, 4, 1),
        (
            tf.constant(10, dtype=tf.int32),
            tf.constant(4, dtype=tf.int32),
            tf.constant(100, dtype=tf.int32),
        ),
        (
            tf.constant(10, dtype=tf.int32),
            tf.constant(4, dtype=tf.int32),
            tf.constant(1, dtype=tf.int32),
        ),
    ],
)
def test_random_indices(batch_size: int, n: int, n_max: int):
    indices = layers_utils.random_indices(batch_size, n, n_max)
    assert np.all(tf.shape(indices).numpy() == [batch_size, n])
    assert np.max(indices.numpy()) < n_max
    assert np.max(indices.numpy()) >= 0


def test_random_indices_raises_invalid_argument_error():
    with pytest.raises(tf.errors.InvalidArgumentError):
        layers_utils.random_indices(2, 2, 0)


def test_batch_flatten():
    x = tf.ones([5, 6, 7, 8, 9])
    x_flat = layers_utils.batch_flatten(x)
    assert np.all(tf.shape(x_flat).numpy() == [5 * 6 * 7 * 8, 9])


def test_get_candidate_values():
    x = tf.constant([[0, 1, 2], [3, 4, 5], [6, 7, 8]], dtype=tf.float32)
    candidate_ids = tf.constant([[0, 1], [0, 0], [2, 0]])
    expected_result = [
        [[0, 1, 2], [3, 4, 5]],
        [[0, 1, 2], [0, 1, 2]],
        [[6, 7, 8], [0, 1, 2]],
    ]
    actual_result = layers_utils.get_candidate_values(x, candidate_ids)
    assert np.all(expected_result == actual_result)


@pytest.mark.parametrize(
    "x, y, mask, expected_output",
    [
        ([1, 2, 3], [2, 1, 3], None, 1 / 3),
        ([1, 2, 3], [2, 1, 3], [1.0, 1.0, 0.0], 0.0),
        ([[1, 2], [1, 2]], [[0, 0], [1, 2]], None, 0.5),
        ([[1, 2], [1, 2]], [[0, 0], [1, 2]], [[1.0, 1.0], [1.0, 0.0]], 0.5),
        ([[1, 2], [1, 2]], [[0, 0], [1, 3]], [[1.0, 1.0], [1.0, 1.0]], 0.25),
    ],
)
def test_reduce_mean_equal(
    x: Union[List[List[int]], List[int]],
    y: List[int],
    mask: Optional[List[int]],
    expected_output: float,
):
    assert expected_output == layers_utils.reduce_mean_equal(x, y, mask)
