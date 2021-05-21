import pytest
import tensorflow as tf
import numpy as np
import rasa.utils.tensorflow.layers_utils as layers_utils


@pytest.mark.parametrize(
    "batch_size, n, n_max", [(10, 4, 100), (10, 4, 0)],
)
def test_random_indices(batch_size, n, n_max):
    indices = layers_utils.random_indices(batch_size, n, n_max)
    assert np.all(tf.shape(indices).numpy() == [batch_size, n])
    assert np.max(indices.numpy()) <= n_max
    assert np.max(indices.numpy()) >= 0


def test_batch_flatten():
    x = tf.ones([5, 6, 7, 8, 9])
    x_flat = layers_utils.batch_flatten(x)
    assert np.all(tf.shape(x_flat).numpy() == [5 * 6 * 7 * 8, 9])


@pytest.mark.parametrize(
    "shape, padding, expected_result",
    [
        (
            [5, 7],
            0,
            [
                [1, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0],
                [1, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
        ),
        (
            [5, 7],
            42,
            [
                [1, 1, 42, 42, 42, 42, 42],
                [1, 1, 42, 42, 42, 42, 42],
                [1, 1, 42, 42, 42, 42, 42],
                [42, 42, 42, 42, 42, 42, 42],
                [42, 42, 42, 42, 42, 42, 42],
            ],
        ),
    ],
)
def test_pad_right(shape, padding, expected_result):
    x = tf.ones([3, 2])
    x_padded = layers_utils.pad_right(x, shape, value=padding)
    assert np.all(tf.shape(x_padded).numpy() == shape)
    assert np.all(x_padded.numpy() == expected_result)


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
    "x, y, expected_output",
    [([1, 2, 3], [2, 1, 3], 1 / 3), ([[1, 2], [1, 2]], [[0, 0], [1, 2]], 0.5)],
)
def test_reduce_mean_equal(x, y, expected_output):
    assert expected_output == layers_utils.reduce_mean_equal(x, y)
