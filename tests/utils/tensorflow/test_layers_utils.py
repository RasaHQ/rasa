import pytest
import tensorflow as tf
import numpy as np
import rasa.utils.tensorflow.layers_utils as layers_utils


def test_random_indices():
    indices = layers_utils.random_indices(10, 4, 100)
    assert np.all(tf.shape(indices).numpy() == [10, 4])
    assert np.max(indices.numpy()) <= 100


def test_batch_flatten():
    x = tf.ones([5, 6, 7, 8, 9])
    x_flat = layers_utils.batch_flatten(x)
    assert np.all(tf.shape(x_flat).numpy() == [5 * 6 * 7 * 8, 9])


def test_pad_right():
    x = tf.ones([3, 2])
    x_padded = layers_utils.pad_right(x, [5, 7])
    assert np.all(tf.shape(x_padded).numpy() == [5, 7])
    assert np.all(
        x_padded.numpy()
        == [
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ]
    )
