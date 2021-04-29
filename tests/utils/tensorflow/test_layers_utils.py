import pytest
import tensorflow as tf
import numpy as np
import rasa.utils.tensorflow.layers_utils as layers_utils


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
