import pytest
import numpy as np
import tensorflow as tf
import rasa.utils.tensorflow.layers
from rasa.utils.tensorflow.layers import DenseWithSparseWeights, LocallyConnectedDense


def test_dense_with_sparse_weights_shape():
    layer = DenseWithSparseWeights(sparsity=0.99, units=2)
    layer.build(input_shape=(3, 1))
    x = np.array([[1], [4], [7]])
    y = layer(x)
    assert y.shape == (3, 2)


def test_dense_with_sparse_weights_output_always_dense():
    layer = DenseWithSparseWeights(sparsity=1.00, units=4, use_bias=False)
    layer.build(input_shape=(3, 2))
    x = np.array([[1, 2], [4, 5], [7, 8]])
    y = layer(x)
    num_nonzero = tf.math.count_nonzero(y).numpy()
    assert num_nonzero == 3 * 4


def test_periodic_padding():
    batch_size = 11
    n = 4
    x = tf.ones((batch_size, n))
    padded_x = rasa.utils.tensorflow.layers.periodic_padding(x, axis=1)
    assert padded_x.shape == (11, 6)


def test_locally_connected_dense_shape_with_one_batch_dimension():
    batch_size = 3
    input_size = 5
    kernel_size = 5

    layer = LocallyConnectedDense(kernel_size=kernel_size)
    x = tf.ones((batch_size, input_size))
    y = layer(x)
    assert y.shape == x.shape


def test_locally_connected_dense_shape_with_two_batch_dimensions():
    batch_size_1 = 2
    batch_size_2 = 3
    input_size = 7
    kernel_size = 3

    layer = LocallyConnectedDense(kernel_size=kernel_size)
    x = tf.ones((batch_size_1, batch_size_2, input_size))
    y = layer(x)
    assert y.shape == x.shape
