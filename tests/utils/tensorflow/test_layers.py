import pytest
import numpy as np
import tensorflow as tf
from rasa.utils.tensorflow.layers import DenseWithSparseWeights


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
