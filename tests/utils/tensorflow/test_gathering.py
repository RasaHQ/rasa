import tensorflow as tf
import rasa.utils.tensorflow.gathering
import pytest

sparse_tensor = tf.SparseTensor([[1, 0], [2, 3], [4, 6]], [4, 2, 7], [5, 7])
# [[ 0  0  0  0  0  0  0]
#  [ 4  0  0  0  0  0  0]
#  [ 0  0  0  2  0  0  0]
#  [ 0  0  0  0  0  0  0]
#  [ 0  0  0  0  0  0  7]]


@pytest.mark.parametrize(
    "input_tensor,selection_indices,axis",
    [
        (sparse_tensor, [0, 1, 2], 0),
        (sparse_tensor, [3, 0, 1], 0),
        # (first_sparse_tensor, [4, 4, 4], 0) fails
        (sparse_tensor, [0, 1, 2], 1),
    ],
)
def test_gather_sparse(
    input_tensor: tf.SparseTensor, selection_indices: tf.Tensor, axis: int
):
    """Compares our own sparse gathering method to tf.gather."""
    dense_equivalent_tensor = tf.sparse.to_dense(input_tensor)
    result_tensor = rasa.utils.tensorflow.gathering.gather_sparse(
        input_tensor, selection_indices, axis=axis
    )
    dense_gather_result = tf.gather(
        dense_equivalent_tensor, selection_indices, axis=axis
    )

    tf.assert_equal(tf.sparse.to_dense(result_tensor), dense_gather_result)
