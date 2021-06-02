import tensorflow as tf
from typing import Union


def gather_sparse(
    input_tensor: tf.SparseTensor, selection_indices: tf.Tensor, axis: int = 0
) -> tf.SparseTensor:
    """Gather indices from sparse tensor without transformation to dense tensor.

    For the code comments, assume input_tensor is:
        indices = [[0, 0], [2, 2], [2, 3], [3, 1]]
        values = [1, 2, 2, 3]
        original_shape = [4, 5]
    further, assume selection_indices is: [3,2]
    and axis is 0

    Note: Entries in selection_indices should be unique and sorted. If not,
    this implementation will fail.

    Args:
        input_tensor: Input sparse tensor to be sliced
        selection_indices: Indices inside the sparse tensor that should be picked
        axis: Axis over which selection_indices should operate
    Returns:
    """
    n_indices = tf.size(selection_indices)  # (2)

    # Get indices for the axis
    selection_axis_indices = input_tensor.indices[:, axis]  # [0, 2, 2, 3]

    # Find where indices match the selection
    eq = tf.equal(
        tf.expand_dims(selection_axis_indices, 1), tf.cast(selection_indices, tf.int64),
    )  # [[0, 0], [1, 0], [1, 0], [0, 1]]

    # Mask for selected values
    sel = tf.reduce_any(eq, axis=1)  # [0, 1, 1 ,1]

    # Selected values
    selected_values = tf.boolean_mask(input_tensor.values, sel, axis=0)  # [2, 2, 3]

    # Construct the new index values for axis on which selection has been made.
    n_indices = tf.cast(n_indices, tf.int64)
    selection_axis_indices_new = tf.reduce_sum(
        tf.cast(eq, tf.int64) * tf.range(n_indices), axis=1
    )  # [0, 0, 0, 1]
    selection_axis_indices_new = tf.boolean_mask(
        selection_axis_indices_new, sel, axis=0
    )  # [0, 0, 1]

    # New full indices tensor
    indices_new = tf.boolean_mask(
        input_tensor.indices, sel, axis=0
    )  # [[2,2], [2,3], [3,1]]
    indices_new = tf.concat(
        [
            indices_new[:, :axis],
            tf.expand_dims(selection_axis_indices_new, 1),
            indices_new[:, axis + 1 :],
        ],
        axis=1,
    )  # [[0,2], [0,3], [0,1]]

    # New shape
    shape_new = tf.concat(
        [
            input_tensor.dense_shape[:axis],
            [n_indices],
            input_tensor.dense_shape[axis + 1 :],
        ],
        axis=0,
    )  # [2, 5]
    return tf.SparseTensor(indices_new, selected_values, shape_new)


def gather(
    tensor: Union[tf.Tensor, tf.SparseTensor], indices: tf.Tensor
) -> Union[tf.Tensor, tf.SparseTensor]:
    """Gather indices from dense or sparse tensors."""
    if isinstance(tensor, tf.Tensor):
        return tf.gather(tensor, tf.cast(indices, dtype=tf.int32))
    elif isinstance(tensor, tf.SparseTensor):
        return gather_sparse(tensor, indices)
