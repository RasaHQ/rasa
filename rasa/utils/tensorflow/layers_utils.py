import tensorflow as tf
from tensorflow import Tensor, TensorShape
from typing import Union


def random_indices(batch_size: Tensor, n: Tensor, n_max: Tensor) -> Tensor:
    """Creates batch_size * n random indices that run from 0 to n_max.
    
    Args:
        batch_size: Number of items in each batch
        n: Number of random indices in each example
        n_max: Maximum index

    Returns:
        A uniformly distributed integer tensor of indices
    """
    return tf.random.uniform(shape=(batch_size, n), maxval=n_max, dtype=tf.int32)


def batch_flatten(x: Tensor) -> Tensor:
    """Makes a tensor 2D.
    
    Args:
        x: Any tensor with at least 2 dimensions
        
    Returns:
        The reshaped tensor
    """
    return tf.reshape(x, (-1, x.shape[-1]))


def pad_right(
    x: Tensor, target_shape: TensorShape, value: Union[int, float] = 0
) -> Tensor:
    """Creates a tensor of shape `target_shape` by padding it with `value` on the right.
    
    Args:
        x: Any tensor
        target_shape: Shape of the padded x; must be at least as large as the
            shape of x in all dimensions

    Returns:
        A tensor like x, but padded with zeros
    """
    current_shape = tf.shape(x)
    right_padding = tf.expand_dims(
        tf.convert_to_tensor(target_shape - current_shape), -1
    )
    padding = tf.concat([tf.zeros_like(right_padding), right_padding], -1)
    return tf.pad(x, padding, "CONSTANT", constant_values=value)
