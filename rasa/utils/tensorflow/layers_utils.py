import tensorflow as tf
from tensorflow import Tensor


def random_indices(batch_size: Tensor, n: Tensor, n_max: Tensor) -> Tensor:
    return tf.random.uniform(shape=(batch_size, n), maxval=n_max, dtype=tf.int32)


def batch_flatten(x: Tensor) -> Tensor:
    """Make tensor 2D."""
    return tf.reshape(x, (-1, x.shape[-1]))
