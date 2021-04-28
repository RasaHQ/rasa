import tensorflow as tf
from tensorflow import Tensor


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
