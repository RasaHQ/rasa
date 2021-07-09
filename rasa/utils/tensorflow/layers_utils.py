import tensorflow as tf
from tensorflow import Tensor
from typing import Union, Optional


def random_indices(
    batch_size: Union[Tensor, int], n: Union[Tensor, int], n_max: Union[Tensor, int]
) -> Tensor:
    """Creates `batch_size * n` random indices that run from `0` to `n_max`.

    Args:
        batch_size: Number of items in each batch
        n: Number of random indices in each example
        n_max: Maximum index (excluded)

    Returns:
        A uniformly distributed integer tensor of indices
    """
    return tf.random.uniform(shape=(batch_size, n), maxval=n_max, dtype=tf.int32)


def batch_flatten(x: Tensor) -> Tensor:
    """Flattens all but last dimension of `x` so it becomes 2D.

    Args:
        x: Any tensor with at least 2 dimensions

    Returns:
        The reshaped tensor, where all but the last dimension
        are flattened into the first dimension
    """
    return tf.reshape(x, (-1, x.shape[-1]))


def get_candidate_values(
    x: tf.Tensor,  # (batch_size, ...)
    candidate_ids: tf.Tensor,  # (batch_size, num_candidates)
) -> tf.Tensor:
    """Gathers candidate values according to IDs.

    Args:
        x: Any tensor with at least one dimension
        candidate_ids: Indicator for which candidates to gather

    Returns:
        A tensor of shape `(batch_size, 1, num_candidates, tf.shape(x)[-1])`, where
        for each batch example, we generate a list of `num_candidates` vectors, and
        each candidate is chosen from `x` according to the candidate id. For example:

        ```
        x = [[0 1 2],
                [3 4 5],
                [6 7 8]]
        candidate_ids = [[0, 1], [0, 0], [2, 0]]
        gives
        [
            [[0 1 2],
             [3 4 5]],
            [[0 1 2],
             [0 1 2]],
            [[6 7 8],
             [0 1 2]]
        ]
        ```
    """
    tiled_x = tf.tile(
        tf.expand_dims(batch_flatten(x), 0), (tf.shape(candidate_ids)[0], 1, 1),
    )
    candidate_values = tf.gather(tiled_x, candidate_ids, batch_dims=1)

    return candidate_values  # (batch_size, num_candidates, tf.shape(x)[-1])


def reduce_mean_equal(
    x: tf.Tensor, y: tf.Tensor, mask: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """Computes the mean number of matches between x and y.

    If `x` and `y` have `n` dimensions, then the mean equal
    number of indices is calculated for the last dimension by
    only taking the valid indices into consideration
    (from the mask) and then it is averaged over all
    other `n-1` dimensions.

    For e.g., if:

    x = [[1,2,3,4]
        [5,6,7,8]]
    y = [[1,2,3,4]
        [5,6,0,0]]
    mask = [[1,1,1,1],
            [1,1,1,0]]

    then the output will be calculated as `((4/4) + 2/3) / 2`

    Args:
        x: Any numeric tensor.
        y: Another tensor with same shape and type as x.
        mask: Tensor with a mask to distinguish actual indices from padding indices.
            Shape should be the same as `x` and `y`.

    Returns:
        The mean of "x == y"
    """
    if mask is None:
        mask = tf.ones_like(x, dtype=tf.float32)

    equal_indices = tf.cast(tf.math.equal(x, y), tf.float32) * mask
    return tf.reduce_mean(
        tf.math.divide_no_nan(
            tf.reduce_sum(equal_indices, axis=-1), tf.reduce_sum(mask, axis=-1)
        )
    )
