from typing import Tuple, Union
import tensorflow as tf
import numpy as np

BatchData = Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
MaybeNestedBatchData = Union[Tuple[BatchData, ...], BatchData]
