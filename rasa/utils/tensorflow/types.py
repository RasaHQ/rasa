from typing import Tuple, Union

import numpy as np
import tensorflow as tf

BatchData = Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
MaybeNestedBatchData = Union[Tuple[BatchData, ...], BatchData]
