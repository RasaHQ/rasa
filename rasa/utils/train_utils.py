from collections import defaultdict
import logging
from typing import List, Optional, Text, Dict, Tuple, Union, Any, NamedTuple
import numpy as np
import tensorflow as tf

from rasa.utils.tf_model_data import DataSignature

logger = logging.getLogger(__name__)


# namedtuple for training metrics
class TrainingMetrics(NamedTuple):
    loss: Dict[Text, Union[tf.Tensor, float]]
    score: Dict[Text, Union[tf.Tensor, float]]


def load_tf_config(config: Dict[Text, Any]) -> Optional[tf.compat.v1.ConfigProto]:
    """Prepare `tf.compat.v1.ConfigProto` for training"""

    if config.get("tf_config") is not None:
        return tf.compat.v1.ConfigProto(**config.pop("tf_config"))
    else:
        return None


def batch_to_model_data_format(
    batch: Union[Tuple[np.ndarray], Tuple[tf.Tensor]],
    data_signature: Dict[Text, List[DataSignature]],
) -> Dict[Text, List[tf.Tensor]]:
    """Convert input batch tensors into batch data format.

    Batch contains any number of batch data. The order is equal to the
    key-value pairs in session data. As sparse data were converted into indices, data,
    shape before, this methods converts them into sparse tensors. Dense data is
    kept.
    """

    batch_data = defaultdict(list)

    idx = 0
    for k, signature in data_signature.items():
        for is_sparse, shape in signature:
            if is_sparse:
                # explicitly substitute last dimension in shape with known static value
                batch_data[k].append(
                    tf.SparseTensor(
                        batch[idx],
                        batch[idx + 1],
                        [batch[idx + 2][0], batch[idx + 2][1], shape[-1]],
                    )
                )
                idx += 3
            else:
                batch_data[k].append(batch[idx])
                idx += 1

    return batch_data


def confidence_from_sim(sim: "tf.Tensor", similarity_type: Text) -> "tf.Tensor":
    if similarity_type == "cosine":
        # clip negative values to zero
        return tf.nn.relu(sim)
    else:
        # normalize result to [0, 1] with softmax
        return tf.nn.softmax(sim)


def linearly_increasing_batch_size(
    epoch: int, batch_size: Union[List[int], int], epochs: int
) -> int:
    """Linearly increase batch size with every epoch.

    The idea comes from https://arxiv.org/abs/1711.00489.
    """

    if not isinstance(batch_size, list):
        return int(batch_size)

    if epochs > 1:
        return int(
            batch_size[0] + epoch * (batch_size[1] - batch_size[0]) / (epochs - 1)
        )
    else:
        return int(batch_size[0])


def extract_attention(attention_weights) -> Optional["tf.Tensor"]:
    """Extract attention probabilities from t2t dict"""

    attention = [
        tf.expand_dims(t, 0)
        for name, t in attention_weights.items()
        # the strings come from t2t library
        if "multihead_attention/dot_product" in name and not name.endswith("/logits")
    ]

    if attention:
        return tf.concat(attention, 0)


def persist_tensor(
    name: Text,
    tensor: Union["tf.Tensor", Tuple["tf.Tensor"], List["tf.Tensor"]],
    graph: "tf.Graph",
) -> None:
    """Add tensor to collection if it is not None"""

    if tensor is not None:
        graph.clear_collection(name)
        if isinstance(tensor, tuple) or isinstance(tensor, list):
            for t in tensor:
                graph.add_to_collection(name, t)
        else:
            graph.add_to_collection(name, tensor)


def load_tensor(name: Text) -> Optional[Union["tf.Tensor", List["tf.Tensor"]]]:
    """Load tensor or set it to None"""

    tensor_list = tf.get_collection(name)

    if not tensor_list:
        return None

    if len(tensor_list) == 1:
        return tensor_list[0]

    return tensor_list
