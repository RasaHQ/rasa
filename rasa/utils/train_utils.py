import logging
from typing import List, Optional, Text, Dict, Tuple, Union, Any, NamedTuple
import tensorflow as tf


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


def confidence_from_sim(sim: "tf.Tensor", similarity_type: Text) -> "tf.Tensor":
    if similarity_type == "cosine":
        # clip negative values to zero
        return tf.nn.relu(sim)
    else:
        # normalize result to [0, 1] with softmax
        return tf.nn.softmax(sim)


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
