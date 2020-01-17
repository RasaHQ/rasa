from collections import defaultdict
import logging
import scipy.sparse
import typing
from typing import List, Optional, Text, Dict, Tuple, Union, Generator, Any, NamedTuple
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf


logger = logging.getLogger(__name__)


# type for all tf session related data
SessionDataType = Dict[Text, List[np.ndarray]]
# signature for all session related data
# (boolean indicates whether data are sparse or not)
# (list values represent the shape)
SessionDataSignature = Dict[Text, List[Tuple[bool, List[int]]]]


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


def create_label_ids(label_ids: "np.ndarray") -> "np.ndarray":
    """Convert various size label_ids into single dim array.

    for multi-label y, map each distinct row to a string repr
    using join because str(row) uses an ellipsis if len(row) > 1000.
    Idea taken from sklearn's stratify split.
    """

    if label_ids.ndim == 1:
        return label_ids

    if label_ids.ndim == 2 and label_ids.shape[-1] == 1:
        return label_ids[:, 0]

    if label_ids.ndim == 2:
        return np.array([" ".join(row.astype("str")) for row in label_ids])

    if label_ids.ndim == 3 and label_ids.shape[-1] == 1:
        return np.array([" ".join(row.astype("str")) for row in label_ids[:, :, 0]])

    raise ValueError("Unsupported label_ids dimensions")


# noinspection PyPep8Naming
def train_val_split(
    session_data: SessionDataType,
    evaluate_on_num_examples: int,
    random_seed: int,
    label_key: Text,
) -> Tuple[SessionDataType, SessionDataType]:
    """Create random hold out validation set using stratified split."""

    if label_key not in session_data or len(session_data[label_key]) > 1:
        raise ValueError(f"Key '{label_key}' not in SessionData.")

    label_ids = create_label_ids(session_data[label_key][0])

    label_counts = dict(zip(*np.unique(label_ids, return_counts=True, axis=0)))

    check_train_test_sizes(evaluate_on_num_examples, label_counts, session_data)

    counts = np.array([label_counts[label] for label in label_ids])

    multi_values = [v[counts > 1] for values in session_data.values() for v in values]

    solo_values = [v[counts == 1] for values in session_data.values() for v in values]

    output_values = train_test_split(
        *multi_values,
        test_size=evaluate_on_num_examples,
        random_state=random_seed,
        stratify=label_ids[counts > 1],
    )

    session_data_train, session_data_val = convert_train_test_split(
        output_values, session_data, solo_values
    )

    return session_data_train, session_data_val


def check_train_test_sizes(
    evaluate_on_num_examples: int,
    label_counts: Dict[Any, int],
    session_data: SessionDataType,
):
    """Check whether the evaluation data set is too large or too small."""

    num_examples = get_number_of_examples(session_data)

    if evaluate_on_num_examples >= num_examples - len(label_counts):
        raise ValueError(
            f"Validation set of {evaluate_on_num_examples} is too large. Remaining "
            f"train set should be at least equal to number of classes "
            f"{len(label_counts)}."
        )
    elif evaluate_on_num_examples < len(label_counts):
        raise ValueError(
            f"Validation set of {evaluate_on_num_examples} is too small. It should be "
            "at least equal to number of classes {label_counts}."
        )


def convert_train_test_split(
    output_values: List[Any], session_data: SessionDataType, solo_values: List[Any]
) -> Tuple[SessionDataType, SessionDataType]:
    """Convert the output of sklearn.model_selection.train_test_split into train and
    eval session data."""

    session_data_train = defaultdict(list)
    session_data_val = defaultdict(list)

    # output_values = x_train, x_val, y_train, y_val, z_train, z_val, etc.
    # order is kept, e.g. same order as session data keys

    # train datasets have an even index
    index = 0
    for key, values in session_data.items():
        for _ in range(len(values)):
            session_data_train[key].append(
                combine_features(output_values[index * 2], solo_values[index])
            )
            index += 1

    # val datasets have an odd index
    index = 0
    for key, values in session_data.items():
        for _ in range(len(values)):
            session_data_val[key].append(output_values[(index * 2) + 1])
            index += 1

    return session_data_train, session_data_val


def combine_features(
    feature_1: Union[np.ndarray, scipy.sparse.spmatrix],
    feature_2: Union[np.ndarray, scipy.sparse.spmatrix],
) -> Union[np.ndarray, scipy.sparse.spmatrix]:
    """Concatenate features."""

    if isinstance(feature_1, scipy.sparse.spmatrix) and isinstance(
        feature_2, scipy.sparse.spmatrix
    ):
        if feature_2.shape[0] == 0:
            return feature_1
        if feature_1.shape[0] == 0:
            return feature_2
        return scipy.sparse.vstack([feature_1, feature_2])

    return np.concatenate([feature_1, feature_2])


def shuffle_session_data(session_data: SessionDataType) -> SessionDataType:
    """Shuffle session data."""

    data_points = get_number_of_examples(session_data)
    ids = np.random.permutation(data_points)
    return session_data_for_ids(session_data, ids)


def session_data_for_ids(session_data: SessionDataType, ids: np.ndarray):
    """Filter session data by ids."""

    new_session_data = defaultdict(list)
    for k, values in session_data.items():
        for v in values:
            new_session_data[k].append(v[ids])
    return new_session_data


def split_session_data_by_label_ids(
    session_data: SessionDataType,
    label_ids: "np.ndarray",
    unique_label_ids: "np.ndarray",
) -> List[SessionDataType]:
    """Reorganize session data into a list of session data with the same labels."""

    label_data = []
    for label_id in unique_label_ids:
        ids = label_ids == label_id
        label_data.append(session_data_for_ids(session_data, ids))
    return label_data


# noinspection PyPep8Naming
def balance_session_data(
    session_data: SessionDataType, batch_size: int, shuffle: bool, label_key: Text
) -> SessionDataType:
    """Mix session data to account for class imbalance.

    This batching strategy puts rare classes approximately in every other batch,
    by repeating them. Mimics stratified batching, but also takes into account
    that more populated classes should appear more often.
    """

    if label_key not in session_data or len(session_data[label_key]) > 1:
        raise ValueError(f"Key '{label_key}' not in SessionDataType.")

    label_ids = create_label_ids(session_data[label_key][0])

    unique_label_ids, counts_label_ids = np.unique(
        label_ids, return_counts=True, axis=0
    )
    num_label_ids = len(unique_label_ids)

    # need to call every time, so that the data is shuffled inside each class
    label_data = split_session_data_by_label_ids(
        session_data, label_ids, unique_label_ids
    )

    data_idx = [0] * num_label_ids
    num_data_cycles = [0] * num_label_ids
    skipped = [False] * num_label_ids

    new_session_data = defaultdict(list)
    num_examples = get_number_of_examples(session_data)

    while min(num_data_cycles) == 0:
        if shuffle:
            indices_of_labels = np.random.permutation(num_label_ids)
        else:
            indices_of_labels = range(num_label_ids)

        for index in indices_of_labels:
            if num_data_cycles[index] > 0 and not skipped[index]:
                skipped[index] = True
                continue
            else:
                skipped[index] = False

            index_batch_size = (
                int(counts_label_ids[index] / num_examples * batch_size) + 1
            )

            for k, values in label_data[index].items():
                for i, v in enumerate(values):
                    if len(new_session_data[k]) < i + 1:
                        new_session_data[k].append([])
                    new_session_data[k][i].append(
                        v[data_idx[index] : data_idx[index] + index_batch_size]
                    )

            data_idx[index] += index_batch_size
            if data_idx[index] >= counts_label_ids[index]:
                num_data_cycles[index] += 1
                data_idx[index] = 0

            if min(num_data_cycles) > 0:
                break

    final_session_data = defaultdict(list)
    for k, values in new_session_data.items():
        for v in values:
            final_session_data[k].append(np.concatenate(np.array(v)))

    return final_session_data


def get_number_of_examples(session_data: SessionDataType) -> int:
    """Obtain number of examples in session data.

    Raise a ValueError if number of examples differ for different data in session data.
    """

    example_lengths = [v.shape[0] for values in session_data.values() for v in values]

    # check if number of examples is the same for all values
    if not all(length == example_lengths[0] for length in example_lengths):
        raise ValueError(
            f"Number of examples differs for keys '{session_data.keys()}'. Number of "
            f"examples should be the same for all data in session data."
        )

    return example_lengths[0]


def gen_batch(
    session_data: SessionDataType,
    batch_size: int,
    label_key: Text,
    batch_strategy: Text = "sequence",
    shuffle: bool = False,
) -> Generator[Tuple, None, None]:
    """Generate batches."""

    if shuffle:
        session_data = shuffle_session_data(session_data)

    if batch_strategy == "balanced":
        session_data = balance_session_data(
            session_data, batch_size, shuffle, label_key
        )

    num_examples = get_number_of_examples(session_data)
    num_batches = num_examples // batch_size + int(num_examples % batch_size > 0)

    for batch_num in range(num_batches):
        start = batch_num * batch_size
        end = start + batch_size

        yield prepare_batch(session_data, start, end)


def prepare_batch(
    session_data: SessionDataType,
    start: Optional[int] = None,
    end: Optional[int] = None,
    tuple_sizes: Optional[Dict[Text, int]] = None,
) -> Tuple[Optional[np.ndarray]]:
    """Slices session data into batch using given start and end value."""

    batch_data = []

    for key, values in session_data.items():
        # add None for not present values during processing
        if not values:
            if tuple_sizes:
                batch_data += [None] * tuple_sizes[key]
            else:
                batch_data.append(None)
            continue

        for v in values:
            if start is not None and end is not None:
                _data = v[start:end]
            elif start is not None:
                _data = v[start:]
            elif end is not None:
                _data = v[:end]
            else:
                _data = v[:]

            if isinstance(_data[0], scipy.sparse.spmatrix):
                batch_data.extend(scipy_matrix_to_values(_data))
            else:
                batch_data.append(pad_dense_data(_data))

    # len of batch_data is equal to the number of keys in session data
    return tuple(batch_data)


def scipy_matrix_to_values(array_of_sparse: np.ndarray) -> List[np.ndarray]:
    """Convert a scipy matrix into inidces, data, and shape."""

    if not isinstance(array_of_sparse[0], scipy.sparse.coo_matrix):
        array_of_sparse = [x.tocoo() for x in array_of_sparse]

    max_seq_len = max([x.shape[0] for x in array_of_sparse])

    indices = np.hstack(
        [
            np.vstack([i * np.ones_like(x.row), x.row, x.col])
            for i, x in enumerate(array_of_sparse)
        ]
    ).T
    data = np.hstack([x.data for x in array_of_sparse])

    shape = np.array((len(array_of_sparse), max_seq_len, array_of_sparse[0].shape[-1]))

    return [indices.astype(np.int64), data.astype(np.float32), shape.astype(np.int64)]


def pad_dense_data(array_of_dense: np.ndarray) -> np.ndarray:
    """Pad data of different lengths.

    Sequential data is padded with zeros. Zeros are added to the end of data.
    """

    if array_of_dense[0].ndim < 2:
        # data doesn't contain a sequence
        return array_of_dense

    data_size = len(array_of_dense)
    max_seq_len = max([x.shape[0] for x in array_of_dense])

    data_padded = np.zeros(
        [data_size, max_seq_len, array_of_dense[0].shape[-1]],
        dtype=array_of_dense[0].dtype,
    )
    for i in range(data_size):
        data_padded[i, : array_of_dense[i].shape[0], :] = array_of_dense[i]

    return data_padded.astype(np.float32)


def batch_to_session_data(
    batch: Union[Tuple[np.ndarray], Tuple[tf.Tensor]],
    session_data_signature: SessionDataSignature,
) -> Dict[Text, List[tf.Tensor]]:
    """Convert input batch tensors into batch data format.

    Batch contains any number of batch data. The order is equal to the
    key-value pairs in session data. As sparse data were converted into indices, data,
    shape before, this methods converts them into sparse tensors. Dense data is
    kept.
    """

    batch_data = defaultdict(list)

    idx = 0
    for k, signature in session_data_signature.items():
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


def batch_tuple_sizes(session_data: SessionDataType) -> Dict[Text, int]:

    # save the amount of placeholders attributed to session data keys
    tuple_sizes = defaultdict(int)

    idx = 0
    for k, values in session_data.items():
        tuple_sizes[k] = 0
        for v in values:
            if isinstance(v[0], scipy.sparse.spmatrix):
                tuple_sizes[k] += 3
                idx += 3
            else:
                tuple_sizes[k] += 1
                idx += 1

    return tuple_sizes


def create_tf_dataset(
    session_data: SessionDataType,
    batch_size: Union["tf.Tensor", int],
    label_key: Text,
    batch_strategy: Text = "sequence",
    shuffle: bool = False,
) -> "tf.data.Dataset":
    """Create tf dataset."""

    shapes, types = get_shapes_types(session_data)

    return tf.data.Dataset.from_generator(
        lambda batch_size_: gen_batch(
            session_data, batch_size_, label_key, batch_strategy, shuffle
        ),
        output_types=types,
        output_shapes=shapes,
        args=([batch_size]),
    )


def get_shapes_types(session_data: SessionDataType) -> Tuple:
    """Extract shapes and types from session data."""

    types = []
    shapes = []

    def append_shape(v: np.ndarray):
        if isinstance(v[0], scipy.sparse.spmatrix):
            # scipy matrix is converted into indices, data, shape
            shapes.append((None, v[0].ndim + 1))
            shapes.append((None,))
            shapes.append((v[0].ndim + 1))
        elif v[0].ndim == 0:
            shapes.append((None,))
        elif v[0].ndim == 1:
            shapes.append((None, v[0].shape[-1]))
        else:
            shapes.append((None, None, v[0].shape[-1]))

    def append_type(v: np.ndarray):
        if isinstance(v[0], scipy.sparse.spmatrix):
            # scipy matrix is converted into indices, data, shape
            types.append(tf.int64)
            types.append(tf.float32)
            types.append(tf.int64)
        else:
            types.append(tf.float32)

    for values in session_data.values():
        for v in values:
            append_shape(v)
            append_type(v)

    return tuple(shapes), tuple(types)


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


def output_validation_stat(
    eval_init_op: "tf.Operation",
    metrics: TrainingMetrics,
    session: "tf.Session",
    is_training: "tf.Session",
    batch_size_in: "tf.Tensor",
    ep_batch_size: int,
) -> TrainingMetrics:
    """Output training statistics"""

    session.run(eval_init_op, feed_dict={batch_size_in: ep_batch_size})
    ep_val_metrics = TrainingMetrics(
        loss=defaultdict(lambda: 0.0), score=defaultdict(lambda: 0.0)
    )
    batches_per_epoch = 0
    while True:
        try:
            batch_val_metrics = session.run([metrics], feed_dict={is_training: False})
            batch_val_metrics = batch_val_metrics[0]
            batches_per_epoch += 1
            for name, value in batch_val_metrics.loss.items():
                ep_val_metrics.loss[name] += value
            for name, value in batch_val_metrics.score.items():
                ep_val_metrics.score[name] += value

        except tf.errors.OutOfRangeError:
            break

    for name, value in ep_val_metrics.loss.items():
        ep_val_metrics.loss[name] = value / batches_per_epoch
    for name, value in ep_val_metrics.score.items():
        ep_val_metrics.score[name] = value / batches_per_epoch

    return ep_val_metrics


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
