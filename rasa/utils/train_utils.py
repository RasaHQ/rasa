from collections import defaultdict
import logging
import scipy.sparse
import typing
from typing import List, Optional, Text, Dict, Tuple, Union, Generator, Callable, Any
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensor2tensor.models.transformer import (
    transformer_base,
    transformer_prepare_encoder,
    transformer_encoder,
)
from tensor2tensor.layers.common_attention import large_compatible_negative
from rasa.utils.common import is_logging_disabled


if typing.TYPE_CHECKING:
    from tensor2tensor.utils.hparam import HParams

# avoid warning println on contrib import - remove for tf 2
tf.contrib._warning = None

logger = logging.getLogger(__name__)


# type for all tf session related data
SessionDataType = Dict[Text, List[np.ndarray]]


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
    batch: Union[Tuple[np.ndarray], Tuple[tf.Tensor]], session_data: SessionDataType
) -> Tuple[Dict[Text, List[tf.Tensor]], Dict[Text, int]]:
    """Convert input batch tensors into batch data format.

    Batch contains any number of batch data. The order is equal to the
    key-value pairs in session data. As sparse data were converted into indices, data,
    shape before, this methods converts them into sparse tensors. Dense data is
    kept.
    """

    batch_data = defaultdict(list)
    # save the amount of placeholders attributed to session data keys
    tuple_sizes = defaultdict(int)

    idx = 0
    for k, values in session_data.items():
        tuple_sizes[k] = 0
        for v in values:
            if isinstance(v[0], scipy.sparse.spmatrix):
                # explicitly substitute last dimension in shape with known static value
                batch_data[k].append(
                    tf.SparseTensor(
                        batch[idx],
                        batch[idx + 1],
                        [batch[idx + 2][0], batch[idx + 2][1], v[0].shape[-1]],
                    )
                )
                tuple_sizes[k] += 3
                idx += 3
            else:
                batch_data[k].append(batch[idx])
                tuple_sizes[k] += 1
                idx += 1

    return batch_data, tuple_sizes


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


def create_iterator_init_datasets(
    session_data: SessionDataType,
    eval_session_data: SessionDataType,
    batch_size: Union["tf.Tensor", int],
    batch_strategy: Text,
    label_key: Text,
) -> Tuple["tf.data.Iterator", "tf.Operation", "tf.Operation"]:
    """Create iterator and init datasets."""

    train_dataset = create_tf_dataset(
        session_data,
        batch_size,
        label_key=label_key,
        batch_strategy=batch_strategy,
        shuffle=True,
    )

    iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types, train_dataset.output_shapes
    )

    train_init_op = iterator.make_initializer(train_dataset)

    if eval_session_data is not None:
        eval_init_op = iterator.make_initializer(
            create_tf_dataset(eval_session_data, batch_size, label_key=label_key)
        )
    else:
        eval_init_op = None

    return iterator, train_init_op, eval_init_op


# noinspection PyPep8Naming
def tf_dense_layer_for_sparse(
    inputs: tf.SparseTensor,
    units: int,
    name: Text,
    C2: float,
    activation: Optional[Callable] = tf.nn.relu,
    use_bias: bool = True,
) -> tf.Tensor:
    """Dense layer for sparse input tensor"""

    if not isinstance(inputs, tf.SparseTensor):
        raise ValueError("Input tensor should be sparse.")

    with tf.variable_scope("dense_layer_for_sparse_" + name, reuse=tf.AUTO_REUSE):
        kernel_regularizer = tf.contrib.layers.l2_regularizer(C2)
        kernel = tf.get_variable(
            "kernel",
            shape=[inputs.shape[-1], units],
            dtype=inputs.dtype,
            regularizer=kernel_regularizer,
        )
        bias = tf.get_variable("bias", shape=[units], dtype=inputs.dtype)

        # outputs will be 2D
        outputs = tf.sparse.matmul(
            tf.sparse.reshape(inputs, [-1, int(inputs.shape[-1])]), kernel
        )

        if len(inputs.shape) == 3:
            # reshape back
            outputs = tf.reshape(
                outputs, (tf.shape(inputs)[0], tf.shape(inputs)[1], -1)
            )

        if use_bias:
            outputs = tf.nn.bias_add(outputs, bias)

    if activation is None:
        return outputs

    return activation(outputs)


# noinspection PyPep8Naming
def create_tf_fnn(
    x_in: "tf.Tensor",
    layer_sizes: List[int],
    droprate: float,
    C2: float,
    is_training: "tf.Tensor",
    layer_name_suffix: Text,
    activation: Optional[Callable] = tf.nn.relu,
    use_bias: bool = True,
    kernel_initializer: Optional["tf.keras.initializers.Initializer"] = None,
) -> "tf.Tensor":
    """Create nn with hidden layers and name suffix."""

    reg = tf.contrib.layers.l2_regularizer(C2)
    x = tf.nn.relu(x_in)
    for i, layer_size in enumerate(layer_sizes):
        x = tf.layers.dense(
            inputs=x,
            units=layer_size,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=reg,
            name=f"hidden_layer_{layer_name_suffix}_{i}",
            reuse=tf.AUTO_REUSE,
        )
        x = tf.layers.dropout(x, rate=droprate, training=is_training)
    return x


def tf_normalize_if_cosine(x: "tf.Tensor", similarity_type: Text) -> "tf.Tensor":
    """Normalize embedding if similarity type is cosine."""

    if similarity_type == "cosine":
        return tf.nn.l2_normalize(x, -1)
    elif similarity_type == "inner":
        return x
    else:
        raise ValueError(
            f"Wrong similarity type '{similarity_type}', "
            f"should be 'cosine' or 'inner'"
        )


# noinspection PyPep8Naming
def create_tf_embed(
    x: "tf.Tensor",
    embed_dim: int,
    C2: float,
    similarity_type: Text,
    layer_name_suffix: Text,
) -> "tf.Tensor":
    """Create dense embedding layer with a name."""

    reg = tf.contrib.layers.l2_regularizer(C2)
    embed_x = tf.layers.dense(
        inputs=x,
        units=embed_dim,
        activation=None,
        kernel_regularizer=reg,
        name=f"embed_layer_{layer_name_suffix}",
        reuse=tf.AUTO_REUSE,
    )
    # normalize embedding vectors for cosine similarity
    return tf_normalize_if_cosine(embed_x, similarity_type)


def create_t2t_hparams(
    num_transformer_layers: int,
    transformer_size: int,
    num_heads: int,
    droprate: float,
    pos_encoding: Text,
    max_seq_length: int,
    is_training: "tf.Tensor",
) -> "HParams":
    """Create parameters for t2t transformer."""

    hparams = transformer_base()

    hparams.num_hidden_layers = num_transformer_layers
    hparams.hidden_size = transformer_size
    # it seems to be factor of 4 for transformer architectures in t2t
    hparams.filter_size = hparams.hidden_size * 4
    hparams.num_heads = num_heads
    hparams.relu_dropout = droprate
    hparams.pos = pos_encoding

    hparams.max_length = max_seq_length

    hparams.unidirectional_encoder = True

    hparams.self_attention_type = "dot_product_relative_v2"
    hparams.max_relative_position = 5
    hparams.add_relative_to_values = True

    # When not in training mode, set all forms of dropout to zero.
    for key, value in hparams.values().items():
        if key.endswith("dropout") or key == "label_smoothing":
            setattr(hparams, key, value * tf.cast(is_training, tf.float32))

    return hparams


# noinspection PyUnresolvedReferences
# noinspection PyPep8Naming
def create_t2t_transformer_encoder(
    x_in: "tf.Tensor",
    mask: "tf.Tensor",
    attention_weights: Dict[Text, "tf.Tensor"],
    hparams: "HParams",
    C2: float,
    is_training: "tf.Tensor",
) -> "tf.Tensor":
    """Create t2t transformer encoder."""

    with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
        x = create_tf_fnn(
            x_in,
            [hparams.hidden_size],
            hparams.layer_prepostprocess_dropout,
            C2,
            is_training,
            layer_name_suffix="pre_embed",
            activation=None,
            use_bias=False,
            kernel_initializer=tf.random_normal_initializer(
                0.0, hparams.hidden_size ** -0.5
            ),
        )
        if hparams.multiply_embedding_mode == "sqrt_depth":
            x *= hparams.hidden_size ** 0.5

        x *= tf.expand_dims(mask, -1)
        (
            x,
            self_attention_bias,
            encoder_decoder_attention_bias,
        ) = transformer_prepare_encoder(x, None, hparams)

        x *= tf.expand_dims(mask, -1)

        x = tf.nn.dropout(x, 1.0 - hparams.layer_prepostprocess_dropout)

        attn_bias_for_padding = None
        # Otherwise the encoder will just use encoder_self_attention_bias.
        if hparams.unidirectional_encoder:
            attn_bias_for_padding = encoder_decoder_attention_bias

        x = transformer_encoder(
            x,
            self_attention_bias,
            hparams,
            nonpadding=mask,
            save_weights_to=attention_weights,
            attn_bias_for_padding=attn_bias_for_padding,
        )

        x *= tf.expand_dims(mask, -1)

        return tf.nn.dropout(tf.nn.relu(x), 1.0 - hparams.layer_prepostprocess_dropout)


def _tf_make_flat(x: "tf.Tensor") -> "tf.Tensor":
    """Make tensor 2D."""

    return tf.reshape(x, (-1, x.shape[-1]))


def _tf_sample_neg(
    batch_size: "tf.Tensor", all_bs: "tf.Tensor", neg_ids: "tf.Tensor"
) -> "tf.Tensor":
    """Sample negative examples for given indices"""

    tiled_all_bs = tf.tile(tf.expand_dims(all_bs, 0), (batch_size, 1, 1))

    return tf.batch_gather(tiled_all_bs, neg_ids)


def _tf_get_bad_mask(
    pos_b: "tf.Tensor", all_bs: "tf.Tensor", neg_ids: "tf.Tensor"
) -> "tf.Tensor":
    """Calculate bad mask for given indices.

    Checks that input features are different for positive negative samples.
    """

    pos_b_in_flat = tf.expand_dims(pos_b, -2)
    neg_b_in_flat = _tf_sample_neg(tf.shape(pos_b)[0], all_bs, neg_ids)

    return tf.cast(
        tf.reduce_all(tf.equal(neg_b_in_flat, pos_b_in_flat), axis=-1),
        pos_b_in_flat.dtype,
    )


def _tf_get_negs(
    all_embed: "tf.Tensor", all_raw: "tf.Tensor", raw_pos: "tf.Tensor", num_neg: int
) -> Tuple["tf.Tensor", "tf.Tensor"]:
    """Get negative examples from given tensor."""

    if len(raw_pos.shape) == 3:
        batch_size = tf.shape(raw_pos)[0]
        seq_length = tf.shape(raw_pos)[1]
    else:  # len(raw_pos.shape) == 2
        batch_size = tf.shape(raw_pos)[0]
        seq_length = 1

    raw_flat = _tf_make_flat(raw_pos)

    total_candidates = tf.shape(all_embed)[0]

    all_indices = tf.tile(
        tf.expand_dims(tf.range(0, total_candidates, 1), 0),
        (batch_size * seq_length, 1),
    )
    shuffled_indices = tf.transpose(
        tf.random.shuffle(tf.transpose(all_indices, (1, 0))), (1, 0)
    )
    neg_ids = shuffled_indices[:, :num_neg]

    bad_negs = _tf_get_bad_mask(raw_flat, all_raw, neg_ids)
    if len(raw_pos.shape) == 3:
        bad_negs = tf.reshape(bad_negs, (batch_size, seq_length, -1))

    neg_embed = _tf_sample_neg(batch_size * seq_length, all_embed, neg_ids)
    if len(raw_pos.shape) == 3:
        neg_embed = tf.reshape(
            neg_embed, (batch_size, seq_length, -1, all_embed.shape[-1])
        )

    return neg_embed, bad_negs


def sample_negatives(
    a_embed: "tf.Tensor",
    b_embed: "tf.Tensor",
    b_raw: "tf.Tensor",
    all_b_embed: "tf.Tensor",
    all_b_raw: "tf.Tensor",
    num_neg: int,
) -> Tuple[
    "tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor"
]:
    """Sample negative examples."""

    neg_dial_embed, dial_bad_negs = _tf_get_negs(
        _tf_make_flat(a_embed), _tf_make_flat(b_raw), b_raw, num_neg
    )

    neg_bot_embed, bot_bad_negs = _tf_get_negs(all_b_embed, all_b_raw, b_raw, num_neg)
    return (
        tf.expand_dims(a_embed, -2),
        tf.expand_dims(b_embed, -2),
        neg_dial_embed,
        neg_bot_embed,
        dial_bad_negs,
        bot_bad_negs,
    )


def tf_raw_sim(
    a: "tf.Tensor", b: "tf.Tensor", mask: Optional["tf.Tensor"]
) -> "tf.Tensor":
    """Calculate similarity between given tensors."""

    sim = tf.reduce_sum(a * b, -1)
    if mask is not None:
        sim *= tf.expand_dims(mask, 2)

    return sim


def tf_sim(
    pos_dial_embed: "tf.Tensor",
    pos_bot_embed: "tf.Tensor",
    neg_dial_embed: "tf.Tensor",
    neg_bot_embed: "tf.Tensor",
    dial_bad_negs: "tf.Tensor",
    bot_bad_negs: "tf.Tensor",
    mask: Optional["tf.Tensor"],
) -> Tuple["tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor"]:
    """Define similarity."""

    # calculate similarity with several
    # embedded actions for the loss
    neg_inf = large_compatible_negative(pos_dial_embed.dtype)

    sim_pos = tf_raw_sim(pos_dial_embed, pos_bot_embed, mask)
    sim_neg = tf_raw_sim(pos_dial_embed, neg_bot_embed, mask) + neg_inf * bot_bad_negs
    sim_neg_bot_bot = (
        tf_raw_sim(pos_bot_embed, neg_bot_embed, mask) + neg_inf * bot_bad_negs
    )
    sim_neg_dial_dial = (
        tf_raw_sim(pos_dial_embed, neg_dial_embed, mask) + neg_inf * dial_bad_negs
    )
    sim_neg_bot_dial = (
        tf_raw_sim(pos_bot_embed, neg_dial_embed, mask) + neg_inf * dial_bad_negs
    )

    # output similarities between user input and bot actions
    # and similarities between bot actions and similarities between user inputs
    return sim_pos, sim_neg, sim_neg_bot_bot, sim_neg_dial_dial, sim_neg_bot_dial


def tf_calc_accuracy(sim_pos: "tf.Tensor", sim_neg: "tf.Tensor") -> "tf.Tensor":
    """Calculate accuracy"""

    max_all_sim = tf.reduce_max(tf.concat([sim_pos, sim_neg], -1), -1)
    return tf.reduce_mean(
        tf.cast(tf.math.equal(max_all_sim, tf.squeeze(sim_pos, -1)), tf.float32)
    )


# noinspection PyPep8Naming
def tf_loss_margin(
    sim_pos: "tf.Tensor",
    sim_neg: "tf.Tensor",
    sim_neg_bot_bot: "tf.Tensor",
    sim_neg_dial_dial: "tf.Tensor",
    sim_neg_bot_dial: "tf.Tensor",
    mask: Optional["tf.Tensor"],
    mu_pos: float,
    mu_neg: float,
    use_max_sim_neg: bool,
    C_emb: float,
) -> "tf.Tensor":
    """Define max margin loss."""

    # loss for maximizing similarity with correct action
    loss = tf.maximum(0.0, mu_pos - tf.squeeze(sim_pos, -1))

    # loss for minimizing similarity with `num_neg` incorrect actions
    if use_max_sim_neg:
        # minimize only maximum similarity over incorrect actions
        max_sim_neg = tf.reduce_max(sim_neg, -1)
        loss += tf.maximum(0.0, mu_neg + max_sim_neg)
    else:
        # minimize all similarities with incorrect actions
        max_margin = tf.maximum(0.0, mu_neg + sim_neg)
        loss += tf.reduce_sum(max_margin, -1)

    # penalize max similarity between pos bot and neg bot embeddings
    max_sim_neg_bot = tf.maximum(0.0, tf.reduce_max(sim_neg_bot_bot, -1))
    loss += max_sim_neg_bot * C_emb

    # penalize max similarity between pos dial and neg dial embeddings
    max_sim_neg_dial = tf.maximum(0.0, tf.reduce_max(sim_neg_dial_dial, -1))
    loss += max_sim_neg_dial * C_emb

    # penalize max similarity between pos bot and neg dial embeddings
    max_sim_neg_dial = tf.maximum(0.0, tf.reduce_max(sim_neg_bot_dial, -1))
    loss += max_sim_neg_dial * C_emb

    if mask is not None:
        # mask loss for different length sequences
        loss *= mask
        # average the loss over sequence length
        loss = tf.reduce_sum(loss, -1) / tf.reduce_sum(mask, 1)

    # average the loss over the batch
    loss = tf.reduce_mean(loss)

    # add regularization losses
    loss += tf.losses.get_regularization_loss()

    return loss


def tf_loss_softmax(
    sim_pos: "tf.Tensor",
    sim_neg: "tf.Tensor",
    sim_neg_bot_bot: "tf.Tensor",
    sim_neg_dial_dial: "tf.Tensor",
    sim_neg_bot_dial: "tf.Tensor",
    mask: Optional["tf.Tensor"],
    scale_loss: bool,
) -> "tf.Tensor":
    """Define softmax loss."""

    logits = tf.concat(
        [sim_pos, sim_neg, sim_neg_bot_bot, sim_neg_dial_dial, sim_neg_bot_dial], -1
    )

    # create labels for softmax
    if len(logits.shape) == 3:
        pos_labels = tf.ones_like(logits[:, :, :1])
        neg_labels = tf.zeros_like(logits[:, :, 1:])
    else:  # len(logits.shape) == 2
        pos_labels = tf.ones_like(logits[:, :1])
        neg_labels = tf.zeros_like(logits[:, 1:])
    labels = tf.concat([pos_labels, neg_labels], -1)

    if mask is None:
        mask = 1.0

    if scale_loss:
        # mask loss by prediction confidence
        pred = tf.nn.softmax(logits)
        if len(pred.shape) == 3:
            pos_pred = pred[:, :, 0]
        else:  # len(pred.shape) == 2
            pos_pred = pred[:, 0]
        mask *= tf.pow((1 - pos_pred) / 0.5, 4)

    loss = tf.losses.softmax_cross_entropy(labels, logits, mask)
    # add regularization losses
    loss += tf.losses.get_regularization_loss()

    return loss


# noinspection PyPep8Naming
def choose_loss(
    sim_pos: "tf.Tensor",
    sim_neg: "tf.Tensor",
    sim_neg_bot_bot: "tf.Tensor",
    sim_neg_dial_dial: "tf.Tensor",
    sim_neg_bot_dial: "tf.Tensor",
    mask: Optional["tf.Tensor"],
    loss_type: Text,
    mu_pos: float,
    mu_neg: float,
    use_max_sim_neg: bool,
    C_emb: float,
    scale_loss: bool,
) -> "tf.Tensor":
    """Use loss depending on given option."""

    if loss_type == "margin":
        return tf_loss_margin(
            sim_pos,
            sim_neg,
            sim_neg_bot_bot,
            sim_neg_dial_dial,
            sim_neg_bot_dial,
            mask,
            mu_pos,
            mu_neg,
            use_max_sim_neg,
            C_emb,
        )
    elif loss_type == "softmax":
        return tf_loss_softmax(
            sim_pos,
            sim_neg,
            sim_neg_bot_bot,
            sim_neg_dial_dial,
            sim_neg_bot_dial,
            mask,
            scale_loss,
        )
    else:
        raise ValueError(
            f"Wrong loss type '{loss_type}', " f"should be 'margin' or 'softmax'"
        )


# noinspection PyPep8Naming
def calculate_loss_acc(
    a_embed: "tf.Tensor",
    b_embed: "tf.Tensor",
    b_raw: "tf.Tensor",
    all_b_embed: "tf.Tensor",
    all_b_raw: "tf.Tensor",
    num_neg: int,
    mask: Optional["tf.Tensor"],
    loss_type: Text,
    mu_pos: float,
    mu_neg: float,
    use_max_sim_neg: bool,
    C_emb: float,
    scale_loss: bool,
) -> Tuple["tf.Tensor", "tf.Tensor"]:
    """Calculate loss and accuracy."""

    (
        pos_dial_embed,
        pos_bot_embed,
        neg_dial_embed,
        neg_bot_embed,
        dial_bad_negs,
        bot_bad_negs,
    ) = sample_negatives(a_embed, b_embed, b_raw, all_b_embed, all_b_raw, num_neg)

    # calculate similarities
    (sim_pos, sim_neg, sim_neg_bot_bot, sim_neg_dial_dial, sim_neg_bot_dial) = tf_sim(
        pos_dial_embed,
        pos_bot_embed,
        neg_dial_embed,
        neg_bot_embed,
        dial_bad_negs,
        bot_bad_negs,
        mask,
    )

    acc = tf_calc_accuracy(sim_pos, sim_neg)

    loss = choose_loss(
        sim_pos,
        sim_neg,
        sim_neg_bot_bot,
        sim_neg_dial_dial,
        sim_neg_bot_dial,
        mask,
        loss_type,
        mu_pos,
        mu_neg,
        use_max_sim_neg,
        C_emb,
        scale_loss,
    )

    return loss, acc


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
    loss: "tf.Tensor",
    acc: "tf.Tensor",
    session: "tf.Session",
    is_training: "tf.Session",
    batch_size_in: "tf.Tensor",
    ep_batch_size: int,
) -> Tuple[float, float]:
    """Output training statistics"""

    session.run(eval_init_op, feed_dict={batch_size_in: ep_batch_size})
    ep_val_loss = 0
    ep_val_acc = 0
    batches_per_epoch = 0
    while True:
        try:
            batch_val_loss, batch_val_acc = session.run(
                [loss, acc], feed_dict={is_training: False}
            )
            batches_per_epoch += 1
            ep_val_loss += batch_val_loss
            ep_val_acc += batch_val_acc
        except tf.errors.OutOfRangeError:
            break

    return ep_val_loss / batches_per_epoch, ep_val_acc / batches_per_epoch


def train_tf_dataset(
    train_init_op: "tf.Operation",
    eval_init_op: "tf.Operation",
    batch_size_in: "tf.Tensor",
    loss: "tf.Tensor",
    acc: "tf.Tensor",
    train_op: "tf.Tensor",
    session: "tf.Session",
    is_training: "tf.Session",
    epochs: int,
    batch_size: Union[List[int], int],
    evaluate_on_num_examples: int,
    evaluate_every_num_epochs: int,
) -> None:
    """Train tf graph"""

    session.run(tf.global_variables_initializer())

    if evaluate_on_num_examples:
        logger.info(
            f"Validation accuracy is calculated every {evaluate_every_num_epochs} "
            f"epochs."
        )
    pbar = tqdm(range(epochs), desc="Epochs", disable=is_logging_disabled())

    train_loss = 0
    train_acc = 0
    val_loss = 0
    val_acc = 0
    for ep in pbar:

        ep_batch_size = linearly_increasing_batch_size(ep, batch_size, epochs)

        session.run(train_init_op, feed_dict={batch_size_in: ep_batch_size})

        ep_train_loss = 0
        ep_train_acc = 0
        batches_per_epoch = 0
        while True:
            try:
                _, batch_train_loss, batch_train_acc = session.run(
                    [train_op, loss, acc], feed_dict={is_training: True}
                )
                batches_per_epoch += 1
                ep_train_loss += batch_train_loss
                ep_train_acc += batch_train_acc

            except tf.errors.OutOfRangeError:
                break

        train_loss = ep_train_loss / batches_per_epoch
        train_acc = ep_train_acc / batches_per_epoch

        postfix_dict = {"loss": f"{train_loss:.3f}", "acc": f"{train_acc:.3f}"}

        if eval_init_op is not None:
            if (ep + 1) % evaluate_every_num_epochs == 0 or (ep + 1) == epochs:
                val_loss, val_acc = output_validation_stat(
                    eval_init_op,
                    loss,
                    acc,
                    session,
                    is_training,
                    batch_size_in,
                    ep_batch_size,
                )

            postfix_dict.update(
                {"val_loss": f"{val_loss:.3f}", "val_acc": f"{val_acc:.3f}"}
            )

        pbar.set_postfix(postfix_dict)

    final_message = (
        f"Finished training embedding policy, "
        f"train loss={train_loss:.3f}, train accuracy={train_acc:.3f}"
    )
    if eval_init_op is not None:
        final_message += (
            f", validation loss={val_loss:.3f}, validation accuracy={val_acc:.3f}"
        )
    logger.info(final_message)


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


def normalize(values: "np.ndarray", ranking_length: Optional[int] = 0) -> "np.ndarray":
    """Normalizes an array of positive numbers over the top `ranking_length` values.

    Other values will be set to 0.
    """

    new_values = values.copy()  # prevent mutation of the input
    if 0 < ranking_length < len(new_values):
        ranked = sorted(new_values, reverse=True)
        new_values[new_values < ranked[ranking_length - 1]] = 0

    if np.sum(new_values) > 0:
        new_values = new_values / np.sum(new_values)

    return new_values
