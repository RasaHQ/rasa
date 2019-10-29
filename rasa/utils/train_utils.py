from collections import defaultdict
import logging
import scipy.sparse
import typing
from typing import (
    List,
    Optional,
    Text,
    Dict,
    Tuple,
    Union,
    Generator,
    Callable,
    Any,
    NamedTuple,
)
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


# namedtuple for all tf session related data
class SessionData(NamedTuple):
    X: Dict[Text, np.ndarray]
    Y: Dict[Text, np.ndarray]
    labels: Dict[Text, np.ndarray]


def load_tf_config(config: Dict[Text, Any]) -> Optional[tf.compat.v1.ConfigProto]:
    """Prepare `tf.compat.v1.ConfigProto` for training"""
    if config.get("tf_config") is not None:
        return tf.compat.v1.ConfigProto(**config.pop("tf_config"))
    else:
        return None


# noinspection PyPep8Naming
def train_val_split(
    session_data: "SessionData",
    evaluate_on_num_examples: int,
    random_seed: int,
    label_key: Text = "labels",
) -> Tuple["SessionData", "SessionData"]:
    """Create random hold out validation set using stratified split."""
    if label_key not in session_data.labels:
        raise ValueError(f"Key '{label_key}' not in SessionData.labels.")

    label_counts = dict(
        zip(*np.unique(session_data.labels[label_key], return_counts=True, axis=0))
    )

    num_examples = get_number_of_examples(session_data)
    if evaluate_on_num_examples >= num_examples - len(label_counts):
        raise ValueError(
            f"Validation set of {evaluate_on_num_examples} is too large. Remaining train set "
            "should be at least equal to number of classes {len(label_counts)}."
        )
    elif evaluate_on_num_examples < len(label_counts):
        raise ValueError(
            f"Validation set of {evaluate_on_num_examples} is too small. It should be "
            "at least equal to number of classes {label_counts}."
        )

    counts = np.array([label_counts[label] for label in session_data.labels[label_key]])

    multi_values = []
    [multi_values.append(v[counts > 1]) for k, v in session_data.X.items()]
    [multi_values.append(v[counts > 1]) for k, v in session_data.Y.items()]
    [multi_values.append(v[counts > 1]) for k, v in session_data.labels.items()]

    solo_values = []
    [solo_values.append(v[counts == 1]) for k, v in session_data.X.items()]
    [solo_values.append(v[counts == 1]) for k, v in session_data.Y.items()]
    [solo_values.append(v[counts == 1]) for k, v in session_data.labels.items()]

    keys = [k for d in session_data for k, v in d.items()]

    output_values = train_test_split(
        *multi_values,
        test_size=evaluate_on_num_examples,
        random_state=random_seed,
        stratify=session_data.labels[label_key][counts > 1],
    )

    X_train = {}
    Y_train = {}
    labels_train = {}
    X_val = {}
    Y_val = {}
    labels_val = {}

    # output_values = x_train, x_val, y_train, y_val, z_train, z_val, etc.
    # order is kept, so first session_data.X values, then session_data.Y values, and
    # finally session_data.labels values
    for i in range(len(session_data.X)):
        X_train[keys[i]] = combine_features(output_values[i * 2], solo_values[i])

    for i in range(len(session_data.X), len(session_data.X) + len(session_data.Y)):
        Y_train[keys[i]] = combine_features(output_values[i * 2], solo_values[i])

    for i in range(
        len(session_data.X) + len(session_data.Y),
        len(session_data.X) + len(session_data.Y) + len(session_data.labels),
    ):
        labels_train[keys[i]] = combine_features(output_values[i * 2], solo_values[i])

    for i in range(len(session_data.X)):
        X_val[keys[i]] = output_values[(i * 2) + 1]

    for i in range(len(session_data.X), len(session_data.X) + len(session_data.Y)):
        Y_val[keys[i]] = output_values[(i * 2) + 1]

    for i in range(
        len(session_data.X) + len(session_data.Y),
        len(session_data.X) + len(session_data.Y) + len(session_data.labels),
    ):
        labels_val[keys[i]] = output_values[(i * 2) + 1]

    return (
        SessionData(X_train, Y_train, labels_train),
        SessionData(X_val, Y_val, labels_val),
    )


def combine_features(
    feature_1: Union[np.ndarray, scipy.sparse.spmatrix],
    feature_2: Union[np.ndarray, scipy.sparse.spmatrix],
) -> Union[np.ndarray, scipy.sparse.spmatrix]:
    if isinstance(feature_1, scipy.sparse.spmatrix) and isinstance(
        feature_2, scipy.sparse.spmatrix
    ):
        if feature_2.shape[0] == 0:
            return feature_1
        if feature_1.shape[0] == 0:
            return feature_2
        return scipy.sparse.vstack([feature_1, feature_2])

    return np.concatenate([feature_1, feature_2])


def shuffle_session_data(session_data: "SessionData") -> "SessionData":
    """Shuffle session data."""
    data_points = get_number_of_examples(session_data)
    ids = np.random.permutation(data_points)
    return session_data_for_ids(session_data, ids)


def session_data_for_ids(session_data: SessionData, ids: np.ndarray):
    """Filter session data by ids."""
    X = {k: v[ids] for k, v in session_data.X.items()}
    Y = {k: v[ids] for k, v in session_data.Y.items()}
    labels = {k: v[ids] for k, v in session_data.labels.items()}

    return SessionData(X, Y, labels)


def split_session_data_by_label(
    session_data: "SessionData", label_key: Text, unique_label_ids: "np.ndarray"
) -> List["SessionData"]:
    """Reorganize session data into a list of session data with the same labels."""
    if label_key not in session_data.labels:
        raise ValueError(f"Key '{label_key}' not in SessionData.labels.")

    label_data = []
    for label_id in unique_label_ids:
        ids = session_data.labels[label_key] == label_id
        label_data.append(session_data_for_ids(session_data, ids))
    return label_data


# noinspection PyPep8Naming
def balance_session_data(
    session_data: "SessionData",
    batch_size: int,
    shuffle: bool,
    label_key: Text = "labels",
) -> "SessionData":
    """Mix session data to account for class imbalance.

    This batching strategy puts rare classes approximately in every other batch,
    by repeating them. Mimics stratified batching, but also takes into account
    that more populated classes should appear more often.
    """
    if label_key not in session_data.labels:
        raise ValueError(f"Key '{label_key}' not in SessionData.labels.")

    num_examples = get_number_of_examples(session_data)

    unique_label_ids, counts_label_ids = np.unique(
        session_data.labels[label_key], return_counts=True, axis=0
    )
    num_label_ids = len(unique_label_ids)

    # need to call every time, so that the data is shuffled inside each class
    label_data = split_session_data_by_label(session_data, label_key, unique_label_ids)

    data_idx = [0] * num_label_ids
    num_data_cycles = [0] * num_label_ids
    skipped = [False] * num_label_ids

    new_X = defaultdict(list)
    new_Y = defaultdict(list)
    new_labels = defaultdict(list)

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

            for k, v in label_data[index].X.items():
                new_X[k].append(v[data_idx[index] : data_idx[index] + index_batch_size])
            for k, v in label_data[index].Y.items():
                new_Y[k].append(v[data_idx[index] : data_idx[index] + index_batch_size])
            for k, v in label_data[index].labels.items():
                new_labels[k].append(
                    v[data_idx[index] : data_idx[index] + index_batch_size]
                )

            data_idx[index] += index_batch_size
            if data_idx[index] >= counts_label_ids[index]:
                num_data_cycles[index] += 1
                data_idx[index] = 0

            if min(num_data_cycles) > 0:
                break

    new_X = {k: np.concatenate(v) for k, v in new_X.items()}
    new_Y = {k: np.concatenate(v) for k, v in new_Y.items()}
    new_labels = {k: np.concatenate(v) for k, v in new_labels.items()}

    return SessionData(X=new_X, Y=new_Y, labels=new_labels)


def get_number_of_examples(session_data: SessionData):
    example_lengths = [v.shape[0] for v in session_data.X.values()]

    # check if number of examples is the same for all X
    if not all(length == example_lengths[0] for length in example_lengths):
        raise ValueError(
            f"Number of examples differs for X ({session_data.X.keys()}). There should "
            f"be the same."
        )

    return example_lengths[0]


def gen_batch(
    session_data: "SessionData",
    batch_size: int,
    batch_strategy: Text = "sequence",
    shuffle: bool = False,
) -> Generator[Tuple["np.ndarray", "np.ndarray"], None, None]:
    """Generate batches."""

    if shuffle:
        session_data = shuffle_session_data(session_data)

    if batch_strategy == "balanced":
        session_data = balance_session_data(session_data, batch_size, shuffle)

    num_examples = get_number_of_examples(session_data)
    num_batches = num_examples // batch_size + int(num_examples % batch_size > 0)

    for batch_num in range(num_batches):
        start = batch_num * batch_size
        end = (batch_num + 1) * batch_size

        batch_data = [sparse_to_dense(v[start:end]) for v in session_data.X.values()]
        batch_data = batch_data + [
            sparse_to_dense(v[start:end]) for v in session_data.Y.values()
        ]
        batch_data = batch_data + [
            sparse_to_dense(v[start:end]) for v in session_data.labels.values()
        ]

        yield tuple(batch_data)


def sparse_to_dense(examples: Union[np.ndarray, scipy.sparse.csr_matrix]):
    if isinstance(examples[0], scipy.sparse.spmatrix):
        return examples.toarray()
    return examples


# noinspection PyPep8Naming
def create_tf_dataset(
    session_data: "SessionData",
    batch_size: Union["tf.Tensor", int],
    batch_strategy: Text = "sequence",
    shuffle: bool = False,
) -> "tf.data.Dataset":
    """Create tf dataset."""

    # set batch and sequence length to None
    shapes = _get_shape(session_data)
    types = tuple([np.float32] * len(shapes))

    return tf.data.Dataset.from_generator(
        lambda batch_size_: gen_batch(
            session_data, batch_size_, batch_strategy, shuffle
        ),
        output_types=types,
        output_shapes=shapes,
        args=([batch_size]),
    )


def _get_shape(session_data: SessionData) -> Tuple:
    shapes = []

    def append_shape(v: Union[np.ndarray, scipy.sparse.spmatrix]):
        if v[0].ndim == 0:
            shapes.append((None))
        elif v[0].ndim == 1:
            shapes.append((None, v.shape[-1]))
        else:
            shapes.append((None, None, v.shape[-1]))

    for v in session_data.X.values():
        append_shape(v)
    for v in session_data.Y.values():
        append_shape(v)
    for v in session_data.labels.values():
        append_shape(v)

    return tuple(shapes)


def create_iterator_init_datasets(
    session_data: "SessionData",
    eval_session_data: "SessionData",
    batch_size: Union["tf.Tensor", int],
    batch_strategy: Text,
) -> Tuple["tf.data.Iterator", "tf.Operation", "tf.Operation"]:
    """Create iterator and init datasets."""

    train_dataset = create_tf_dataset(
        session_data, batch_size, batch_strategy=batch_strategy, shuffle=True
    )

    iterator = tf.data.Iterator.from_structure(
        train_dataset.output_types,
        train_dataset.output_shapes,
        output_classes=train_dataset.output_classes,
    )

    train_init_op = iterator.make_initializer(train_dataset)

    if eval_session_data is not None:
        eval_init_op = iterator.make_initializer(
            create_tf_dataset(eval_session_data, batch_size)
        )
    else:
        eval_init_op = None

    return iterator, train_init_op, eval_init_op


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
            name="hidden_layer_{}_{}".format(layer_name_suffix, i),
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
            "Wrong similarity type '{}', "
            "should be 'cosine' or 'inner'"
            "".format(similarity_type)
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
        name="embed_layer_{}".format(layer_name_suffix),
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


def _tf_calc_iou_mask(
    pos_b: "tf.Tensor", all_bs: "tf.Tensor", neg_ids: "tf.Tensor"
) -> "tf.Tensor":
    """Calculate IOU mask for given indices"""

    pos_b_in_flat = tf.expand_dims(pos_b, -2)
    neg_b_in_flat = _tf_sample_neg(tf.shape(pos_b)[0], all_bs, neg_ids)

    intersection_b_in_flat = tf.minimum(neg_b_in_flat, pos_b_in_flat)
    union_b_in_flat = tf.maximum(neg_b_in_flat, pos_b_in_flat)

    iou = tf.reduce_sum(intersection_b_in_flat, -1) / tf.reduce_sum(union_b_in_flat, -1)
    return 1.0 - tf.nn.relu(tf.sign(1.0 - iou))


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

    bad_negs = _tf_calc_iou_mask(raw_flat, all_raw, neg_ids)
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
            "Wrong loss type '{}', "
            "should be 'margin' or 'softmax'"
            "".format(loss_type)
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
            "Validation accuracy is calculated every {} epochs"
            "".format(evaluate_every_num_epochs)
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

        postfix_dict = {
            "loss": "{:.3f}".format(train_loss),
            "acc": "{:.3f}".format(train_acc),
        }

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
                {
                    "val_loss": "{:.3f}".format(val_loss),
                    "val_acc": "{:.3f}".format(val_acc),
                }
            )

        pbar.set_postfix(postfix_dict)

    final_message = (
        "Finished training embedding policy, "
        "train loss={:.3f}, train accuracy={:.3f}"
        "".format(train_loss, train_acc)
    )
    if eval_init_op is not None:
        final_message += (
            ", validation loss={:.3f}, validation accuracy={:.3f}"
            "".format(val_loss, val_acc)
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


def persist_tensor(name: Text, tensor: "tf.Tensor", graph: "tf.Graph") -> None:
    """Add tensor to collection if it is not None"""

    if tensor is not None:
        graph.clear_collection(name)
        graph.add_to_collection(name, tensor)


def load_tensor(name: Text) -> Optional["tf.Tensor"]:
    """Load tensor or set it to None"""

    tensor_list = tf.get_collection(name)
    return tensor_list[0] if tensor_list else None
