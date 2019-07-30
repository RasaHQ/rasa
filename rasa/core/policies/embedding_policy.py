from collections import namedtuple
import copy
import json
import logging
import os
import pickle
import warnings

import numpy as np
import typing
from tqdm import tqdm
from typing import Any, List, Optional, Text, Dict, Tuple, Union, Generator, Callable

import rasa.utils.io
from rasa.core import utils
from rasa.core.domain import Domain
from rasa.core.featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    LabelTokenizerSingleStateFeaturizer,
    MaxHistoryTrackerFeaturizer,
)
from rasa.core.policies.policy import Policy
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.common import is_logging_disabled

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensor2tensor.models.transformer import (
    transformer_base,
    transformer_prepare_encoder,
    transformer_encoder,
)
from tensor2tensor.layers.common_attention import large_compatible_negative

if typing.TYPE_CHECKING:
    from tensor2tensor.utils.hparam import HParams

# avoid warning println on contrib import - remove for tf 2
tf.contrib._warning = None
logger = logging.getLogger(__name__)

# namedtuple for all tf session related data
SessionData = namedtuple("SessionData", ("X", "Y", "labels"))


class EmbeddingPolicy(Policy):
    """Transformer Embedding Dialogue Policy (TEDP)

    Transformer version of the REDP used in our paper https://arxiv.org/abs/1811.11707
    """

    SUPPORTS_ONLINE_TRAINING = True

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # nn architecture
        # a list of hidden layers sizes before user embed layer
        # number of hidden layers is equal to the length of this list
        "hidden_layers_sizes_dial": [],
        # a list of hidden layers sizes before bot embed layer
        # number of hidden layers is equal to the length of this list
        "hidden_layers_sizes_bot": [],
        # number of units in transformer
        "transformer_size": 128,
        # number of transformer layers
        "num_transformer_layers": 1,
        # type of positional encoding in transformer
        "pos_encoding": "timing",  # string 'timing' or 'emb'
        # max sequence length if pos_encoding='emb'
        "max_seq_length": 256,
        # number of attention heads in transformer
        "num_heads": 4,
        # training parameters
        # initial and final batch sizes:
        # batch size will be linearly increased for each epoch
        "batch_size": [8, 32],
        # how to create batches
        "batch_strategy": "sequence",  # string 'sequence' or 'balanced'
        # number of epochs
        "epochs": 1,
        # set random seed to any int to get reproducible results
        "random_seed": None,
        # embedding parameters
        # dimension size of embedding vectors
        "embed_dim": 20,
        # the type of the similarity
        "num_neg": 20,
        # flag if minimize only maximum similarity over incorrect actions
        "similarity_type": "auto",  # string 'auto' or 'cosine' or 'inner'
        # the type of the loss function
        "loss_type": "softmax",  # string 'softmax' or 'margin'
        # how similar the algorithm should try
        # to make embedding vectors for correct actions
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect actions
        "mu_neg": -0.2,  # should be -1.0 < ... < 1.0 for 'cosine'
        # the number of incorrect actions, the algorithm will minimize
        # their similarity to the user input during training
        "use_max_sim_neg": True,  # flag which loss function to use
        # regularization
        # the scale of L2 regularization
        "C2": 0.001,
        # the scale of how important is to minimize the maximum similarity
        # between embeddings of different actions
        "C_emb": 0.8,
        # dropout rate for dial nn
        "droprate_a": 0.1,
        # dropout rate for bot nn
        "droprate_b": 0.0,
        # visualization of accuracy
        # how often calculate validation accuracy
        "evaluate_every_num_epochs": 20,  # small values may hurt performance
        # how many examples to use for hold out validation set
        "evaluate_on_num_examples": 0,  # large values may hurt performance
    }

    # end default properties (DOC MARKER - don't remove)

    @staticmethod
    def _standard_featurizer(max_history: Optional[int] = None) -> "TrackerFeaturizer":
        if max_history is None:
            return FullDialogueTrackerFeaturizer(LabelTokenizerSingleStateFeaturizer())
        else:
            return MaxHistoryTrackerFeaturizer(
                LabelTokenizerSingleStateFeaturizer(), max_history=max_history
            )

    def __init__(
        self,
        featurizer: Optional["TrackerFeaturizer"] = None,
        priority: int = 1,
        graph: Optional["tf.Graph"] = None,
        session: Optional["tf.Session"] = None,
        user_placeholder: Optional["tf.Tensor"] = None,
        bot_placeholder: Optional["tf.Tensor"] = None,
        similarity_all: Optional["tf.Tensor"] = None,
        pred_confidence: Optional["tf.Tensor"] = None,
        similarity: Optional["tf.Tensor"] = None,
        dial_embed: Optional["tf.Tensor"] = None,
        bot_embed: Optional["tf.Tensor"] = None,
        all_bot_embed: Optional["tf.Tensor"] = None,
        attention_weights: Optional["tf.Tensor"] = None,
        max_history: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Declare instant variables with default values"""

        if not featurizer:
            featurizer = self._standard_featurizer(max_history)
        super(EmbeddingPolicy, self).__init__(featurizer, priority)

        self._load_params(**kwargs)

        # encode all actions with numbers
        self._encoded_all_actions = None

        # tf related instances
        self.graph = graph
        self.session = session
        self.a_in = user_placeholder
        self.b_in = bot_placeholder
        self.sim_all = similarity_all
        self.pred_confidence = pred_confidence
        self.sim = similarity

        # persisted embeddings
        self.dial_embed = dial_embed
        self.bot_embed = bot_embed
        self.all_bot_embed = all_bot_embed

        self.attention_weights = attention_weights
        # internal tf instances
        self._iterator = None
        self._train_op = None
        self._is_training = None

    # init helpers
    def _load_nn_architecture_params(self, config: Dict[Text, Any]) -> None:
        self.hidden_layers_sizes = {
            "a": config["hidden_layers_sizes_dial"],
            "b": config["hidden_layers_sizes_bot"],
        }

        self.pos_encoding = config["pos_encoding"]
        self.max_seq_length = config["max_seq_length"]
        self.num_heads = config["num_heads"]

        self.transformer_size = config["transformer_size"]
        self.num_transformer_layers = config["num_transformer_layers"]

        self.batch_size = config["batch_size"]
        self.batch_strategy = config["batch_strategy"]

        self.epochs = config["epochs"]

        self.random_seed = config["random_seed"]

    def _load_embedding_params(self, config: Dict[Text, Any]) -> None:
        self.embed_dim = config["embed_dim"]
        self.mu_pos = config["mu_pos"]
        self.mu_neg = config["mu_neg"]
        self.similarity_type = config["similarity_type"]
        self.loss_type = config["loss_type"]
        if self.similarity_type == "auto":
            if self.loss_type == "softmax":
                self.similarity_type = "inner"
            elif self.loss_type == "margin":
                self.similarity_type = "cosine"

        self.num_neg = config["num_neg"]
        self.use_max_sim_neg = config["use_max_sim_neg"]

    def _load_regularization_params(self, config: Dict[Text, Any]) -> None:
        self.C2 = config["C2"]
        self.C_emb = config["C_emb"]
        self.droprate = {"bot": config["droprate_b"], "dial": config["droprate_a"]}

    def _load_visual_params(self, config: Dict[Text, Any]) -> None:
        self.evaluate_every_num_epochs = config["evaluate_every_num_epochs"]
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs
        self.evaluate_on_num_examples = config["evaluate_on_num_examples"]

    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        config = copy.deepcopy(self.defaults)
        config.update(kwargs)

        self._tf_config = self._load_tf_config(config)
        self._load_nn_architecture_params(config)
        self._load_embedding_params(config)
        self._load_regularization_params(config)
        self._load_visual_params(config)

    # data helpers
    # noinspection PyPep8Naming
    @staticmethod
    def _labels_for_Y(data_Y: "np.ndarray") -> "np.ndarray":
        """Prepare Y data for training: extract actions indices."""

        return data_Y.argmax(axis=-1)

    # noinspection PyPep8Naming
    def _action_features_for_Y(self, labels: "np.ndarray") -> "np.ndarray":
        """Prepare Y data for training: features for action labels."""

        if len(labels.shape) == 2:
            return np.stack(
                [
                    np.stack(
                        [
                            self._encoded_all_actions[action_idx]
                            for action_idx in action_ids
                        ]
                    )
                    for action_ids in labels
                ]
            )
        else:
            return np.stack(
                [self._encoded_all_actions[action_idx] for action_idx in labels]
            )

    # noinspection PyPep8Naming
    def _create_session_data(
        self, data_X: "np.ndarray", data_Y: Optional["np.ndarray"] = None
    ) -> "SessionData":
        """Combine all tf session related data into a named tuple"""

        if data_Y is not None:
            # training time
            labels = self._labels_for_Y(data_Y)
            Y = self._action_features_for_Y(labels)

            # idea taken from sklearn's stratify split
            if labels.ndim == 2:
                # for multi-label y, map each distinct row to a string repr
                # using join because str(row) uses an ellipsis if len(row) > 1000
                labels = np.array([" ".join(row.astype("str")) for row in labels])
        else:
            # prediction time
            labels = None
            Y = None

        return SessionData(X=data_X, Y=Y, labels=labels)

    # noinspection PyPep8Naming
    def _train_val_split(
        self, session_data: "SessionData"
    ) -> Tuple["SessionData", "SessionData"]:
        """Create random hold out validation set using stratified split."""

        label_counts = dict(
            zip(*np.unique(session_data.labels, return_counts=True, axis=0))
        )
        counts = np.array([label_counts[label] for label in session_data.labels])

        multi_X = session_data.X[counts > 1]
        multi_Y = session_data.Y[counts > 1]
        multi_labels = session_data.labels[counts > 1]

        solo_X = session_data.X[counts == 1]
        solo_Y = session_data.Y[counts == 1]
        solo_labels = session_data.labels[counts == 1]

        (X_train, X_val, Y_train, Y_val, labels_train, labels_val) = train_test_split(
            multi_X,
            multi_Y,
            multi_labels,
            test_size=self.evaluate_on_num_examples,
            random_state=self.random_seed,
            stratify=multi_labels,
        )
        X_train = np.concatenate([X_train, solo_X])
        Y_train = np.concatenate([Y_train, solo_Y])
        labels_train = np.concatenate([labels_train, solo_labels])

        return (
            SessionData(X=X_train, Y=Y_train, labels=labels_train),
            SessionData(X=X_val, Y=Y_val, labels=labels_val),
        )

    @staticmethod
    def _shuffle_session_data(session_data: "SessionData") -> "SessionData":
        """Shuffle session data."""

        ids = np.random.permutation(len(session_data.X))
        return SessionData(
            X=session_data.X[ids],
            Y=session_data.Y[ids],
            labels=session_data.labels[ids],
        )

    # tf helpers:
    # noinspection PyPep8Naming
    def _gen_batch(
        self,
        session_data: "SessionData",
        batch_size: int,
        batch_strategy: Text = "sequence",
        shuffle: bool = False,
    ) -> Generator[Tuple["np.ndarray", "np.ndarray"], None, None]:
        """Generate batches."""

        if shuffle:
            session_data = self._shuffle_session_data(session_data)

        if batch_strategy == "balanced":
            num_examples = len(session_data.X)
            unique_labels, counts_labels = np.unique(
                session_data.labels, return_counts=True, axis=0
            )
            num_labels = len(unique_labels)

            label_data = []
            for label in unique_labels:
                label_data.append(
                    SessionData(
                        X=session_data.X[session_data.labels == label],
                        Y=session_data.Y[session_data.labels == label],
                        labels=None,  # ignore new labels
                    )
                )

            data_idx = [0] * num_labels
            num_data_cycles = [0] * num_labels
            skipped = [False] * num_labels
            new_X = []
            new_Y = []
            while min(num_data_cycles) == 0:
                if shuffle:
                    ids = np.random.permutation(num_labels)
                else:
                    ids = range(num_labels)

                for i in ids:
                    if num_data_cycles[i] > 0 and not skipped[i]:
                        skipped[i] = True
                        continue
                    else:
                        skipped[i] = False

                    num_i = int(counts_labels[i] / num_examples * batch_size) + 1

                    new_X.append(label_data[i].X[data_idx[i] : data_idx[i] + num_i])
                    new_Y.append(label_data[i].Y[data_idx[i] : data_idx[i] + num_i])

                    data_idx[i] += num_i
                    if data_idx[i] >= counts_labels[i]:
                        num_data_cycles[i] += 1
                        data_idx[i] = 0

                    if min(num_data_cycles) > 0:
                        break

            session_data = SessionData(
                X=np.concatenate(new_X), Y=np.concatenate(new_Y), labels=None
            )  # ignore new labels

        num_batches = session_data.X.shape[0] // batch_size + int(
            session_data.X.shape[0] % batch_size > 0
        )

        for batch_num in range(num_batches):
            batch_x = session_data.X[
                batch_num * batch_size : (batch_num + 1) * batch_size
            ]
            batch_y = session_data.Y[
                batch_num * batch_size : (batch_num + 1) * batch_size
            ]

            yield batch_x, batch_y

    # noinspection PyPep8Naming
    def _create_tf_dataset(
        self,
        session_data: "SessionData",
        batch_size: Union["tf.Tensor", int],
        batch_strategy: Text = "sequence",
        shuffle: bool = False,
    ) -> "tf.data.Dataset":
        """Create tf dataset."""

        # set batch and sequence length to None
        shape_X = (None, None, session_data.X[0].shape[-1])

        if session_data.Y[0].ndim == 1:
            shape_Y = (None, session_data.Y[0].shape[-1])
        else:
            shape_Y = (None, None, session_data.Y[0].shape[-1])

        return tf.data.Dataset.from_generator(
            lambda batch_size_: self._gen_batch(
                session_data, batch_size_, batch_strategy, shuffle
            ),
            output_types=(tf.float32, tf.float32),
            output_shapes=(shape_X, shape_Y),
            args=([batch_size]),
        )

    @staticmethod
    def _create_tf_iterator(dataset: "tf.data.Dataset") -> "tf.data.Iterator":
        """Create tf iterator."""

        return tf.data.Iterator.from_structure(
            dataset.output_types,
            dataset.output_shapes,
            output_classes=dataset.output_classes,
        )

    def _create_tf_nn(
        self,
        x_in: "tf.Tensor",
        layer_sizes: List[int],
        droprate: float,
        layer_name_suffix: Text,
        activation: Optional[Callable] = tf.nn.relu,
        use_bias: bool = True,
        kernel_initializer: Optional["tf.keras.initializers.Initializer"] = None,
    ) -> "tf.Tensor":
        """Create nn with hidden layers and name suffix."""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
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
            x = tf.layers.dropout(x, rate=droprate, training=self._is_training)
        return x

    def _tf_normalize_if_cosine(self, x: "tf.Tensor") -> "tf.Tensor":
        """Normalize embedding if similarity type is cosine."""

        if self.similarity_type == "cosine":
            return tf.nn.l2_normalize(x, -1)
        elif self.similarity_type == "inner":
            return x
        else:
            raise ValueError(
                "Wrong similarity type '{}', "
                "should be 'cosine' or 'inner'"
                "".format(self.similarity_type)
            )

    def _create_tf_embed(self, x: "tf.Tensor", layer_name_suffix: Text) -> "tf.Tensor":
        """Create dense embedding layer with a name."""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        embed_x = tf.layers.dense(
            inputs=x,
            units=self.embed_dim,
            activation=None,
            kernel_regularizer=reg,
            name="embed_layer_{}".format(layer_name_suffix),
            reuse=tf.AUTO_REUSE,
        )
        # normalize embedding vectors for cosine similarity
        return self._tf_normalize_if_cosine(embed_x)

    def _create_tf_bot_embed(self, b_in: "tf.Tensor") -> "tf.Tensor":
        """Create embedding bot vector."""

        b = self._create_tf_nn(
            b_in,
            self.hidden_layers_sizes["b"],
            self.droprate["bot"],
            layer_name_suffix="bot",
        )
        return self._create_tf_embed(b, layer_name_suffix="bot")

    def _create_t2t_hparams(self) -> "HParams":
        """Create parameters for t2t transformer."""

        hparams = transformer_base()

        hparams.num_hidden_layers = self.num_transformer_layers
        hparams.hidden_size = self.transformer_size
        # it seems to be factor of 4 for transformer architectures in t2t
        hparams.filter_size = hparams.hidden_size * 4
        hparams.num_heads = self.num_heads
        hparams.relu_dropout = self.droprate["dial"]
        hparams.pos = self.pos_encoding

        hparams.max_length = self.max_seq_length

        hparams.unidirectional_encoder = True

        hparams.self_attention_type = "dot_product_relative_v2"
        hparams.max_relative_position = 5
        hparams.add_relative_to_values = True

        return hparams

    # noinspection PyUnresolvedReferences
    def _create_t2t_transformer_encoder(
        self,
        x_in: "tf.Tensor",
        mask: "tf.Tensor",
        attention_weights: Dict[Text, "tf.Tensor"],
    ) -> "tf.Tensor":
        """Create t2t transformer encoder."""

        hparams = self._create_t2t_hparams()

        # When not in training mode, set all forms of dropout to zero.
        for key, value in hparams.values().items():
            if key.endswith("dropout") or key == "label_smoothing":
                setattr(hparams, key, value * tf.cast(self._is_training, tf.float32))

        with tf.variable_scope("transformer", reuse=tf.AUTO_REUSE):
            x = self._create_tf_nn(
                x_in,
                [hparams.hidden_size],
                hparams.layer_prepostprocess_dropout,
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

            return tf.nn.dropout(
                tf.nn.relu(x), 1.0 - hparams.layer_prepostprocess_dropout
            )

    def _create_tf_dial(self, a_in) -> Tuple["tf.Tensor", "tf.Tensor"]:
        """Create dialogue level embedding and mask."""

        # mask different length sequences
        # if there is at least one `-1` it should be masked
        mask = tf.sign(tf.reduce_max(self.a_in, -1) + 1)

        a = self._create_tf_nn(
            a_in,
            self.hidden_layers_sizes["a"],
            self.droprate["dial"],
            layer_name_suffix="dial",
        )

        self.attention_weights = {}
        a = self._create_t2t_transformer_encoder(a, mask, self.attention_weights)

        dial_embed = self._create_tf_embed(a, layer_name_suffix="dial")

        if isinstance(self.featurizer, MaxHistoryTrackerFeaturizer):
            # pick last action if max history featurizer is used
            dial_embed = dial_embed[:, -1:, :]
            mask = mask[:, -1:]

        return dial_embed, mask

    @staticmethod
    def _tf_make_flat(x: "tf.Tensor") -> "tf.Tensor":
        """Make tensor 2D."""

        return tf.reshape(x, (-1, x.shape[-1]))

    @staticmethod
    def _tf_sample_neg(
        batch_size: "tf.Tensor", all_bs: "tf.Tensor", neg_ids: "tf.Tensor"
    ) -> "tf.Tensor":
        """Sample negative examples for given indices"""

        tiled_all_bs = tf.tile(tf.expand_dims(all_bs, 0), (batch_size, 1, 1))

        return tf.batch_gather(tiled_all_bs, neg_ids)

    def _tf_calc_iou_mask(
        self, pos_b: "tf.Tensor", all_bs: "tf.Tensor", neg_ids: "tf.Tensor"
    ) -> "tf.Tensor":
        """Calculate IOU mask for given indices"""

        pos_b_in_flat = tf.expand_dims(pos_b, -2)
        neg_b_in_flat = self._tf_sample_neg(tf.shape(pos_b)[0], all_bs, neg_ids)

        intersection_b_in_flat = tf.minimum(neg_b_in_flat, pos_b_in_flat)
        union_b_in_flat = tf.maximum(neg_b_in_flat, pos_b_in_flat)

        iou = tf.reduce_sum(intersection_b_in_flat, -1) / tf.reduce_sum(
            union_b_in_flat, -1
        )
        return 1.0 - tf.nn.relu(tf.sign(1.0 - iou))

    def _tf_get_negs(
        self, all_embed: "tf.Tensor", all_raw: "tf.Tensor", raw_pos: "tf.Tensor"
    ) -> Tuple["tf.Tensor", "tf.Tensor"]:
        """Get negative examples from given tensor."""

        batch_size = tf.shape(raw_pos)[0]
        seq_length = tf.shape(raw_pos)[1]
        raw_flat = self._tf_make_flat(raw_pos)

        total_candidates = tf.shape(all_embed)[0]

        all_indices = tf.tile(
            tf.expand_dims(tf.range(0, total_candidates, 1), 0),
            (batch_size * seq_length, 1),
        )
        shuffled_indices = tf.transpose(
            tf.random.shuffle(tf.transpose(all_indices, (1, 0))), (1, 0)
        )
        neg_ids = shuffled_indices[:, : self.num_neg]

        bad_negs_flat = self._tf_calc_iou_mask(raw_flat, all_raw, neg_ids)
        bad_negs = tf.reshape(bad_negs_flat, (batch_size, seq_length, -1))

        neg_embed_flat = self._tf_sample_neg(
            batch_size * seq_length, all_embed, neg_ids
        )
        neg_embed = tf.reshape(
            neg_embed_flat, (batch_size, seq_length, -1, all_embed.shape[-1])
        )

        return neg_embed, bad_negs

    def _sample_negatives(
        self, all_actions: "tf.Tensor"
    ) -> Tuple[
        "tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor"
    ]:
        """Sample negative examples."""

        pos_dial_embed = tf.expand_dims(self.dial_embed, -2)
        neg_dial_embed, dial_bad_negs = self._tf_get_negs(
            self._tf_make_flat(self.dial_embed),
            self._tf_make_flat(self.b_in),
            self.b_in,
        )
        pos_bot_embed = tf.expand_dims(self.bot_embed, -2)
        neg_bot_embed, bot_bad_negs = self._tf_get_negs(
            self.all_bot_embed, all_actions, self.b_in
        )
        return (
            pos_dial_embed,
            pos_bot_embed,
            neg_dial_embed,
            neg_bot_embed,
            dial_bad_negs,
            bot_bad_negs,
        )

    @staticmethod
    def _tf_raw_sim(a: "tf.Tensor", b: "tf.Tensor", mask: "tf.Tensor") -> "tf.Tensor":
        """Calculate similarity between given tensors."""

        return tf.reduce_sum(a * b, -1) * tf.expand_dims(mask, 2)

    def _tf_sim(
        self,
        pos_dial_embed: "tf.Tensor",
        pos_bot_embed: "tf.Tensor",
        neg_dial_embed: "tf.Tensor",
        neg_bot_embed: "tf.Tensor",
        dial_bad_negs: "tf.Tensor",
        bot_bad_negs: "tf.Tensor",
        mask: "tf.Tensor",
    ) -> Tuple["tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor"]:
        """Define similarity."""

        # calculate similarity with several
        # embedded actions for the loss
        neg_inf = large_compatible_negative(pos_dial_embed.dtype)

        sim_pos = self._tf_raw_sim(pos_dial_embed, pos_bot_embed, mask)
        sim_neg = (
            self._tf_raw_sim(pos_dial_embed, neg_bot_embed, mask)
            + neg_inf * bot_bad_negs
        )
        sim_neg_bot_bot = (
            self._tf_raw_sim(pos_bot_embed, neg_bot_embed, mask)
            + neg_inf * bot_bad_negs
        )
        sim_neg_dial_dial = (
            self._tf_raw_sim(pos_dial_embed, neg_dial_embed, mask)
            + neg_inf * dial_bad_negs
        )
        sim_neg_bot_dial = (
            self._tf_raw_sim(pos_bot_embed, neg_dial_embed, mask)
            + neg_inf * dial_bad_negs
        )

        # output similarities between user input and bot actions
        # and similarities between bot actions and similarities between user inputs
        return sim_pos, sim_neg, sim_neg_bot_bot, sim_neg_dial_dial, sim_neg_bot_dial

    @staticmethod
    def _tf_calc_accuracy(sim_pos: "tf.Tensor", sim_neg: "tf.Tensor") -> "tf.Tensor":
        """Calculate accuracy"""

        max_all_sim = tf.reduce_max(tf.concat([sim_pos, sim_neg], -1), -1)
        return tf.reduce_mean(
            tf.cast(tf.math.equal(max_all_sim, sim_pos[:, :, 0]), tf.float32)
        )

    def _tf_loss_margin(
        self,
        sim_pos: "tf.Tensor",
        sim_neg: "tf.Tensor",
        sim_neg_bot_bot: "tf.Tensor",
        sim_neg_dial_dial: "tf.Tensor",
        sim_neg_bot_dial: "tf.Tensor",
        mask: "tf.Tensor",
    ) -> "tf.Tensor":
        """Define max margin loss."""

        # loss for maximizing similarity with correct action
        loss = tf.maximum(0.0, self.mu_pos - sim_pos[:, :, 0])

        # loss for minimizing similarity with `num_neg` incorrect actions
        if self.use_max_sim_neg:
            # minimize only maximum similarity over incorrect actions
            max_sim_neg = tf.reduce_max(sim_neg, -1)
            loss += tf.maximum(0.0, self.mu_neg + max_sim_neg)
        else:
            # minimize all similarities with incorrect actions
            max_margin = tf.maximum(0.0, self.mu_neg + sim_neg)
            loss += tf.reduce_sum(max_margin, -1)

        # penalize max similarity between pos bot and neg bot embeddings
        max_sim_neg_bot = tf.maximum(0.0, tf.reduce_max(sim_neg_bot_bot, -1))
        loss += max_sim_neg_bot * self.C_emb

        # penalize max similarity between pos dial and neg dial embeddings
        max_sim_neg_dial = tf.maximum(0.0, tf.reduce_max(sim_neg_dial_dial, -1))
        loss += max_sim_neg_dial * self.C_emb

        # penalize max similarity between pos bot and neg dial embeddings
        max_sim_neg_dial = tf.maximum(0.0, tf.reduce_max(sim_neg_bot_dial, -1))
        loss += max_sim_neg_dial * self.C_emb

        # mask loss for different length sequences
        loss *= mask
        # average the loss over sequence length
        loss = tf.reduce_sum(loss, -1) / tf.reduce_sum(mask, 1)
        # average the loss over the batch
        loss = tf.reduce_mean(loss)

        # add regularization losses
        loss += tf.losses.get_regularization_loss()

        return loss

    @staticmethod
    def _tf_loss_softmax(
        sim_pos: "tf.Tensor",
        sim_neg: "tf.Tensor",
        sim_neg_bot_bot: "tf.Tensor",
        sim_neg_dial_dial: "tf.Tensor",
        sim_neg_bot_dial: "tf.Tensor",
        mask: "tf.Tensor",
    ) -> "tf.Tensor":
        """Define softmax loss."""

        logits = tf.concat(
            [sim_pos, sim_neg, sim_neg_bot_bot, sim_neg_dial_dial, sim_neg_bot_dial], -1
        )

        # create labels for softmax
        pos_labels = tf.ones_like(logits[:, :, :1])
        neg_labels = tf.zeros_like(logits[:, :, 1:])
        labels = tf.concat([pos_labels, neg_labels], -1)

        # mask loss by prediction confidence
        pred = tf.nn.softmax(logits)
        already_learned = tf.pow((1 - pred[:, :, 0]) / 0.5, 4)

        loss = tf.losses.softmax_cross_entropy(labels, logits, mask * already_learned)
        # add regularization losses
        loss += tf.losses.get_regularization_loss()

        return loss

    def _choose_loss(
        self,
        sim_pos: "tf.Tensor",
        sim_neg: "tf.Tensor",
        sim_neg_bot_bot: "tf.Tensor",
        sim_neg_dial_dial: "tf.Tensor",
        sim_neg_bot_dial: "tf.Tensor",
        mask: "tf.Tensor",
    ) -> "tf.Tensor":
        """Use loss depending on given option."""

        if self.loss_type == "margin":
            return self._tf_loss_margin(
                sim_pos,
                sim_neg,
                sim_neg_bot_bot,
                sim_neg_dial_dial,
                sim_neg_bot_dial,
                mask,
            )
        elif self.loss_type == "softmax":
            return self._tf_loss_softmax(
                sim_pos,
                sim_neg,
                sim_neg_bot_bot,
                sim_neg_dial_dial,
                sim_neg_bot_dial,
                mask,
            )
        else:
            raise ValueError(
                "Wrong loss type '{}', "
                "should be 'margin' or 'softmax'"
                "".format(self.loss_type)
            )

    def _build_tf_train_graph(self) -> Tuple["tf.Tensor", "tf.Tensor"]:
        """Bulid train graph using iterator."""

        # session data are int counts but we need a float tensors
        self.a_in, self.b_in = self._iterator.get_next()

        all_actions = tf.constant(
            self._encoded_all_actions, dtype=tf.float32, name="all_actions"
        )

        self.dial_embed, mask = self._create_tf_dial(self.a_in)

        self.bot_embed = self._create_tf_bot_embed(self.b_in)
        self.all_bot_embed = self._create_tf_bot_embed(all_actions)

        if isinstance(self.featurizer, MaxHistoryTrackerFeaturizer):
            # add time dimension if max history featurizer is used
            self.b_in = self.b_in[:, tf.newaxis, :]
            self.bot_embed = self.bot_embed[:, tf.newaxis, :]

        (
            pos_dial_embed,
            pos_bot_embed,
            neg_dial_embed,
            neg_bot_embed,
            dial_bad_negs,
            bot_bad_negs,
        ) = self._sample_negatives(all_actions)

        # calculate similarities
        (
            sim_pos,
            sim_neg,
            sim_neg_bot_bot,
            sim_neg_dial_dial,
            sim_neg_bot_dial,
        ) = self._tf_sim(
            pos_dial_embed,
            pos_bot_embed,
            neg_dial_embed,
            neg_bot_embed,
            dial_bad_negs,
            bot_bad_negs,
            mask,
        )

        acc = self._tf_calc_accuracy(sim_pos, sim_neg)

        loss = self._choose_loss(
            sim_pos, sim_neg, sim_neg_bot_bot, sim_neg_dial_dial, sim_neg_bot_dial, mask
        )
        return loss, acc

    # training helpers
    def _linearly_increasing_batch_size(self, epoch: int) -> int:
        """Linearly increase batch size with every epoch.

        The idea comes from https://arxiv.org/abs/1711.00489.
        """

        if not isinstance(self.batch_size, list):
            return int(self.batch_size)

        if self.epochs > 1:
            return int(
                self.batch_size[0]
                + epoch * (self.batch_size[1] - self.batch_size[0]) / (self.epochs - 1)
            )
        else:
            return int(self.batch_size[0])

    def _train_tf_dataset(
        self,
        train_init_op: "tf.Operation",
        eval_init_op: "tf.Operation",
        batch_size_in: "tf.Tensor",
        loss: "tf.Tensor",
        acc: "tf.Tensor",
    ) -> None:
        """Train tf graph"""

        self.session.run(tf.global_variables_initializer())

        if self.evaluate_on_num_examples:
            logger.info(
                "Validation accuracy is calculated every {} epochs"
                "".format(self.evaluate_every_num_epochs)
            )
        pbar = tqdm(range(self.epochs), desc="Epochs", disable=is_logging_disabled())

        train_loss = 0
        train_acc = 0
        eval_loss = 0
        eval_acc = 0
        for ep in pbar:

            batch_size = self._linearly_increasing_batch_size(ep)

            self.session.run(train_init_op, feed_dict={batch_size_in: batch_size})

            ep_train_loss = 0
            ep_train_acc = 0
            batches_per_epoch = 0
            while True:
                try:
                    _, batch_train_loss, batch_train_acc = self.session.run(
                        [self._train_op, loss, acc], feed_dict={self._is_training: True}
                    )
                    batches_per_epoch += 1
                    ep_train_loss += batch_train_loss
                    ep_train_acc += batch_train_acc

                except tf.errors.OutOfRangeError:
                    break

            train_loss = ep_train_loss / batches_per_epoch
            train_acc = ep_train_acc / batches_per_epoch

            pbar.set_postfix(
                {"loss": "{:.3f}".format(train_loss), "acc": "{:.3f}".format(train_acc)}
            )

            if eval_init_op is not None:
                if (ep + 1) % self.evaluate_every_num_epochs == 0 or (
                    ep + 1
                ) == self.epochs:
                    eval_loss, eval_acc = self._output_training_stat_dataset(
                        eval_init_op, loss, acc
                    )
                    if (ep + 1) != self.epochs:
                        logger.info(
                            "Evaluation results: "
                            "validation loss: {:.3f}, "
                            "validation accuracy: {:.3f}"
                            "".format(eval_loss, eval_acc)
                        )

        final_message = (
            "Finished training embedding policy, "
            "train loss={:.3f}, train accuracy={:.3f}"
            "".format(train_loss, train_acc)
        )
        if eval_init_op is not None:
            final_message += (
                ", validation loss={:.3f}, validation accuracy={:.3f}"
                "".format(eval_loss, eval_acc)
            )
        logger.info(final_message)

    def _output_training_stat_dataset(
        self, eval_init_op: "tf.Operation", loss: "tf.Tensor", acc: "tf.Tensor"
    ) -> Tuple[float, float]:
        """Output training statistics"""

        self.session.run(eval_init_op)
        ep_val_loss = 0
        ep_val_acc = 0
        batches_per_epoch = 0
        while True:
            try:
                batch_val_loss, batch_val_acc = self.session.run(
                    [loss, acc], feed_dict={self._is_training: False}
                )
                batches_per_epoch += 1
                ep_val_loss += batch_val_loss
                ep_val_acc += batch_val_acc
            except tf.errors.OutOfRangeError:
                break

        return ep_val_loss / batches_per_epoch, ep_val_acc / batches_per_epoch

    # prepare for prediction
    def _create_tf_placeholders(self, session_data: "SessionData") -> None:
        """Create placeholders for prediction."""

        dialogue_len = None  # use dynamic time
        self.a_in = tf.placeholder(
            dtype=tf.float32,
            shape=(None, dialogue_len, session_data.X.shape[-1]),
            name="a",
        )
        self.b_in = tf.placeholder(
            dtype=tf.float32,
            shape=(None, dialogue_len, None, session_data.Y.shape[-1]),
            name="b",
        )

    def _build_tf_pred_graph(self, session_data: "SessionData") -> "tf.Tensor":
        """Rebuild tf graph for prediction."""

        self._create_tf_placeholders(session_data)

        self.dial_embed, mask = self._create_tf_dial(self.a_in)

        self.sim_all = self._tf_raw_sim(
            self.dial_embed[:, :, tf.newaxis, :],
            self.all_bot_embed[tf.newaxis, tf.newaxis, :, :],
            mask,
        )

        if self.similarity_type == "cosine":
            # clip negative values to zero
            confidence = tf.nn.relu(self.sim_all)
        else:
            # normalize result to [0, 1] with softmax
            confidence = tf.nn.softmax(self.sim_all)

        self.bot_embed = self._create_tf_bot_embed(self.b_in)

        self.sim = self._tf_raw_sim(
            self.dial_embed[:, :, tf.newaxis, :], self.bot_embed, mask
        )

        return confidence

    def _extract_attention(self) -> Optional["tf.Tensor"]:
        """Extract attention probabilities from t2t dict"""

        attention = [
            tf.expand_dims(t, 0)
            for name, t in self.attention_weights.items()
            if name.endswith("multihead_attention/dot_product_attention")
        ]

        if attention:
            return tf.concat(attention, 0)
        else:
            return

    # training methods
    def train(
        self,
        training_trackers: List["DialogueStateTracker"],
        domain: "Domain",
        **kwargs: Any
    ) -> None:
        """Train the policy on given training trackers."""

        logger.debug("Started training embedding policy.")

        # set numpy random seed
        np.random.seed(self.random_seed)

        # dealing with training data
        training_data = self.featurize_for_training(training_trackers, domain, **kwargs)

        # encode all actions with policies' featurizer
        self._encoded_all_actions = self.featurizer.state_featurizer.create_encoded_all_actions(
            domain
        )

        # check if number of negatives is less than number of actions
        logger.debug(
            "Check if num_neg {} is smaller "
            "than number of actions {}, "
            "else set num_neg to the number of actions - 1"
            "".format(self.num_neg, domain.num_actions)
        )
        # noinspection PyAttributeOutsideInit
        self.num_neg = min(self.num_neg, domain.num_actions - 1)

        # extract actual training data to feed to tf session
        session_data = self._create_session_data(training_data.X, training_data.y)

        if self.evaluate_on_num_examples:
            session_data, eval_session_data = self._train_val_split(session_data)
        else:
            eval_session_data = None

        self.graph = tf.Graph()

        with self.graph.as_default():
            # set random seed in tf
            tf.set_random_seed(self.random_seed)

            # allows increasing batch size
            batch_size_in = tf.placeholder(tf.int64)
            train_dataset = self._create_tf_dataset(
                session_data,
                batch_size_in,
                batch_strategy=self.batch_strategy,
                shuffle=True,
            )

            self._iterator = self._create_tf_iterator(train_dataset)

            train_init_op = self._iterator.make_initializer(train_dataset)

            if eval_session_data is not None:
                eval_init_op = self._iterator.make_initializer(
                    self._create_tf_dataset(
                        eval_session_data,
                        # pick maximum batch_size for eval
                        self._linearly_increasing_batch_size(self.epochs),
                    )
                )
            else:
                eval_init_op = None

            self._is_training = tf.placeholder_with_default(False, shape=())
            loss, acc = self._build_tf_train_graph()

            # define which optimizer to use
            self._train_op = tf.train.AdamOptimizer().minimize(loss)

            # train tensorflow graph
            self.session = tf.Session(config=self._tf_config)
            self._train_tf_dataset(
                train_init_op, eval_init_op, batch_size_in, loss, acc
            )

            # rebuild the graph for prediction
            self.pred_confidence = self._build_tf_pred_graph(session_data)

            self.attention_weights = self._extract_attention()

    def continue_training(
        self,
        training_trackers: List["DialogueStateTracker"],
        domain: "Domain",
        **kwargs: Any
    ) -> None:
        """Continue training an already trained policy."""

        batch_size = kwargs.get("batch_size", 5)
        epochs = kwargs.get("epochs", 50)

        with self.graph.as_default():
            for _ in range(epochs):
                training_data = self._training_data_for_continue_training(
                    batch_size, training_trackers, domain
                )

                session_data = self._create_session_data(
                    training_data.X, training_data.y
                )
                train_dataset = self._create_tf_dataset(session_data, batch_size)
                train_init_op = self._iterator.make_initializer(train_dataset)
                self.session.run(train_init_op)

                # fit to one extra example using updated trackers
                while True:
                    try:
                        self.session.run(
                            self._train_op, feed_dict={self._is_training: True}
                        )

                    except tf.errors.OutOfRangeError:
                        break

    def tf_feed_dict_for_prediction(
        self, tracker: "DialogueStateTracker", domain: "Domain"
    ) -> Dict["tf.Tensor", "np.ndarray"]:
        """Create feed dictionary for tf session."""

        # noinspection PyPep8Naming
        data_X = self.featurizer.create_X([tracker], domain)
        session_data = self._create_session_data(data_X)

        return {self.a_in: session_data.X}

    def predict_action_probabilities(
        self, tracker: "DialogueStateTracker", domain: "Domain"
    ) -> List[float]:
        """Predict the next action the bot should take.

        Return the list of probabilities for the next actions.
        """

        if self.session is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
            return [0.0] * domain.num_actions

        tf_feed_dict = self.tf_feed_dict_for_prediction(tracker, domain)

        confidence = self.session.run(self.pred_confidence, feed_dict=tf_feed_dict)

        return confidence[0, -1, :].tolist()

    def _persist_tensor(self, name: Text, tensor: "tf.Tensor") -> None:
        """Add tensor to collection if it is not None"""

        if tensor is not None:
            self.graph.clear_collection(name)
            self.graph.add_to_collection(name, tensor)

    def persist(self, path: Text) -> None:
        """Persists the policy to a storage."""

        if self.session is None:
            warnings.warn(
                "Method `persist(...)` was called "
                "without a trained model present. "
                "Nothing to persist then!"
            )
            return

        self.featurizer.persist(path)

        meta = {"priority": self.priority}

        meta_file = os.path.join(path, "embedding_policy.json")
        utils.dump_obj_as_json_to_file(meta_file, meta)

        file_name = "tensorflow_embedding.ckpt"
        checkpoint = os.path.join(path, file_name)
        rasa.utils.io.create_directory_for_file(checkpoint)

        with self.graph.as_default():
            self._persist_tensor("user_placeholder", self.a_in)
            self._persist_tensor("bot_placeholder", self.b_in)

            self._persist_tensor("similarity_all", self.sim_all)
            self._persist_tensor("pred_confidence", self.pred_confidence)
            self._persist_tensor("similarity", self.sim)

            self._persist_tensor("dial_embed", self.dial_embed)
            self._persist_tensor("bot_embed", self.bot_embed)
            self._persist_tensor("all_bot_embed", self.all_bot_embed)

            self._persist_tensor("attention_weights", self.attention_weights)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        tf_config_file = os.path.join(path, file_name + ".tf_config.pkl")
        with open(tf_config_file, "wb") as f:
            pickle.dump(self._tf_config, f)

    @staticmethod
    def load_tensor(name: Text) -> Optional["tf.Tensor"]:
        """Load tensor or set it to None"""

        tensor_list = tf.get_collection(name)
        return tensor_list[0] if tensor_list else None

    @classmethod
    def load(cls, path: Text) -> "EmbeddingPolicy":
        """Loads a policy from the storage.

        **Needs to load its featurizer**
        """

        if not os.path.exists(path):
            raise Exception(
                "Failed to load dialogue model. Path '{}' "
                "doesn't exist".format(os.path.abspath(path))
            )

        featurizer = TrackerFeaturizer.load(path)

        file_name = "tensorflow_embedding.ckpt"
        checkpoint = os.path.join(path, file_name)

        if not os.path.exists(checkpoint + ".meta"):
            return cls(featurizer=featurizer)

        meta_file = os.path.join(path, "embedding_policy.json")
        meta = json.loads(rasa.utils.io.read_file(meta_file))

        tf_config_file = os.path.join(path, "{}.tf_config.pkl".format(file_name))

        with open(tf_config_file, "rb") as f:
            _tf_config = pickle.load(f)

        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session(config=_tf_config)
            saver = tf.train.import_meta_graph(checkpoint + ".meta")

            saver.restore(session, checkpoint)

            a_in = cls.load_tensor("user_placeholder")
            b_in = cls.load_tensor("bot_placeholder")

            sim_all = cls.load_tensor("similarity_all")
            pred_confidence = cls.load_tensor("pred_confidence")
            sim = cls.load_tensor("similarity")

            dial_embed = cls.load_tensor("dial_embed")
            bot_embed = cls.load_tensor("bot_embed")
            all_bot_embed = cls.load_tensor("all_bot_embed")

            attention_weights = cls.load_tensor("attention_weights")

        return cls(
            featurizer=featurizer,
            priority=meta["priority"],
            graph=graph,
            session=session,
            user_placeholder=a_in,
            bot_placeholder=b_in,
            similarity_all=sim_all,
            pred_confidence=pred_confidence,
            similarity=sim,
            dial_embed=dial_embed,
            bot_embed=bot_embed,
            all_bot_embed=all_bot_embed,
            attention_weights=attention_weights,
        )
