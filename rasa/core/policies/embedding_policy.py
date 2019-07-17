from collections import namedtuple
import copy
import json
import logging
import os
import warnings

import numpy as np
import typing
from tqdm import tqdm
from typing import Any, List, Optional, Text, Dict, Tuple, Union

import rasa.utils.io
from rasa.core import utils
from rasa.core.domain import Domain
from rasa.core.featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    LabelTokenizerSingleStateFeaturizer,
    MaxHistoryTrackerFeaturizer
)
from rasa.core.policies.policy import Policy
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.common import is_logging_disabled

import tensorflow as tf

try:
    from tensor2tensor.layers import common_attention
    from tensor2tensor.models.transformer import transformer_base, transformer_prepare_encoder, transformer_encoder
except ImportError:
    common_attention = None
    transformer_base = None
    transformer_prepare_encoder = None
    transformer_encoder = None

try:
    import cPickle as pickle
except ImportError:
    import pickle


logger = logging.getLogger(__name__)

# namedtuple for all tf session related data
SessionData = namedtuple(
    "SessionData",
    (
        "X",
        "Y",
        "slots",
        "previous_actions",
        "actions_for_Y",
    ),
)


class EmbeddingPolicy(Policy):
    """Recurrent Embedding Dialogue Policy (REDP)

    Transformer version of the policy used in our paper https://arxiv.org/abs/1811.11707
    """

    SUPPORTS_ONLINE_TRAINING = True

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # nn architecture
        # a list of hidden layers sizes before user embed layer
        # number of hidden layers is equal to the length of this list
        "hidden_layers_sizes_a": [],
        # a list of hidden layers sizes before bot embed layer
        # number of hidden layers is equal to the length of this list
        "hidden_layers_sizes_b": [],

        "pos_encoding": "timing",  # {"timing", "emb", "custom_timing"}
        # introduce phase shift in time encodings between transformers
        # 0.5 - 0.8 works on small dataset
        "pos_max_timescale": 1.0e1,
        "max_seq_length": 256,
        "num_heads": 4,
        # number of units in rnn cell
        "rnn_size": 128,
        "num_rnn_layers": 1,
        # training parameters
        # flag if to turn on layer normalization for lstm cell
        "layer_norm": True,
        # initial and final batch sizes - batch size will be
        # linearly increased for each epoch
        "batch_size": [8, 32],
        # number of epochs
        "epochs": 1,
        # set random seed to any int to get reproducible results
        "random_seed": None,
        # embedding parameters
        # dimension size of embedding vectors
        "embed_dim": 20,
        # how similar the algorithm should try
        # to make embedding vectors for correct actions
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect actions
        "mu_neg": -0.2,  # should be -1.0 < ... < 1.0 for 'cosine'
        # the type of the similarity
        "similarity_type": "auto",  # string 'auto' or 'cosine' or 'inner'
        "loss_type": 'softmax',  # string 'softmax' or 'margin'
        # the number of incorrect actions, the algorithm will minimize
        # their similarity to the user input during training
        "num_neg": 20,
        # flag if minimize only maximum similarity over incorrect actions
        "use_max_sim_neg": True,  # flag which loss function to use
        # regularization
        # the scale of L2 regularization
        "C2": 0.001,
        # the scale of how important is to minimize the maximum similarity
        # between embeddings of different actions
        "C_emb": 0.8,
        # dropout rate for user nn
        "droprate_a": 0.0,
        # dropout rate for bot nn
        "droprate_b": 0.0,
        # dropout rate for rnn
        "droprate_rnn": 0.1,
        # attention parameters
        # flag to use attention over user input
        # as an input to rnn
        "attn_before_rnn": True,
        # flag to use attention over prev bot actions
        # and copy it to output bypassing rnn
        "attn_after_rnn": True,
        # flag to use `sparsemax` instead of `softmax` for attention
        "sparse_attention": False,  # flag to use sparsemax for probs
        # the range of allowed location-based attention shifts
        "attn_shift_range": None,  # if None, set to mean dialogue length / 2
        # visualization of accuracy
        # how often calculate train accuracy
        "evaluate_every_num_epochs": 20,  # small values may hurt performance
        # how many examples to use for calculation of train accuracy
        "evaluate_on_num_examples": 100,  # large values may hurt performance
    }

    # end default properties (DOC MARKER - don't remove)

    @staticmethod
    def _standard_featurizer(max_history=None):
        if max_history is None:
            return FullDialogueTrackerFeaturizer(LabelTokenizerSingleStateFeaturizer())
        else:
            return MaxHistoryTrackerFeaturizer(LabelTokenizerSingleStateFeaturizer(), max_history=max_history)

    @staticmethod
    def _check_t2t():
        if common_attention is None:
            raise ImportError("Please install tensor2tensor")

    def __init__(
        self,
        featurizer: Optional['FullDialogueTrackerFeaturizer'] = None,
        priority: int = 1,
        encoded_all_actions: Optional['np.ndarray'] = None,
        graph: Optional['tf.Graph'] = None,
        session: Optional['tf.Session'] = None,
        intent_placeholder: Optional['tf.Tensor'] = None,
        action_placeholder: Optional['tf.Tensor'] = None,
        slots_placeholder: Optional['tf.Tensor'] = None,
        prev_act_placeholder: Optional['tf.Tensor'] = None,
        similarity_all: Optional['tf.Tensor'] = None,
        pred_confidence: Optional['tf.Tensor'] = None,
        similarity: Optional['tf.Tensor'] = None,
        dial_embed: Optional['tf.Tensor'] = None,
        bot_embed: Optional['tf.Tensor'] = None,
        all_bot_embed: Optional['tf.Tensor'] = None,
        attention_weights=None,
        max_history: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        # check if t2t is installed
        self._check_t2t()

        if not featurizer:
            featurizer = self._standard_featurizer(max_history)
        super(EmbeddingPolicy, self).__init__(featurizer, priority)

        # flag if to use the same embeddings for user and bot
        try:
            self.share_embedding = self.featurizer.state_featurizer.use_shared_vocab
        except AttributeError:
            self.share_embedding = False

        self._load_params(**kwargs)

        # chrono initialization for forget bias
        self.characteristic_time = None

        # encode all actions with numbers
        # persist this array for prediction time
        self.encoded_all_actions = encoded_all_actions

        # tf related instances
        self.graph = graph
        self.session = session
        self.a_in = intent_placeholder
        self.b_in = action_placeholder
        self.c_in = slots_placeholder
        self.b_prev_in = prev_act_placeholder
        self.sim_all = similarity_all
        self.pred_confidence = pred_confidence
        self.sim = similarity

        # persisted embeddings
        self.dial_embed = dial_embed
        self.bot_embed = bot_embed
        self.all_bot_embed = all_bot_embed

        self.attention_weights = attention_weights
        # internal tf instances
        self._train_op = None
        self._is_training = None

    # init helpers
    def _load_nn_architecture_params(self, config: Dict[Text, Any]) -> None:
        self.hidden_layer_sizes = {
            "a": config["hidden_layers_sizes_a"],
            "b": config["hidden_layers_sizes_b"],
        }

        if self.share_embedding:
            if self.hidden_layer_sizes["a"] != self.hidden_layer_sizes["b"]:
                raise ValueError(
                    "Due to sharing vocabulary "
                    "in the featurizer, embedding weights "
                    "are shared as well. "
                    "So hidden_layers_sizes_a={} should be "
                    "equal to hidden_layers_sizes_b={}"
                    "".format(
                        self.hidden_layer_sizes["a"], self.hidden_layer_sizes["b"]
                    )
                )
        self.pos_encoding = config['pos_encoding']
        self.pos_max_timescale = config['pos_max_timescale']
        self.max_seq_length = config['max_seq_length']
        self.num_heads = config['num_heads']

        self.rnn_size = config["rnn_size"]
        self.num_rnn_layers = config["num_rnn_layers"]
        self.layer_norm = config["layer_norm"]

        self.batch_size = config["batch_size"]

        self.epochs = config["epochs"]

        self.random_seed = config["random_seed"]

    def _load_embedding_params(self, config: Dict[Text, Any]) -> None:
        self.embed_dim = config["embed_dim"]
        self.mu_pos = config["mu_pos"]
        self.mu_neg = config["mu_neg"]
        self.similarity_type = config["similarity_type"]
        self.loss_type = config['loss_type']
        if self.similarity_type == 'auto':
            if self.loss_type == 'softmax':
                self.similarity_type = 'inner'
            elif self.loss_type == 'margin':
                self.similarity_type = 'cosine'

        self.num_neg = config["num_neg"]
        self.use_max_sim_neg = config["use_max_sim_neg"]

    def _load_regularization_params(self, config: Dict[Text, Any]) -> None:
        self.C2 = config["C2"]
        self.C_emb = config["C_emb"]
        self.droprate = {
            "a": config["droprate_a"],
            "b": config["droprate_b"],
            "rnn": config["droprate_rnn"],
        }

    def _load_attn_params(self, config: Dict[Text, Any]) -> None:
        self.sparse_attention = config["sparse_attention"]
        self.attn_shift_range = config["attn_shift_range"]
        self.attn_after_rnn = config["attn_after_rnn"]
        self.attn_before_rnn = config["attn_before_rnn"]

    def is_using_attention(self):
        return self.attn_after_rnn or self.attn_before_rnn

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
        self._load_attn_params(config)
        self._load_visual_params(config)

    # data helpers
    # noinspection PyPep8Naming
    def _create_X_slots_previous_actions(
        self, data_X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract feature vectors

        for user input (X), slots and
        previously executed actions from training data.
        """

        featurizer = self.featurizer.state_featurizer
        slot_start = featurizer.user_feature_len
        previous_start = slot_start + featurizer.slot_feature_len

        X = data_X[:, :, :slot_start]
        slots = data_X[:, :, slot_start:previous_start]
        previous_actions = data_X[:, :, previous_start:]

        return X, slots, previous_actions

    # noinspection PyPep8Naming
    @staticmethod
    def _actions_for_Y(data_Y: np.ndarray) -> np.ndarray:
        """Prepare Y data for training: extract actions indices."""
        return data_Y.argmax(axis=-1)

    # noinspection PyPep8Naming
    def _action_features_for_Y(self, actions_for_Y: np.ndarray) -> np.ndarray:
        """Prepare Y data for training: features for action labels."""

        if len(actions_for_Y.shape) == 2:
            return np.stack(
                [
                    np.stack(
                        [self.encoded_all_actions[action_idx] for action_idx in action_ids]
                    )
                    for action_ids in actions_for_Y
                ]
            )
        else:
            return np.stack(
                [
                    self.encoded_all_actions[action_idx] for action_idx in actions_for_Y
                ]
            )

    # noinspection PyPep8Naming
    def _create_session_data(
        self, data_X: np.ndarray, data_Y: Optional[np.ndarray] = None
    ) -> SessionData:
        """Combine all tf session related data into a named tuple"""

        X, slots, previous_actions = self._create_X_slots_previous_actions(data_X)

        if data_Y is not None:
            # training time
            actions_for_Y = self._actions_for_Y(data_Y)
            Y = self._action_features_for_Y(actions_for_Y)
        else:
            # prediction time
            actions_for_Y = None
            Y = None

        # is needed to calculate train accuracy
        if isinstance(self.featurizer, FullDialogueTrackerFeaturizer):
            dial_len = X.shape[1]
        else:
            dial_len = 1

        return SessionData(
            X=X,
            Y=Y,
            slots=slots,
            previous_actions=previous_actions,
            actions_for_Y=actions_for_Y,
        )

    @staticmethod
    def _sample_session_data(session_data: 'SessionData',
                             num_samples: int) -> 'SessionData':
        ids = np.random.permutation(len(session_data.X))[:num_samples]
        return SessionData(
            X=session_data.X[ids],
            Y=session_data.Y[ids],
            slots=session_data.slots[ids],
            previous_actions=session_data.previous_actions[ids],
            actions_for_Y=session_data.actions_for_Y[ids],
        )

    # tf helpers:
    @staticmethod
    def _create_tf_dataset(session_data: 'SessionData',
                           batch_size: Union['tf.Tensor', int],
                           shuffle: bool = True) -> 'tf.data.Dataset':
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (session_data.X, session_data.Y,
             session_data.slots, session_data.previous_actions)
        )
        if shuffle:
            train_dataset = train_dataset.shuffle(buffer_size=len(session_data.X))
        train_dataset = train_dataset.batch(batch_size)

        return train_dataset

    @staticmethod
    def _create_tf_iterator(dataset):
        return tf.data.Iterator.from_structure(dataset.output_types,
                                               dataset.output_shapes,
                                               output_classes=dataset.output_classes)

    def _create_tf_nn(
        self,
        x_in: 'tf.Tensor',
        layer_sizes: List,
        droprate: float,
        layer_name_suffix: Text,
    ) -> 'tf.Tensor':
        """Create nn with hidden layers and name suffix."""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        x = tf.nn.relu(x_in)
        for i, layer_size in enumerate(layer_sizes):
            x = tf.layers.dense(
                inputs=x,
                units=layer_size,
                activation=tf.nn.relu,
                kernel_regularizer=reg,
                name="hidden_layer_{}_{}".format(layer_name_suffix, i),
                reuse=tf.AUTO_REUSE,
            )
            x = tf.layers.dropout(x, rate=droprate, training=self._is_training)
        return x

    def _tf_normalize_if_cosine(self, a: 'tf.Tensor') -> 'tf.Tensor':

        if self.similarity_type not in {"cosine", "inner"}:
            raise ValueError(
                "Wrong similarity type {}, "
                "should be 'cosine' or 'inner'"
                "".format(self.similarity_type)
            )

        if self.similarity_type == "cosine":
            return tf.nn.l2_normalize(a, -1)
        else:
            return a

    def _create_tf_embed(self, x: 'tf.Tensor', layer_name_suffix: Text) -> 'tf.Tensor':
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

    def _create_tf_bot_embed(self, b_in: 'tf.Tensor') -> 'tf.Tensor':
        """Create embedding bot vector."""

        layer_name_suffix = "a_and_b" if self.share_embedding else "b"

        b = self._create_tf_nn(
            b_in,
            self.hidden_layer_sizes["b"],
            self.droprate["b"],
            layer_name_suffix=layer_name_suffix,
        )
        return self._create_tf_embed(b, layer_name_suffix=layer_name_suffix)

    def _create_hparams(self):
        hparams = transformer_base()

        hparams.num_hidden_layers = self.num_rnn_layers
        hparams.hidden_size = self.rnn_size
        # it seems to be factor of 4 for transformer architectures in t2t
        hparams.filter_size = hparams.hidden_size * 4
        hparams.num_heads = self.num_heads
        hparams.relu_dropout = self.droprate["rnn"]
        hparams.pos = self.pos_encoding

        hparams.max_length = self.max_seq_length

        hparams.unidirectional_encoder = True

        hparams.self_attention_type = "dot_product_relative_v2"
        hparams.max_relative_position = 5
        hparams.add_relative_to_values = True
        return hparams

    def _create_tf_transformer_encoder(self, a_in, c_in, b_prev_in, mask, attention_weights):
        hparams = self._create_hparams()

        x_in = tf.concat([a_in, b_prev_in, c_in], -1)

        # When not in training mode, set all forms of dropout to zero.
        for key, value in hparams.values().items():
            if key.endswith("dropout") or key == "label_smoothing":
                setattr(hparams, key, value * tf.cast(self._is_training, tf.float32))
        reg = tf.contrib.layers.l2_regularizer(self.C2)

        x = tf.layers.dense(inputs=x_in,
                            units=hparams.hidden_size,
                            use_bias=False,
                            kernel_initializer=tf.random_normal_initializer(0.0, hparams.hidden_size ** -0.5),
                            kernel_regularizer=reg,
                            name='transformer_embed_layer',
                            reuse=tf.AUTO_REUSE)

        x = tf.layers.dropout(x, rate=hparams.layer_prepostprocess_dropout, training=self._is_training)

        if hparams.multiply_embedding_mode == "sqrt_depth":
            x *= hparams.hidden_size ** 0.5

        x *= tf.expand_dims(mask, -1)

        with tf.variable_scope('transformer', reuse=tf.AUTO_REUSE):
            (x,
             self_attention_bias,
             encoder_decoder_attention_bias
             ) = transformer_prepare_encoder(x, None, hparams)

            if hparams.pos == 'custom_timing':
                x = common_attention.add_timing_signal_1d(x, max_timescale=self.pos_max_timescale)

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

            return tf.nn.relu(x)

    def _create_tf_dial(self) -> Tuple['tf.Tensor', 'tf.Tensor']:
        # mask different length sequences
        # if there is at least one `-1` it should be masked
        mask = tf.sign(tf.reduce_max(self.a_in, -1) + 1)

        self.attention_weights = {}
        a = self._create_tf_transformer_encoder(
            self.a_in, self.c_in, self.b_prev_in, mask, self.attention_weights
        )
        dial_embed = self._create_tf_embed(a, layer_name_suffix="out")

        if isinstance(self.featurizer, MaxHistoryTrackerFeaturizer):
            # pick last action if max history featurizer is used
            dial_embed = dial_embed[:, -1:, :]
            mask = mask[:, -1:]

        return dial_embed, mask

    @staticmethod
    def _tf_make_flat(x):
        return tf.reshape(x, (-1, x.shape[-1]))

    @staticmethod
    def _tf_sample_neg(batch_size,
                       all_bs,
                       neg_ids,
                       ) -> 'tf.Tensor':

        tiled_all_bs = tf.tile(tf.expand_dims(all_bs, 0), (batch_size, 1, 1))

        return tf.batch_gather(tiled_all_bs, neg_ids)

    def _tf_calc_iou_mask(self,
                          pos_b,
                          all_bs,
                          neg_ids,
                          ) -> 'tf.Tensor':

        pos_b_in_flat = pos_b[:, tf.newaxis, :]
        neg_b_in_flat = self._tf_sample_neg(tf.shape(pos_b)[0], all_bs, neg_ids)

        intersection_b_in_flat = tf.minimum(neg_b_in_flat, pos_b_in_flat)
        union_b_in_flat = tf.maximum(neg_b_in_flat, pos_b_in_flat)

        iou = tf.reduce_sum(intersection_b_in_flat, -1) / tf.reduce_sum(union_b_in_flat, -1)
        return 1. - tf.nn.relu(tf.sign(1. - iou))

    def _tf_get_negs(self, all_embed, all_raw, raw_pos):

        batch_size = tf.shape(raw_pos)[0]
        seq_length = tf.shape(raw_pos)[1]
        raw_flat = self._tf_make_flat(raw_pos)

        neg_ids = tf.random.categorical(tf.log(tf.ones((batch_size * seq_length,
                                                        tf.shape(all_raw)[0]))),
                                        self.num_neg)

        bad_negs_flat = self._tf_calc_iou_mask(raw_flat, all_raw, neg_ids)
        bad_negs = tf.reshape(bad_negs_flat, (batch_size, seq_length, -1))

        neg_embed_flat = self._tf_sample_neg(batch_size * seq_length, all_embed, neg_ids)
        neg_embed = tf.reshape(neg_embed_flat,
                               (batch_size, seq_length, -1, all_embed.shape[-1]))

        return neg_embed, bad_negs

    def _sample_negatives(self, all_actions):

        # sample negatives
        pos_dial_embed = self.dial_embed[:, :, tf.newaxis, :]
        neg_dial_embed, dial_bad_negs = self._tf_get_negs(
            self._tf_make_flat(self.dial_embed),
            self._tf_make_flat(self.b_in),
            self.b_in
        )
        pos_bot_embed = self.bot_embed[:, :, tf.newaxis, :]
        neg_bot_embed, bot_bad_negs = self._tf_get_negs(
            self.all_bot_embed,
            all_actions,
            self.b_in
        )
        return (pos_dial_embed, pos_bot_embed, neg_dial_embed, neg_bot_embed,
                dial_bad_negs, bot_bad_negs)

    @staticmethod
    def _tf_raw_sim(
        a: 'tf.Tensor',
        b: 'tf.Tensor',
        mask: 'tf.Tensor',
    ) -> 'tf.Tensor':

        return tf.reduce_sum(a * b, -1) * tf.expand_dims(mask, 2)

    def _tf_sim(
        self,
        pos_dial_embed: 'tf.Tensor',
        pos_bot_embed: 'tf.Tensor',
        neg_dial_embed: 'tf.Tensor',
        neg_bot_embed: 'tf.Tensor',
        dial_bad_negs: 'tf.Tensor',
        bot_bad_negs: 'tf.Tensor',
        mask: 'tf.Tensor',
    ) -> Tuple['tf.Tensor', 'tf.Tensor', 'tf.Tensor', 'tf.Tensor', 'tf.Tensor']:
        """Define similarity."""

        # calculate similarity with several
        # embedded actions for the loss
        neg_inf = common_attention.large_compatible_negative(pos_dial_embed.dtype)

        sim_pos = self._tf_raw_sim(pos_dial_embed, pos_bot_embed, mask)
        sim_neg = self._tf_raw_sim(pos_dial_embed, neg_bot_embed,
                                   mask) + neg_inf * bot_bad_negs
        sim_neg_bot_bot = self._tf_raw_sim(pos_bot_embed, neg_bot_embed,
                                           mask) + neg_inf * bot_bad_negs
        sim_neg_dial_dial = self._tf_raw_sim(pos_dial_embed, neg_dial_embed,
                                             mask) + neg_inf * dial_bad_negs
        sim_neg_bot_dial = self._tf_raw_sim(pos_bot_embed, neg_dial_embed,
                                            mask) + neg_inf * dial_bad_negs

        # output similarities between user input and bot actions
        # and similarities between bot actions and similarities between user inputs
        return sim_pos, sim_neg, sim_neg_bot_bot, sim_neg_dial_dial, sim_neg_bot_dial

    @staticmethod
    def _tf_calc_accuracy(sim_pos, sim_neg):

        max_all_sim = tf.reduce_max(tf.concat([sim_pos, sim_neg], -1), -1)
        return tf.reduce_mean(tf.cast(tf.math.equal(max_all_sim, sim_pos[:, :, 0]),
                                      tf.float32))

    def _tf_loss_margin(
        self,
        sim_pos: 'tf.Tensor',
        sim_neg: 'tf.Tensor',
        sim_neg_bot_bot: 'tf.Tensor',
        sim_neg_dial_dial: 'tf.Tensor',
        sim_neg_bot_dial: 'tf.Tensor',
        mask: 'tf.Tensor',
    ) -> 'tf.Tensor':
        """Define max margin loss."""

        # loss for maximizing similarity with correct action
        loss = tf.maximum(0., self.mu_pos - sim_pos[:, :, 0])

        # loss for minimizing similarity with `num_neg` incorrect actions
        if self.use_max_sim_neg:
            # minimize only maximum similarity over incorrect actions
            max_sim_neg = tf.reduce_max(sim_neg, -1)
            loss += tf.maximum(0., self.mu_neg + max_sim_neg)
        else:
            # minimize all similarities with incorrect actions
            max_margin = tf.maximum(0., self.mu_neg + sim_neg)
            loss += tf.reduce_sum(max_margin, -1)

        # penalize max similarity between pos bot and neg bot embeddings
        max_sim_neg_bot = tf.maximum(0., tf.reduce_max(sim_neg_bot_bot, -1))
        loss += max_sim_neg_bot * self.C_emb

        # penalize max similarity between pos dial and neg dial embeddings
        max_sim_neg_dial = tf.maximum(0., tf.reduce_max(sim_neg_dial_dial, -1))
        loss += max_sim_neg_dial * self.C_emb

        # penalize max similarity between pos bot and neg dial embeddings
        max_sim_neg_dial = tf.maximum(0., tf.reduce_max(sim_neg_bot_dial, -1))
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
        sim_pos: 'tf.Tensor',
        sim_neg: 'tf.Tensor',
        sim_neg_bot_bot: 'tf.Tensor',
        sim_neg_dial_dial: 'tf.Tensor',
        sim_neg_bot_dial: 'tf.Tensor',
        mask: 'tf.Tensor',
    ) -> 'tf.Tensor':
        """Define softmax loss."""

        logits = tf.concat([sim_pos,
                            sim_neg,
                            sim_neg_bot_bot,
                            sim_neg_dial_dial,
                            sim_neg_bot_dial
                            ], -1)

        # create labels for softmax
        pos_labels = tf.ones_like(logits[:, :, :1])
        neg_labels = tf.zeros_like(logits[:, :, 1:])
        labels = tf.concat([pos_labels, neg_labels], -1)

        # mask loss by prediction confidence
        pred = tf.nn.softmax(logits)
        already_learned = tf.pow((1 - pred[:, :, 0]) / 0.5, 4)

        loss = tf.losses.softmax_cross_entropy(labels,
                                               logits,
                                               mask * already_learned)
        # add regularization losses
        loss += tf.losses.get_regularization_loss()

        return loss

    def _choose_loss(self,
                     sim_pos: 'tf.Tensor',
                     sim_neg: 'tf.Tensor',
                     sim_neg_bot_bot: 'tf.Tensor',
                     sim_neg_dial_dial: 'tf.Tensor',
                     sim_neg_bot_dial: 'tf.Tensor',
                     mask: 'tf.Tensor') -> 'tf.Tensor':

        if self.loss_type == 'margin':
            return self._tf_loss_margin(sim_pos, sim_neg,
                                        sim_neg_bot_bot,
                                        sim_neg_dial_dial,
                                        sim_neg_bot_dial,
                                        mask)
        elif self.loss_type == 'softmax':
            return self._tf_loss_softmax(sim_pos, sim_neg,
                                         sim_neg_bot_bot,
                                         sim_neg_dial_dial,
                                         sim_neg_bot_dial,
                                         mask)
        else:
            raise ValueError(
                "Wrong loss type {}, "
                "should be 'margin' or 'softmax'"
                "".format(self.loss_type)
            )

    def _build_tf_train_graph(self, iterator):

        # session data are int counts but we need a float tensors
        (self.a_in,
         self.b_in,
         self.c_in,
         self.b_prev_in) = (tf.cast(x_in, tf.float32) for x_in in iterator.get_next())

        all_actions = tf.constant(self.encoded_all_actions,
                                  dtype=tf.float32,
                                  name="all_actions")

        self.dial_embed, mask = self._create_tf_dial()

        self.bot_embed = self._create_tf_bot_embed(self.b_in)
        self.all_bot_embed = self._create_tf_bot_embed(all_actions)

        if isinstance(self.featurizer, MaxHistoryTrackerFeaturizer):
            # add time dimension if max history featurizer is used
            self.b_in = self.b_in[:, tf.newaxis, :]
            self.bot_embed = self.bot_embed[:, tf.newaxis, :]

        (pos_dial_embed,
         pos_bot_embed,
         neg_dial_embed,
         neg_bot_embed,
         dial_bad_negs,
         bot_bad_negs) = self._sample_negatives(all_actions)

        # calculate similarities
        (sim_pos,
         sim_neg,
         sim_neg_bot_bot,
         sim_neg_dial_dial,
         sim_neg_bot_dial) = self._tf_sim(pos_dial_embed,
                                          pos_bot_embed,
                                          neg_dial_embed,
                                          neg_bot_embed,
                                          dial_bad_negs,
                                          bot_bad_negs,
                                          mask)

        acc = self._tf_calc_accuracy(sim_pos, sim_neg)

        loss = self._choose_loss(sim_pos, sim_neg,
                                 sim_neg_bot_bot,
                                 sim_neg_dial_dial,
                                 sim_neg_bot_dial,
                                 mask)
        return loss, acc

    def _create_tf_placeholders(self, session_data):
        dialogue_len = None  # use dynamic time for rnn
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
        self.c_in = tf.placeholder(
            dtype=tf.float32,
            shape=(None, dialogue_len, session_data.slots.shape[-1]),
            name="slt",
        )
        self.b_prev_in = tf.placeholder(
            dtype=tf.float32,
            shape=(None, dialogue_len, session_data.Y.shape[-1]),
            name="b_prev",
        )

    def _build_tf_pred_graph(self):
        self.dial_embed, mask = self._create_tf_dial()

        self.sim_all = self._tf_raw_sim(
            self.dial_embed[:, :, tf.newaxis, :],
            self.all_bot_embed[tf.newaxis, tf.newaxis, :, :],
            mask
        )

        if self.similarity_type == "cosine":
            # clip negative values to zero
            confidence = tf.nn.relu(self.sim_all)
        else:
            # normalize result to [0, 1] with softmax
            confidence = tf.nn.softmax(self.sim_all)

        self.bot_embed = self._create_tf_bot_embed(self.b_in)

        self.sim = self._tf_raw_sim(
            self.dial_embed[:, :, tf.newaxis, :],
            self.bot_embed,
            mask
        )

        return confidence

    # training methods
    def train(
        self,
        training_trackers: List['DialogueStateTracker'],
        domain: 'Domain',
        **kwargs: Any
    ) -> None:
        """Train the policy on given training trackers."""

        logger.debug("Started training embedding policy.")

        # set numpy random seed
        np.random.seed(self.random_seed)

        # dealing with training data
        training_data = self.featurize_for_training(training_trackers, domain, **kwargs)

        # encode all actions with policies' featurizer
        self.encoded_all_actions = self.featurizer.state_featurizer.create_encoded_all_actions(
            domain
        )

        # check if number of negatives is less than number of actions
        logger.debug(
            "Check if num_neg {} is smaller "
            "than number of actions {}, "
            "else set num_neg to the number of actions - 1"
            "".format(self.num_neg, domain.num_actions)
        )
        self.num_neg = min(self.num_neg, domain.num_actions - 1)

        # extract actual training data to feed to tf session
        session_data = self._create_session_data(training_data.X, training_data.y)

        self.graph = tf.Graph()

        with self.graph.as_default():
            # set random seed in tf
            tf.set_random_seed(self.random_seed)

            # allows increasing batch size
            batch_size_in = tf.placeholder(tf.int64)
            train_dataset = self._create_tf_dataset(session_data, batch_size_in)

            iterator = self._create_tf_iterator(train_dataset)

            train_init_op = iterator.make_initializer(train_dataset)

            if self.evaluate_on_num_examples:
                eval_session_data = self._sample_session_data(session_data, self.evaluate_on_num_examples)
                eval_train_dataset = self._create_tf_dataset(eval_session_data, self.evaluate_on_num_examples, shuffle=False)
                eval_init_op = iterator.make_initializer(eval_train_dataset)
            else:
                eval_init_op = None

            self._is_training = tf.placeholder_with_default(False, shape=())
            loss, acc = self._build_tf_train_graph(iterator)

            # define which optimizer to use
            self._train_op = tf.train.AdamOptimizer().minimize(loss)

            # train tensorflow graph
            self.session = tf.Session(config=self._tf_config)
            self._train_tf_dataset(train_init_op, eval_init_op, batch_size_in,
                                   loss, acc)

            # rebuild the graph for prediction
            self._create_tf_placeholders(session_data)
            self.pred_confidence = self._build_tf_pred_graph()

            # if self.attention_weights.items():
            #     self.attention_weights = tf.concat([tf.expand_dims(t, 0)
            #                                         for name, t in self.attention_weights.items()
            #                                         if name.endswith('multihead_attention/dot_product_attention')], 0)

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

    def _train_tf_dataset(self,
                          train_init_op,
                          eval_init_op,
                          batch_size_in,
                          loss: 'tf.Tensor',
                          acc,
                          ) -> None:
        """Train tf graph"""

        self.session.run(tf.global_variables_initializer())

        if self.evaluate_on_num_examples:
            logger.info(
                "Accuracy is updated every {} epochs"
                "".format(self.evaluate_every_num_epochs)
            )
        pbar = tqdm(range(self.epochs), desc="Epochs", disable=is_logging_disabled())

        eval_acc = 0
        eval_loss = 0
        for ep in pbar:

            batch_size = self._linearly_increasing_batch_size(ep)

            self.session.run(train_init_op, feed_dict={batch_size_in: batch_size})

            ep_train_loss = 0
            ep_train_acc = 0
            batches_per_epoch = 0
            while True:
                try:
                    _, batch_train_loss, batch_train_acc = self.session.run(
                        [self._train_op, loss, acc],
                        feed_dict={self._is_training: True}
                    )
                    batches_per_epoch += 1
                    ep_train_loss += batch_train_loss
                    ep_train_acc += batch_train_acc

                except tf.errors.OutOfRangeError:
                    break

            ep_train_loss /= batches_per_epoch
            ep_train_acc /= batches_per_epoch

            pbar.set_postfix({
                "loss": "{:.3f}".format(ep_train_loss),
                "acc": "{:.3f}".format(ep_train_acc)
            })

            if self.evaluate_on_num_examples and eval_init_op is not None:
                if (ep == 0 or
                        (ep + 1) % self.evaluate_every_num_epochs == 0 or
                        (ep + 1) == self.epochs):
                    eval_loss, eval_acc = self._output_training_stat_dataset(
                        eval_init_op, loss, acc
                    )
                    logger.info("Evaluation results: loss: {:.3f}, acc: {:.3f}"
                                "".format(eval_loss, eval_acc))

        if self.evaluate_on_num_examples:
            logger.info("Finished training embedding classifier, "
                        "loss={:.3f}, accuracy={:.3f}"
                        "".format(eval_loss, eval_acc))

    def _output_training_stat_dataset(self, eval_init_op, loss, acc) -> Tuple[float, float]:
        """Output training statistics"""

        self.session.run(eval_init_op)

        return self.session.run([loss, acc], feed_dict={self._is_training: False})

    def continue_training(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any
    ) -> None:
        """Continue training an already trained policy."""

        batch_size = kwargs.get("batch_size", 5)
        epochs = kwargs.get("epochs", 50)

        for _ in range(epochs):
            training_data = self._training_data_for_continue_training(
                batch_size, training_trackers, domain
            )

            session_data = self._create_session_data(training_data.X, training_data.y)

            b = self._create_batch_b(session_data.Y, session_data.actions_for_Y)

            # fit to one extra example using updated trackers
            self.session.run(
                self._train_op,
                feed_dict={
                    self.a_in: session_data.X,
                    self.b_in: b,
                    self.c_in: session_data.slots,
                    self.b_prev_in: session_data.previous_actions,
                    self._is_training: True,
                },
            )

    def tf_feed_dict_for_prediction(self,
                                    tracker: DialogueStateTracker,
                                    domain: Domain) -> Dict:
        # noinspection PyPep8Naming
        data_X = self.featurizer.create_X([tracker], domain)
        session_data = self._create_session_data(data_X)

        return {self.a_in: session_data.X,
                self.c_in: session_data.slots,
                self.b_prev_in: session_data.previous_actions}

    def predict_action_probabilities(
        self, tracker: DialogueStateTracker, domain: Domain
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

    def _persist_tensor(self, name: Text, tensor: 'tf.Tensor') -> None:
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
            self._persist_tensor("intent_placeholder", self.a_in)
            self._persist_tensor("action_placeholder", self.b_in)
            self._persist_tensor("slots_placeholder", self.c_in)
            self._persist_tensor("prev_act_placeholder", self.b_prev_in)

            self._persist_tensor("similarity_all", self.sim_all)
            self._persist_tensor("pred_confidence", self.pred_confidence)
            self._persist_tensor("similarity", self.sim)

            self._persist_tensor("dial_embed", self.dial_embed)
            self._persist_tensor("bot_embed", self.bot_embed)
            self._persist_tensor("all_bot_embed", self.all_bot_embed)

            self._persist_tensor("attention_weights", self.attention_weights)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        encoded_actions_file = os.path.join(
            path, file_name + ".encoded_all_actions.pkl"
        )
        with open(encoded_actions_file, "wb") as f:
            pickle.dump(self.encoded_all_actions, f)

        tf_config_file = os.path.join(path, file_name + ".tf_config.pkl")
        with open(tf_config_file, "wb") as f:
            pickle.dump(self._tf_config, f)

    @staticmethod
    def load_tensor(name: Text) -> Optional['tf.Tensor']:
        tensor_list = tf.get_collection(name)
        return tensor_list[0] if tensor_list else None

    @classmethod
    def load(cls, path: Text) -> "EmbeddingPolicy":
        """Loads a policy from the storage.

            **Needs to load its featurizer**"""

        if not os.path.exists(path):
            raise Exception(
                "Failed to load dialogue model. Path {} "
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
            sess = tf.Session(config=_tf_config)
            saver = tf.train.import_meta_graph(checkpoint + ".meta")

            saver.restore(sess, checkpoint)

            a_in = cls.load_tensor("intent_placeholder")
            b_in = cls.load_tensor("action_placeholder")
            c_in = cls.load_tensor("slots_placeholder")
            b_prev_in = cls.load_tensor("prev_act_placeholder")

            sim_all = cls.load_tensor("similarity_all")
            pred_confidence = cls.load_tensor("pred_confidence")
            sim = cls.load_tensor("similarity")

            dial_embed = cls.load_tensor("dial_embed")
            bot_embed = cls.load_tensor("bot_embed")
            all_bot_embed = cls.load_tensor("all_bot_embed")

            attention_weights = cls.load_tensor("attention_weights")

        encoded_actions_file = os.path.join(
            path, "{}.encoded_all_actions.pkl".format(file_name)
        )

        with open(encoded_actions_file, "rb") as f:
            encoded_all_actions = pickle.load(f)

        return cls(
            featurizer=featurizer,
            priority=meta["priority"],
            encoded_all_actions=encoded_all_actions,
            graph=graph,
            session=sess,
            intent_placeholder=a_in,
            action_placeholder=b_in,
            slots_placeholder=c_in,
            prev_act_placeholder=b_prev_in,
            similarity_all=sim_all,
            pred_confidence=pred_confidence,
            similarity=sim,
            dial_embed=dial_embed,
            bot_embed=bot_embed,
            all_bot_embed=all_bot_embed,
            attention_weights=attention_weights,
        )
