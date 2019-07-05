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
from rasa.core.actions.action import ACTION_LISTEN_NAME
from rasa.core.domain import Domain
from rasa.core.featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    LabelTokenizerSingleStateFeaturizer,
    MaxHistoryTrackerFeaturizer
)
from rasa.core.policies.policy import Policy

import tensorflow as tf
from tensorflow.python.ops import gen_array_ops
from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models.transformer import transformer_base, transformer_prepare_encoder, transformer_encoder
from tensor2tensor.models.evolved_transformer import evolved_transformer_encoder

from rasa.core.policies.tf_utils import (
    TimeAttentionWrapper,
    ChronoBiasLayerNormBasicLSTMCell,
)
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.common import is_logging_disabled

if typing.TYPE_CHECKING:
    from rasa.core.policies.tf_utils import TimeAttentionWrapperState

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
        "x_for_no_intent",
        "y_for_no_action",
        "y_for_action_listen",
        "all_Y_d",
    ),
)


class EmbeddingPolicy(Policy):
    """Recurrent Embedding Dialogue Policy (REDP)

    The policy that is used in our paper https://arxiv.org/abs/1811.11707
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

        "transformer": False,
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
        "similarity_type": "cosine",  # string 'cosine' or 'inner'
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
        # scale loss with inverse frequency of bot actions
        "scale_loss_by_action_counts": True,
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
        dialogue_len: Optional['tf.Tensor'] = None,
        x_for_no_intent: Optional['tf.Tensor'] = None,
        y_for_no_action: Optional['tf.Tensor'] = None,
        y_for_action_listen: Optional['tf.Tensor'] = None,
        similarity_op: Optional['tf.Tensor'] = None,
        alignment_history: Optional['tf.Tensor'] = None,
        user_embed: Optional['tf.Tensor'] = None,
        bot_embed: Optional['tf.Tensor'] = None,
        slot_embed: Optional['tf.Tensor'] = None,
        dial_embed: Optional['tf.Tensor'] = None,
        rnn_embed: Optional['tf.Tensor'] = None,
        attn_embed: Optional['tf.Tensor'] = None,
        copy_attn_debug: Optional['tf.Tensor'] = None,
        all_time_masks: Optional['tf.Tensor'] = None,
        attention_weights=None,
        max_history: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        # if featurizer:
        #     if not isinstance(featurizer, FullDialogueTrackerFeaturizer):
        #         raise TypeError(
        #             "Passed tracker featurizer of type {}, "
        #             "should be FullDialogueTrackerFeaturizer."
        #             "".format(type(featurizer).__name__)
        #         )
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
        self._dialogue_len = dialogue_len
        self._x_for_no_intent_in = x_for_no_intent
        self._y_for_no_action_in = y_for_no_action
        self._y_for_action_listen_in = y_for_action_listen
        self.sim_op = similarity_op

        # store attention probability distribution as
        # concatenated tensor of each attention types
        self.alignment_history = alignment_history

        # persisted embeddings
        self.user_embed = user_embed
        self.bot_embed = bot_embed
        self.slot_embed = slot_embed
        self.dial_embed = dial_embed

        self.rnn_embed = rnn_embed
        self.attn_embed = attn_embed
        self.copy_attn_debug = copy_attn_debug

        self.all_time_masks = all_time_masks
        self.attention_weights = attention_weights
        # internal tf instances
        self._train_op = None
        self._is_training = None
        self._loss_scales = None

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
        self.transformer = config['transformer']
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
        self.num_neg = config["num_neg"]
        self.use_max_sim_neg = config["use_max_sim_neg"]

    def _load_regularization_params(self, config: Dict[Text, Any]) -> None:
        self.C2 = config["C2"]
        self.C_emb = config["C_emb"]
        self.scale_loss_by_action_counts = config["scale_loss_by_action_counts"]
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
    @staticmethod
    def _create_zero_vector(X: np.ndarray) -> np.ndarray:
        """Create zero vector of shape (1, X.shape[-1])."""

        return np.zeros((1, X.shape[-1]), X.dtype)

    def _create_y_for_action_listen(self, domain: "Domain") -> np.ndarray:
        """Extract feature vector for action_listen"""
        action_listen_idx = domain.index_for_action(ACTION_LISTEN_NAME)
        return self.encoded_all_actions[action_listen_idx : action_listen_idx + 1]

    # noinspection PyPep8Naming
    def _create_all_Y_d(self, dialogue_len: int) -> np.ndarray:
        """Stack encoded_all_intents on top of each other

        to create candidates for training examples and
        to calculate training accuracy.
        """

        return np.stack([self.encoded_all_actions] * dialogue_len)

    # noinspection PyPep8Naming
    def _create_tf_session_data(
        self, domain: "Domain", data_X: np.ndarray, data_Y: Optional[np.ndarray] = None
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

        x_for_no_intent = self._create_zero_vector(X)
        y_for_no_action = self._create_zero_vector(previous_actions)
        y_for_action_listen = self._create_y_for_action_listen(domain)

        # is needed to calculate train accuracy
        if isinstance(self.featurizer, FullDialogueTrackerFeaturizer):
            dial_len = X.shape[1]
        else:
            dial_len = 1
        all_Y_d = self._create_all_Y_d(dial_len)

        return SessionData(
            X=X,
            Y=Y,
            slots=slots,
            previous_actions=previous_actions,
            actions_for_Y=actions_for_Y,
            x_for_no_intent=x_for_no_intent,
            y_for_no_action=y_for_no_action,
            y_for_action_listen=y_for_action_listen,
            all_Y_d=all_Y_d,
        )

    # tf helpers:
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

    def _create_embed(self, x: 'tf.Tensor', layer_name_suffix: Text) -> 'tf.Tensor':
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
        return embed_x

    def _create_tf_user_embed(self, a_in: 'tf.Tensor') -> 'tf.Tensor':
        """Create embedding user vector."""

        layer_name_suffix = "a_and_b" if self.share_embedding else "a"

        a = self._create_tf_nn(
            a_in,
            self.hidden_layer_sizes["a"],
            self.droprate["a"],
            layer_name_suffix=layer_name_suffix,
        )
        return self._create_embed(a, layer_name_suffix=layer_name_suffix)

    def _create_tf_bot_embed(self, b_in: 'tf.Tensor') -> 'tf.Tensor':
        """Create embedding bot vector."""

        layer_name_suffix = "a_and_b" if self.share_embedding else "b"

        b = self._create_tf_nn(
            b_in,
            self.hidden_layer_sizes["b"],
            self.droprate["b"],
            layer_name_suffix=layer_name_suffix,
        )
        return self._create_embed(b, layer_name_suffix=layer_name_suffix)

    def _create_tf_no_intent_embed(self, x_for_no_intent_i: 'tf.Tensor') -> 'tf.Tensor':
        """Create embedding user vector for empty intent."""

        layer_name_suffix = "a_and_b" if self.share_embedding else "a"

        x_for_no_intent = self._create_tf_nn(
            x_for_no_intent_i,
            self.hidden_layer_sizes["a"],
            droprate=0,
            layer_name_suffix=layer_name_suffix,
        )
        return tf.stop_gradient(
            self._create_embed(x_for_no_intent, layer_name_suffix=layer_name_suffix)
        )

    def _create_tf_no_action_embed(self, y_for_no_action_in: 'tf.Tensor') -> 'tf.Tensor':
        """Create embedding bot vector for empty action and action_listen."""

        layer_name_suffix = "a_and_b" if self.share_embedding else "b"

        y_for_no_action = self._create_tf_nn(
            y_for_no_action_in,
            self.hidden_layer_sizes["b"],
            droprate=0,
            layer_name_suffix=layer_name_suffix,
        )
        return tf.stop_gradient(
            self._create_embed(y_for_no_action, layer_name_suffix=layer_name_suffix)
        )

    def _create_rnn_cell(self):
        # type: () -> tf.contrib.rnn.RNNCell
        """Create one rnn cell."""

        # chrono initialization for forget bias
        # assuming that characteristic time is max dialogue length
        # left border that initializes forget gate close to 0
        bias_0 = -1.0

        # right border that initializes forget gate close to 1
        bias_1 = np.log(self.characteristic_time - 1.0)
        fbias = (bias_1 - bias_0) * np.random.random(self.rnn_size) + bias_0

        if self.attn_after_rnn:
            # since attention is copied to rnn output,
            # embedding should be performed inside the cell
            embed_layer_size = self.embed_dim
        else:
            embed_layer_size = None

        keep_prob = 1.0 - (
            self.droprate["rnn"] * tf.cast(self._is_training, tf.float32)
        )

        return ChronoBiasLayerNormBasicLSTMCell(
            num_units=self.rnn_size,
            layer_norm=self.layer_norm,
            forget_bias=fbias,
            input_bias=-fbias,
            dropout_keep_prob=keep_prob,
            out_layer_size=embed_layer_size,
        )

    @staticmethod
    def _num_units(memory: 'tf.Tensor') -> int:
        return memory.shape[-1].value

    def _create_attn_mech(
        self, memory: 'tf.Tensor', real_length: 'tf.Tensor'
    ) -> tf.contrib.seq2seq.AttentionMechanism:

        return tf.contrib.seq2seq.BahdanauAttention(
            num_units=self._num_units(memory),
            memory=memory,
            memory_sequence_length=real_length,
            normalize=True,
            probability_fn=tf.identity,
            # we only attend to memory up to a current time step
            # it does not affect alignments, but
            # is important for interpolation gate
            score_mask_value=0,
        )

    def cell_input_fn(
        self,
        rnn_inputs: 'tf.Tensor',
        attention: 'tf.Tensor',
        num_cell_input_memory_units: int,
    ) -> 'tf.Tensor':
        """Combine rnn inputs and attention into cell input.

        Args:
          rnn_inputs: Tensor, first output from `rnn_and_attn_inputs_fn`.

          attention: Tensor, concatenated all attentions for one time step.

          num_cell_input_memory_units: int, number of the first units in
                                       `attention` that are responsible for
                                       enhancing cell input.

        Returns:
          A Tensor `cell_inputs` to feed to an rnn cell.
        """

        if num_cell_input_memory_units:
            if num_cell_input_memory_units == self.embed_dim:
                # since attention can contain additional
                # attention mechanisms, only attention
                # from previous user input is used as an input
                # for rnn cell and only if memory before rnn
                # is the same size as embed_utter
                return tf.concat(
                    [
                        rnn_inputs[:, : self.embed_dim]
                        + attention[:, :num_cell_input_memory_units],
                        rnn_inputs[:, self.embed_dim :],
                    ],
                    -1,
                )
            else:
                # in current implementation it cannot fall here,
                # but this Exception exists in case
                # attention before rnn is changed
                raise ValueError(
                    "Number of memory units {} is not "
                    "equal to number of utter units {}. "
                    "Please modify cell input function "
                    "accordingly."
                    "".format(num_cell_input_memory_units, self.embed_dim)
                )
        else:
            return rnn_inputs

    def rnn_and_attn_inputs_fn(
        self, inputs: 'tf.Tensor', cell_state: 'tf.Tensor'
    ) -> Tuple['tf.Tensor', 'tf.Tensor']:
        """Construct rnn input and attention mechanism input.

        Args:
          inputs: Tensor, concatenated all embeddings for one time step:
                  [embed_utter, embed_slots, embed_prev_action].

          cell_state: Tensor, state of an rnn cell.

        Returns:
          Tuple of Tensors `rnn_inputs, attn_inputs` to feed to
          rnn and attention mechanisms.
        """

        # the hidden state c and slots are not included,
        # in hope that algorithm would learn correct attention
        # regardless of the hidden state c of an lstm and slots
        if isinstance(cell_state, tf.contrib.rnn.LSTMStateTuple):
            attn_inputs = tf.concat([inputs[:, : self.embed_dim], cell_state.h], -1)
        else:
            attn_inputs = tf.concat([inputs[:, : self.embed_dim], cell_state], -1)

        # include slots in inputs but exclude previous action, since
        # rnn should get previous action from its hidden state
        rnn_inputs = inputs[:, : (self.embed_dim + self.embed_dim)]

        return rnn_inputs, attn_inputs

    def _create_attn_cell(
        self,
        cell: tf.contrib.rnn.RNNCell,
        embed_utter: 'tf.Tensor',
        embed_prev_action: 'tf.Tensor',
        real_length: 'tf.Tensor',
        embed_for_no_intent: 'tf.Tensor',
        embed_for_no_action: 'tf.Tensor',
        embed_for_action_listen: 'tf.Tensor',
    ) -> tf.contrib.rnn.RNNCell:
        """Wrap cell in attention wrapper with given memory."""

        if self.attn_before_rnn:
            # create attention over previous user input
            num_memory_units_before_rnn = self._num_units(embed_utter)
            with tf.variable_scope('before', reuse=tf.AUTO_REUSE):
                attn_mech = self._create_attn_mech(embed_utter, real_length)

            # create mask for empty user input not to pay attention to it
            ignore_mask = tf.reduce_all(
                tf.equal(tf.expand_dims(embed_for_no_intent, 0), embed_utter), -1
            )

            # do not use attention by location before rnn
            attn_shift_range = 0
        else:
            attn_mech = None
            ignore_mask = None
            num_memory_units_before_rnn = None
            attn_shift_range = None

        if self.attn_after_rnn:
            # create attention over previous bot actions
            with tf.variable_scope('after', reuse=tf.AUTO_REUSE):
                attn_mech_after_rnn = self._create_attn_mech(embed_prev_action, real_length)

            # create mask for empty bot action or action_listen
            # not to pay attention to them
            ignore_mask_listen = tf.logical_or(
                tf.reduce_all(
                    tf.equal(tf.expand_dims(embed_for_no_action, 0), embed_prev_action),
                    -1,
                ),
                tf.reduce_all(
                    tf.equal(
                        tf.expand_dims(embed_for_action_listen, 0), embed_prev_action
                    ),
                    -1,
                ),
            )

            if attn_mech is not None:
                # if there is another attention mechanism,
                # create a list of attention mechanisms
                attn_mech = [attn_mech, attn_mech_after_rnn]
                ignore_mask = [ignore_mask, ignore_mask_listen]
                attn_shift_range = [attn_shift_range, self.attn_shift_range]
            else:
                attn_mech = attn_mech_after_rnn
                ignore_mask = ignore_mask_listen
                attn_shift_range = self.attn_shift_range

            # this particular attention mechanism is unusual
            # in the sense that its calculated attention vector is directly
            # added to cell output, therefore enabling copy mechanism

            # `index_of_attn_to_copy` is used by `TimeAttentionWrapper`,
            # to know which attention to copy
            index_of_attn_to_copy = -1
        else:
            index_of_attn_to_copy = None

        return TimeAttentionWrapper(
            cell=cell,
            attention_mechanism=attn_mech,
            sequence_len=self._dialogue_len,
            attn_shift_range=attn_shift_range,
            sparse_attention=self.sparse_attention,
            rnn_and_attn_inputs_fn=self.rnn_and_attn_inputs_fn,
            ignore_mask=ignore_mask,
            cell_input_fn=lambda inputs, attention: (
                self.cell_input_fn(inputs, attention, num_memory_units_before_rnn)
            ),
            index_of_attn_to_copy=index_of_attn_to_copy,
            likelihood_fn=lambda emb_1, emb_2: (self._tf_sim(emb_1, emb_2, None)),
            tensor_not_to_copy=embed_for_action_listen,
            output_attention=True,
            alignment_history=True,
        )

    def _create_tf_dial_embed(
        self,
        embed_utter: 'tf.Tensor',
        embed_slots: 'tf.Tensor',
        embed_prev_action: 'tf.Tensor',
        mask: 'tf.Tensor',
        embed_for_no_intent: 'tf.Tensor',
        embed_for_no_action: 'tf.Tensor',
        embed_for_action_listen: 'tf.Tensor',
    ) -> Tuple['tf.Tensor', Union['tf.Tensor', "TimeAttentionWrapperState"]]:
        """Create rnn for dialogue level embedding."""

        cell_input = tf.concat([embed_utter, embed_slots, embed_prev_action], -1)

        cell = self._create_rnn_cell()

        real_length = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

        if self.is_using_attention():
            cell = self._create_attn_cell(
                cell,
                embed_utter,
                embed_prev_action,
                real_length,
                embed_for_no_intent,
                embed_for_no_action,
                embed_for_action_listen,
            )

        with tf.variable_scope('rnn_decoder', reuse=tf.AUTO_REUSE):
            return tf.nn.dynamic_rnn(
                cell,
                cell_input,
                dtype=tf.float32,
                sequence_length=real_length,
            )

    def _create_transformer_encoder(self, a_in, c_in, b_prev_in, mask, attention_weights):
        x_in = tf.concat([a_in, b_prev_in], -1)
        # print(x_in.shape[-1])
        # exit()

        # x = x_in
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

        # hparams.proximity_bias = True

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
        # a = tf.layers.dense(inputs=a_in,
        #                     units=hparams.hidden_size/3,
        #                     use_bias=False,
        #                     kernel_initializer=tf.random_normal_initializer(0.0, hparams.hidden_size ** -0.5),
        #                     kernel_regularizer=reg,
        #                     name='transformer_embed_layer_a',
        #                     reuse=tf.AUTO_REUSE)
        #
        c = tf.layers.dense(inputs=c_in,
                            units=hparams.hidden_size,
                            use_bias=False,
                            kernel_initializer=tf.random_normal_initializer(0.0, hparams.hidden_size ** -0.5),
                            kernel_regularizer=reg,
                            name='transformer_embed_layer_c',
                            reuse=tf.AUTO_REUSE)
        #
        # b = tf.layers.dense(inputs=b_prev_in,
        #                     units=hparams.hidden_size/3,
        #                     use_bias=False,
        #                     kernel_initializer=tf.random_normal_initializer(0.0, hparams.hidden_size ** -0.5),
        #                     kernel_regularizer=reg,
        #                     name='transformer_embed_layer_b',
        #                     reuse=tf.AUTO_REUSE)

        # x = tf.concat([a, c, b], -1)

        x = tf.layers.dropout(x, rate=hparams.layer_prepostprocess_dropout, training=self._is_training)

        if hparams.multiply_embedding_mode == "sqrt_depth":
            x *= hparams.hidden_size ** 0.5
            c *= hparams.hidden_size ** 0.5

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

            # x = tf.concat([x, c_in], -1)
            # c_gate = tf.layers.dense(inputs=x,
            #                          # units=hparams.hidden_size,
            #                          # activation=tf.nn.softmax,
            #                          units=1,
            #                          activation=tf.math.sigmoid,
            #                          bias_initializer=tf.constant_initializer(-1),
            #                          # use_bias=False,
            #                          # kernel_initializer=tf.random_normal_initializer(0.0, hparams.hidden_size ** -0.5),
            #                          kernel_regularizer=reg,
            #                          name='slots_gate_layer_c',
            #                          reuse=tf.AUTO_REUSE)
            x += c #* c_gate
            # x = common_layers.layer_postprocess(x, c, hparams)
            x *= tf.expand_dims(mask, -1)

            return tf.nn.relu(x), self_attention_bias, x_in

    @staticmethod
    def _rearrange_fn(list_tensor_1d_mask_1d):
        """Rearranges tensor_1d to put all the values
            where mask_1d=1 to the right and
            where mask_1d=0 to the left"""
        tensor_1d, mask_1d = list_tensor_1d_mask_1d

        partitioned_tensor = tf.dynamic_partition(tensor_1d, mask_1d, 2)

        return tf.concat(partitioned_tensor, 0)

    @staticmethod
    def _arrange_back_fn(list_tensor_1d_mask_1d):
        """Arranges back tensor_1d to restore original order
            modified by `_rearrange_fn` according to mask_1d:
            - number of 0s in mask_1d values on the left are set to
              their corresponding places where mask_1d=0,
            - number of 1s in mask_1d values on the right are set to
              their corresponding places where mask_1d=1"""
        tensor_1d, mask_1d = list_tensor_1d_mask_1d

        mask_indices = tf.dynamic_partition(
            tf.range(tf.shape(tensor_1d)[0]), mask_1d, 2
        )

        mask_sum = tf.reduce_sum(mask_1d, axis=0)
        partitioned_tensor = [
            tf.zeros_like(tensor_1d[:-mask_sum]),
            tensor_1d[-mask_sum:],
        ]

        return tf.dynamic_stitch(mask_indices, partitioned_tensor)

    def _action_to_copy(self, x_in, x, self_attention_bias, embed_prev_action, embed_for_action_listen, embed_for_no_action):
        with tf.variable_scope('copy', reuse=tf.AUTO_REUSE):
            ignore_mask_listen = tf.to_float(tf.logical_or(
                tf.reduce_all(
                    tf.equal(tf.expand_dims(embed_for_no_action, 0), embed_prev_action),
                    -1,
                ),
                tf.reduce_all(
                    tf.equal(tf.expand_dims(embed_for_action_listen, 0), embed_prev_action),
                    -1,
                ),
            ))

            triag_mask = tf.expand_dims(
                common_attention.attention_bias_to_padding(self_attention_bias[0, 0, :, tf.newaxis, tf.newaxis, :]), 0)
            diag_mask = 1 - (1 - triag_mask) * tf.cumprod(triag_mask, axis=-1, exclusive=True, reverse=True)

            bias = self_attention_bias + common_attention.attention_bias_ignore_padding(ignore_mask_listen) * tf.expand_dims(diag_mask, 1)

            copy_weights = {}
            common_attention.multihead_attention(x_in,
                                                 embed_prev_action,
                                                 bias,
                                                 self.rnn_size,
                                                 self.embed_dim,
                                                 self.embed_dim,
                                                 1,
                                                 0,
                                                 save_weights_to=copy_weights)

        copy_weights = copy_weights['copy/multihead_attention/dot_product_attention'][:, 0, :, :]
        bias = bias[:, 0, :, :]
        shape = tf.shape(copy_weights)
        copy_weights = tf.reshape(copy_weights, (-1, shape[-1]))
        x_flat = tf.reshape(x_in, (-1, x_in.shape[-1]))
        bias = tf.reshape(bias, (-1, shape[-1]))
        ignore_mask = common_attention.attention_bias_to_padding(bias[:, tf.newaxis, tf.newaxis, :], tf.to_int32)

        s_w = tf.layers.dense(
            inputs=x_flat,
            units=2 * self.attn_shift_range + 1,
            activation=tf.nn.softmax,
            name="shift_weight",
            reuse=tf.AUTO_REUSE
        )
        mask = 1 - ignore_mask
        conv_weights = tf.map_fn(
            self._rearrange_fn, [copy_weights, mask], dtype=copy_weights.dtype
        )

        conv_weights = tf.reverse(conv_weights, axis=[1])

        # preare probs for tf.nn.depthwise_conv2d
        # [in_width, in_channels=batch]
        conv_weights = tf.transpose(conv_weights, [1, 0])
        # [batch=1, in_height=1, in_width=time+1, in_channels=batch]
        conv_weights = conv_weights[tf.newaxis, tf.newaxis, :, :]

        # [filter_height=1, filter_width=2*attn_shift_range+1,
        #   in_channels=batch, channel_multiplier=1]
        conv_s_w = tf.transpose(s_w, [1, 0])
        conv_s_w = conv_s_w[tf.newaxis, :, :, tf.newaxis]

        # perform 1d convolution
        # [batch=1, out_height=1, out_width=time+1, out_channels=batch]
        conv_weights = tf.nn.depthwise_conv2d_native(
            conv_weights, conv_s_w, [1, 1, 1, 1], "SAME"
        )
        conv_weights = conv_weights[0, 0, :, :]
        conv_weights = tf.transpose(conv_weights, [1, 0])

        conv_weights = tf.reverse(conv_weights, axis=[1])

        # arrange probs back to their original time order
        copy_weights = tf.map_fn(
            self._arrange_back_fn, [conv_weights, mask], dtype=conv_weights.dtype
        )

        # sharpening parameter
        g_sh = tf.layers.dense(
            inputs=x_flat,
            units=1,
            activation=lambda a: tf.nn.softplus(a) + 1,
            bias_initializer=tf.constant_initializer(1),
            name="gamma_sharp",
            reuse=tf.AUTO_REUSE
        )

        powed_weights = tf.pow(copy_weights, g_sh)
        copy_weights = powed_weights / (tf.reduce_sum(powed_weights, 1, keepdims=True) + 1e-32)

        copy_weights = tf.reshape(copy_weights, shape)

        # remove current time
        copy_prev = copy_weights * diag_mask
        keep_current = copy_weights * (1 - diag_mask)
        dial_embed = self._create_embed(x, layer_name_suffix="out")
        return tf.matmul(copy_prev, embed_prev_action) + tf.matmul(keep_current, dial_embed), copy_weights

    @staticmethod
    def _alignments_history_from(final_state: "TimeAttentionWrapperState") -> 'tf.Tensor':
        """Extract alignments history form final rnn cell state."""

        alignments_from_state = final_state.alignment_history
        if not isinstance(alignments_from_state, tuple):
            alignments_from_state = [alignments_from_state]

        alignment_history = []
        for alignments in alignments_from_state:
            # reshape to (batch, time, memory_time)
            alignment_history.append(tf.transpose(alignments.stack(), [1, 0, 2]))

        return tf.concat(alignment_history, -1)

    @staticmethod
    def _all_time_masks_from(final_state: "TimeAttentionWrapperState") -> 'tf.Tensor':
        """Extract all time masks form final rnn cell state."""

        # reshape to (batch, time, memory_time) and ignore last time
        # because time_mask is created for the next time step
        return tf.transpose(final_state.all_time_masks.stack(), [1, 0, 2])[:, :-1, :]

    def _sims_rnn_to_max_from(self, cell_output: 'tf.Tensor') -> List['tf.Tensor']:
        """Save intermediate tensors for debug purposes."""

        if self.attn_after_rnn:
            # extract additional debug tensors
            num_add = TimeAttentionWrapper.additional_output_size()
            self.copy_attn_debug = cell_output[:, :, -num_add:]

            # extract additional similarity to maximize
            sim_attn_to_max = cell_output[:, :, -num_add]
            sim_state_to_max = cell_output[:, :, -num_add + 1]
            return [sim_attn_to_max, sim_state_to_max]
        else:
            return []

    def _embed_dialogue_from(self, cell_output: 'tf.Tensor') -> 'tf.Tensor':
        """Extract or calculate dialogue level embedding from cell_output."""

        if self.attn_after_rnn:
            # embedding layer is inside rnn cell
            embed_dialogue = cell_output[:, :, : self.embed_dim]

            # extract additional debug tensors
            num_add = TimeAttentionWrapper.additional_output_size()
            self.rnn_embed = cell_output[
                :, :, self.embed_dim : (self.embed_dim + self.embed_dim)
            ]
            self.attn_embed = cell_output[
                :, :, (self.embed_dim + self.embed_dim) : -num_add
            ]
        else:
            # add embedding layer to rnn cell output
            embed_dialogue = self._create_embed(
                cell_output[:, :, : self.rnn_size], layer_name_suffix="out"
            )
            if self.attn_before_rnn:
                # extract additional debug tensors
                self.attn_embed = cell_output[:, :, self.rnn_size :]

        return embed_dialogue

    def _tf_sample_neg(self,
                       pos_b,
                       neg_bs=None,
                       neg_ids=None,
                       batch_size=None,
                       first_only=False
                       ) -> 'tf.Tensor':

        all_b = pos_b[tf.newaxis, :, :]
        if batch_size is None:
            batch_size = tf.shape(pos_b)[0]
        all_b = tf.tile(all_b, [batch_size, 1, 1])
        if neg_bs is None and neg_ids is None:
            return all_b

        def sample_neg_b():
            if neg_bs is not None:
                _neg_bs = neg_bs
            elif neg_ids is not None:
                _neg_bs = tf.batch_gather(all_b, neg_ids)
            else:
                raise
            return tf.concat([pos_b[:, tf.newaxis, :], _neg_bs], 1)

        if first_only:
            out_b = pos_b[:, tf.newaxis, :]
        else:
            out_b = all_b

        if neg_bs is not None:
            cond = tf.logical_and(self._is_training, tf.shape(neg_bs)[0] > 1)
        elif neg_ids is not None:
            cond = tf.logical_and(self._is_training, tf.shape(neg_ids)[0] > 1)
        else:
            raise

        return tf.cond(cond, sample_neg_b, lambda: out_b)

    def _tf_calc_iou(self,
                     b_raw,
                     neg_bs=None,
                     neg_ids=None
                     ) -> 'tf.Tensor':

        tiled_intent_raw = self._tf_sample_neg(b_raw, neg_bs=neg_bs, neg_ids=neg_ids)
        pos_b_raw = tiled_intent_raw[:, :1, :]
        neg_b_raw = tiled_intent_raw[:, 1:, :]
        intersection_b_raw = tf.minimum(neg_b_raw, pos_b_raw)
        union_b_raw = tf.maximum(neg_b_raw, pos_b_raw)

        return tf.reduce_sum(intersection_b_raw, -1) / tf.reduce_sum(union_b_raw, -1)

    def _tf_sim(
        self,
        embed_dialogue: 'tf.Tensor',
        embed_action: 'tf.Tensor',
        mask: Optional['tf.Tensor'],
    ) -> Union[Tuple['tf.Tensor', 'tf.Tensor'],
               Tuple['tf.Tensor', 'tf.Tensor', 'tf.Tensor']]:
        """Define similarity.

        This method has two roles:
        - calculate similarity between
            two embedding vectors of the same size
            and output binary mask and similarity;
        - calculate similarity with several embedded actions for the loss
            and output similarities between user input and bot actions
            and similarities between bot actions.

        They are kept in the same helper method,
        because it is necessary for them to be mathematically identical.
        """

        if self.similarity_type not in {"cosine", "inner"}:
            raise ValueError(
                "Wrong similarity type {}, "
                "should be 'cosine' or 'inner'"
                "".format(self.similarity_type)
            )

        if len(embed_dialogue.shape) == 2 and len(embed_action.shape) == 2:
            # calculate similarity between
            # two embedding vectors of the same size

            # always use cosine sim for copy mech
            embed_dialogue = tf.nn.l2_normalize(embed_dialogue, -1)
            embed_action = tf.nn.l2_normalize(embed_action, -1)

            cos_sim = tf.reduce_sum(embed_dialogue * embed_action, -1, keepdims=True)

            bin_sim = tf.where(
                cos_sim > (self.mu_pos - self.mu_neg) / 2.0,
                tf.ones_like(cos_sim),
                tf.zeros_like(cos_sim),
            )

            # output binary mask and similarity
            return bin_sim, cos_sim

        else:
            # calculate similarity with several
            # embedded actions for the loss

            if self.similarity_type == "cosine":
                # normalize embedding vectors for cosine similarity
                embed_dialogue = tf.nn.l2_normalize(embed_dialogue, -1)
                embed_action = tf.nn.l2_normalize(embed_action, -1)

            if len(embed_dialogue.shape) == 4:
                embed_dialogue_pos = embed_dialogue[:, :, :1, :]
            else:
                embed_dialogue_pos = tf.expand_dims(embed_dialogue, -2)

            sim = tf.reduce_sum(
                embed_dialogue_pos * embed_action, -1
            ) * tf.expand_dims(mask, 2)

            sim_bot_emb = tf.reduce_sum(
                embed_action[:, :, :1, :] * embed_action[:, :, 1:, :], -1
            ) * tf.expand_dims(mask, 2)

            if len(embed_dialogue.shape) == 4:
                sim_dial_emb = tf.reduce_sum(
                    embed_dialogue[:, :, :1, :] * embed_dialogue[:, :, 1:, :], -1
                ) * tf.expand_dims(mask, 2)
            else:
                sim_dial_emb = None

            if len(embed_dialogue.shape) == 4:
                sim_dial_bot_emb = tf.reduce_sum(
                    embed_dialogue[:, :, :1, :] * embed_action[:, :, 1:, :], -1
                ) * tf.expand_dims(mask, 2)
            else:
                sim_dial_bot_emb = None

            # output similarities between user input and bot actions
            # and similarities between bot actions
            return sim,  sim_bot_emb, sim_dial_emb, sim_dial_bot_emb

    # noinspection PyPep8Naming
    def _scale_loss_by_count_actions(
        self,
        X,
        Y,
        slots,
        previous_actions,
    ) -> Union[np.ndarray, List[List]]:
        """Calculate inverse proportionality of repeated actions."""

        if self.scale_loss_by_action_counts:
            # if isinstance(self.featurizer, FullDialogueTrackerFeaturizer):
            #     full = tf.concat([X, slots, previous_actions, Y], -1)
            # else:
            full = Y

            flat = tf.reshape(full, (-1, full.shape[-1]))
            _, i, c = gen_array_ops.unique_with_counts_v2(flat, axis=[0])
            c = tf.cast(c, tf.float32)

            counts = tf.reshape(tf.gather(c, i), (tf.shape(Y)[0], tf.shape(Y)[1]))

            # do not include [-1 -1 ... -1 0] in averaging
            # and smooth it by taking sqrt

            if isinstance(self.featurizer, FullDialogueTrackerFeaturizer):
                # action_listen is the top one by an order
                max_c = tf.math.top_k(c, 2)[0][1]
            else:
                max_c = tf.reduce_max(c)
            # max_c = tf.math.top_k(c, 2)[0][1]
            # max_c = tf.cond(tf.shape(c)[0] > 1, lambda: tf.math.top_k(c, 2)[0][1], lambda: tf.reduce_max(c))
            # max_c = tf.reduce_max(c)

            return tf.maximum(max_c / counts, 1)
            # return tf.maximum(tf.square(max_c / counts), 1)

            # exit()
        #     full_X = tf.concat(
        #         [X, slots, previous_actions, Y], -1
        #     )
        #     full_X = tf.reshape(full_X, (-1, full_X.shape[-1]))
        #     # include [-1 -1 ... -1 0] as first
        #     # full_X = tf.concat([full_X[-1:], full_X], 0)
        #
        #     _, i, c = gen_array_ops.unique_with_counts_v2(full_X, axis=[0])
        #     c = tf.cast(c, tf.float32)
        #
        #     counts = tf.reshape(tf.gather(c, i), (tf.shape(X)[0], tf.shape(X)[1]))
        #
        #     # do not include [-1 -1 ... -1 0] in averaging
        #     # and smooth it by taking sqrt
        #     return tf.maximum(tf.sqrt(tf.reduce_mean(c) / counts), 1)
        else:
            return [[None]]

    def _regularization_loss(self):
        # type: () -> Union['tf.Tensor', int]
        """Add regularization to the embed layer inside rnn cell."""

        if self.attn_after_rnn:
            vars_to_reg = [
                tf.nn.l2_loss(tf_var)
                for tf_var in tf.trainable_variables()
                if "cell/out_layer/kernel" in tf_var.name
            ]
            if vars_to_reg:
                return self.C2 * tf.add_n(vars_to_reg)

        return 0

    def _tf_loss(
        self,
        sim: 'tf.Tensor',
        sim_bot_emb: 'tf.Tensor',
        sim_dial_emb: 'tf.Tensor',
        sims_rnn_to_max: List['tf.Tensor'],
        bad_negs,
        mask: 'tf.Tensor',
        batch_bad_negs
    ) -> 'tf.Tensor':
        """Define loss."""

        # loss for maximizing similarity with correct action
        loss = tf.maximum(0., self.mu_pos - sim[:, :, 0])

        # loss for minimizing similarity with `num_neg` incorrect actions
        sim_neg = sim[:, :, 1:] + common_attention.large_compatible_negative(bad_negs.dtype) * bad_negs
        if self.use_max_sim_neg:
            # minimize only maximum similarity over incorrect actions
            max_sim_neg = tf.reduce_max(sim_neg, -1)
            loss += tf.maximum(0., self.mu_neg + max_sim_neg)
        else:
            # minimize all similarities with incorrect actions
            max_margin = tf.maximum(0., self.mu_neg + sim_neg)
            loss += tf.reduce_sum(max_margin, -1)

        if isinstance(self.featurizer, FullDialogueTrackerFeaturizer) and self.scale_loss_by_action_counts:
            # scale loss inverse proportionally to number of action counts
            loss *= self._loss_scales

        # penalize max similarity between bot embeddings
        sim_bot_emb += common_attention.large_compatible_negative(bad_negs.dtype) * bad_negs
        max_sim_bot_emb = tf.maximum(0., tf.reduce_max(sim_bot_emb, -1))
        loss += max_sim_bot_emb * self.C_emb

        # penalize max similarity between dial embeddings
        if sim_dial_emb is not None:
            sim_dial_emb += common_attention.large_compatible_negative(batch_bad_negs.dtype) * batch_bad_negs
            max_sim_input_emb = tf.maximum(0., tf.reduce_max(sim_dial_emb, -1))
            loss += max_sim_input_emb * self.C_emb

        # maximize similarity returned by time attention wrapper
        for sim_to_add in sims_rnn_to_max:
            loss += tf.maximum(0.0, 1.0 - sim_to_add)

        # mask loss for different length sequences
        loss *= mask
        # average the loss over sequence length
        loss = tf.reduce_sum(loss, -1) / tf.reduce_sum(mask, 1)

        # average the loss over the batch
        loss = (
            tf.reduce_mean(loss)
            # add regularization losses
            + self._regularization_loss()
            + tf.losses.get_regularization_loss()
        )
        return loss

    def _tf_loss_2(
        self,
        sim: 'tf.Tensor',
        sim_bot_emb: 'tf.Tensor',
        sim_dial_emb: 'tf.Tensor',
        sim_dial_bot_emb,
        sims_rnn_to_max: List['tf.Tensor'],
        bad_negs,
        mask: 'tf.Tensor',
        batch_bad_negs=None,
    ) -> 'tf.Tensor':
        """Define loss."""

        all_sim = [sim[:, :, :1],
                   sim[:, :, 1:] + common_attention.large_compatible_negative(bad_negs.dtype) * bad_negs,
                   sim_bot_emb + common_attention.large_compatible_negative(bad_negs.dtype) * bad_negs,
                   ]
        if sim_dial_emb is not None:
            all_sim.append(sim_dial_emb + common_attention.large_compatible_negative(batch_bad_negs.dtype) * batch_bad_negs)

        if sim_dial_bot_emb is not None:
            all_sim.append(sim_dial_bot_emb + common_attention.large_compatible_negative(bad_negs.dtype) * bad_negs)

        logits = tf.concat(all_sim, -1)
        pos_labels = tf.ones_like(logits[:, :, :1])
        neg_labels = tf.zeros_like(logits[:, :, 1:])
        labels = tf.concat([pos_labels, neg_labels], -1)

        pred = tf.nn.softmax(logits)
        # fake_logits = tf.concat([logits[:, :, :1] - common_attention.large_compatible_negative(logits.dtype),
        #                          logits[:, :, 1:] + common_attention.large_compatible_negative(logits.dtype)], -1)

        # ones = tf.ones_like(pred[:, :, 0])
        # zeros = tf.zeros_like(pred[:, :, 0])

        # already_learned = tf.where(pred[:, :, 0] > 0.8, zeros, ones)
        already_learned = tf.pow((1 - pred[:, :, 0]) / 0.5, 4)

        # if isinstance(self.featurizer, FullDialogueTrackerFeaturizer):
        # if self.scale_loss_by_action_counts:
        #     scale_mask = self._loss_scales * mask
        # else:
        scale_mask = mask
        # else:
        #     scale_mask = 1.0

        loss = tf.losses.softmax_cross_entropy(labels,
                                               logits,
                                               scale_mask * already_learned)
        # add regularization losses
        loss += self._regularization_loss() + tf.losses.get_regularization_loss()

        # maximize similarity returned by time attention wrapper
        add_loss = []
        for sim_to_add in sims_rnn_to_max:
            add_loss.append(tf.maximum(0.0, 1.0 - sim_to_add))

        if add_loss:
            # mask loss for different length sequences
            add_loss = sum(add_loss) * mask
            # average the loss over sequence length
            add_loss = tf.reduce_sum(add_loss, -1) / tf.reduce_sum(mask, 1)
            # average the loss over the batch
            add_loss = tf.reduce_mean(add_loss)

            loss += add_loss

        return loss

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
        # assume that characteristic time is the mean length of the dialogues
        self.characteristic_time = np.mean(training_data.true_length)
        if self.attn_shift_range is None:
            self.attn_shift_range = int(self.characteristic_time / 2)

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
        session_data = self._create_tf_session_data(
            domain, training_data.X, training_data.y
        )

        self.graph = tf.Graph()

        with self.graph.as_default():
            # set random seed in tf
            tf.set_random_seed(self.random_seed)

            batch_size_in = tf.placeholder(tf.int64)
            train_dataset = tf.data.Dataset.from_tensor_slices((session_data.X,
                                                                session_data.Y,
                                                                session_data.slots,
                                                                session_data.previous_actions))
            train_dataset = train_dataset.shuffle(buffer_size=len(session_data.X))
            train_dataset = train_dataset.batch(batch_size_in)

            if self.evaluate_on_num_examples:
                ids = np.random.permutation(len(session_data.X))[:self.evaluate_on_num_examples]

                val_dataset = tf.data.Dataset.from_tensor_slices((session_data.X[ids],
                                                                  session_data.Y[ids],
                                                                  session_data.slots[ids],
                                                                  session_data.previous_actions[ids])
                                                                 ).batch(self.evaluate_on_num_examples)
            else:
                val_dataset = None

            iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                       train_dataset.output_shapes,
                                                       output_classes=train_dataset.output_classes)

            self.a_in, self.b_in, self.c_in, self.b_prev_in = iterator.get_next()

            self.a_in = tf.cast(self.a_in, tf.float32)
            self.b_in = tf.cast(self.b_in, tf.float32)
            self.c_in = tf.cast(self.c_in, tf.float32)
            self.b_prev_in = tf.cast(self.b_prev_in, tf.float32)

            # they don't change
            self._x_for_no_intent_in = tf.constant(
                session_data.x_for_no_intent,
                dtype=tf.float32,
                name="x_for_no_intent",
            )
            self._y_for_no_action_in = tf.constant(
                session_data.y_for_no_action,
                dtype=tf.float32,
                name="y_for_no_action",
            )
            self._y_for_action_listen_in = tf.constant(
                session_data.y_for_action_listen,
                dtype=tf.float32,
                name="y_for_action_listen",
            )
            all_actions = tf.constant(self.encoded_all_actions,
                                      dtype=tf.float32,
                                      name="all_actions")

            # dynamic variables
            self._is_training = tf.placeholder_with_default(False, shape=())
            self._dialogue_len = tf.placeholder(
                dtype=tf.int32, shape=(), name="dialogue_len"
            )

            # mask different length sequences
            # if there is at least one `-1` it should be masked
            mask = tf.sign(tf.reduce_max(self.a_in, -1) + 1)

            self.bot_embed = self._create_tf_bot_embed(self.b_in)
            all_actions_embed = self._create_tf_bot_embed(all_actions)

            embed_prev_action = self._create_tf_bot_embed(self.b_prev_in)
            embed_for_no_action = self._create_tf_no_action_embed(
                self._y_for_no_action_in
            )
            embed_for_action_listen = self._create_tf_no_action_embed(
                self._y_for_action_listen_in
            )

            if self.transformer:
                self.attention_weights = {}
                tr_out, self_attention_bias, tr_in = self._create_transformer_encoder(self.a_in, self.c_in, self.b_prev_in, mask, self.attention_weights)
                # self.dial_embed, self.attention_weights = self._action_to_copy(tr_in, tr_out, self_attention_bias, embed_prev_action, embed_for_action_listen, embed_for_no_action)
                self.dial_embed = self._create_embed(tr_out, layer_name_suffix="out") #+ self._create_embed(self.c_in, layer_name_suffix="slots")
                sims_rnn_to_max = []
            else:
                # create embedding vectors
                self.user_embed = self._create_tf_user_embed(self.a_in)
                self.slot_embed = self._create_embed(self.c_in, layer_name_suffix="slt")

                embed_for_no_intent = self._create_tf_no_intent_embed(
                    self._x_for_no_intent_in
                )

                # get rnn output
                cell_output, final_state = self._create_tf_dial_embed(
                    self.user_embed,
                    self.slot_embed,
                    embed_prev_action,
                    mask,
                    embed_for_no_intent,
                    embed_for_no_action,
                    embed_for_action_listen,
                )
                # process rnn output
                if self.is_using_attention():
                    self.alignment_history = self._alignments_history_from(final_state)

                    self.all_time_masks = self._all_time_masks_from(final_state)

                sims_rnn_to_max = self._sims_rnn_to_max_from(cell_output)
                self.dial_embed = self._embed_dialogue_from(cell_output)

            # calculate similarities
            if isinstance(self.featurizer, MaxHistoryTrackerFeaturizer):
                self.b_in = tf.expand_dims(self.b_in, 1)
                self.bot_embed = tf.expand_dims(self.bot_embed, 1)
                self.dial_embed = self.dial_embed[:, -1:, :]
                mask = mask[:, -1:]

            b_raw = tf.reshape(self.b_in, (-1, self.b_in.shape[-1]))

            _, i, c = gen_array_ops.unique_with_counts_v2(b_raw, axis=[0])
            counts = tf.expand_dims(tf.reshape(tf.gather(tf.cast(c, tf.float32), i), (tf.shape(b_raw)[0],)), 0)
            batch_neg_ids = tf.random.categorical(tf.log((1. - tf.eye(tf.shape(b_raw)[0])/counts)), self.num_neg)

            batch_iou_bot = self._tf_calc_iou(b_raw, neg_ids=batch_neg_ids)
            batch_bad_negs = 1. - tf.nn.relu(tf.sign(1. - batch_iou_bot))
            batch_bad_negs = tf.reshape(batch_bad_negs, (tf.shape(self.dial_embed)[0],
                                                         tf.shape(self.dial_embed)[1],
                                                         -1))

            neg_ids = tf.random.categorical(tf.log(tf.ones((tf.shape(b_raw)[0], tf.shape(all_actions)[0]))), self.num_neg)

            tiled_all_actions = tf.tile(tf.expand_dims(all_actions, 0), (tf.shape(b_raw)[0], 1, 1))
            neg_bs = tf.batch_gather(tiled_all_actions, neg_ids)
            iou_bot = self._tf_calc_iou(b_raw, neg_bs)
            bad_negs = 1. - tf.nn.relu(tf.sign(1. - iou_bot))
            bad_negs = tf.reshape(bad_negs, (tf.shape(self.bot_embed)[0],
                                             tf.shape(self.bot_embed)[1],
                                             -1))

            dial_embed_flat = tf.reshape(self.dial_embed, (-1, self.dial_embed.shape[-1]))

            tiled_dial_embed = self._tf_sample_neg(dial_embed_flat, neg_ids=batch_neg_ids, first_only=True)
            tiled_dial_embed = tf.reshape(tiled_dial_embed, (tf.shape(self.dial_embed)[0],
                                                             tf.shape(self.dial_embed)[1],
                                                             -1,
                                                             self.dial_embed.shape[-1]))

            bot_embed_flat = tf.reshape(self.bot_embed, (-1, self.bot_embed.shape[-1]))
            tiled_all_actions_embed = tf.tile(tf.expand_dims(all_actions_embed, 0), (tf.shape(b_raw)[0], 1, 1))
            neg_embs = tf.batch_gather(tiled_all_actions_embed, neg_ids)
            tiled_bot_embed = self._tf_sample_neg(bot_embed_flat, neg_bs=neg_embs)
            tiled_bot_embed = tf.reshape(tiled_bot_embed, (tf.shape(self.bot_embed)[0],
                                                           tf.shape(self.bot_embed)[1],
                                                           -1,
                                                           self.bot_embed.shape[-1]))

            # self.sim_op, sim_bot_emb, sim_dial_emb = self._tf_sim(self.dial_embed, tiled_bot_embed, mask)
            self.sim_op, sim_bot_emb, sim_dial_emb, sim_dial_bot_emb = self._tf_sim(tiled_dial_embed, tiled_bot_embed, mask)

            # construct loss
            if self.scale_loss_by_action_counts:
                self._loss_scales = self._scale_loss_by_count_actions(self.a_in, self.b_in, self.c_in, self.b_prev_in)
            else:
                self._loss_scales = None
            # loss = self._tf_loss_2(self.sim_op, sim_bot_emb, sim_dial_emb, sims_rnn_to_max, bad_negs, mask)
            loss = self._tf_loss_2(self.sim_op, sim_bot_emb, sim_dial_emb, sim_dial_bot_emb, sims_rnn_to_max, bad_negs, mask, batch_bad_negs)

            # define which optimizer to use
            self._train_op = tf.train.AdamOptimizer(
                # learning_rate=0.001, epsilon=1e-16
            ).minimize(loss)

            train_init_op = iterator.make_initializer(train_dataset)
            if self.evaluate_on_num_examples:
                val_init_op = iterator.make_initializer(val_dataset)
            else:
                val_init_op = None

            # train tensorflow graph
            self.session = tf.Session(config=self._tf_config)

            # self._train_tf(session_data, loss, mask)
            self._train_tf_dataset(train_init_op, val_init_op, batch_size_in, loss, mask, session_data.X.shape[1])

            dialogue_len = None  # use dynamic time for rnn
            # create placeholders
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

            # mask different length sequences
            # if there is at least one `-1` it should be masked
            mask = tf.sign(tf.reduce_max(self.a_in, -1) + 1)

            self.bot_embed = self._create_tf_bot_embed(self.b_in)
            embed_prev_action = self._create_tf_bot_embed(self.b_prev_in)

            if self.transformer:
                self.attention_weights = {}
                tr_out, self_attention_bias, tr_in = self._create_transformer_encoder(self.a_in, self.c_in, self.b_prev_in, mask,
                                                                            self.attention_weights)
                # self.dial_embed, self.attention_weights = self._action_to_copy(tr_in, tr_out, self_attention_bias,
                #                                                               embed_prev_action,
                #                                                               embed_for_action_listen,
                #                                                               embed_for_no_action)
                self.dial_embed = self._create_embed(tr_out, layer_name_suffix="out")

            else:
                self.user_embed = self._create_tf_user_embed(self.a_in)
                self.slot_embed = self._create_embed(self.c_in, layer_name_suffix="slt")

                # get rnn output
                cell_output, final_state = self._create_tf_dial_embed(
                    self.user_embed,
                    self.slot_embed,
                    embed_prev_action,
                    mask,
                    embed_for_no_intent,
                    embed_for_no_action,
                    embed_for_action_listen,
                )
                # process rnn output
                if self.is_using_attention():
                    self.alignment_history = self._alignments_history_from(final_state)

                    self.all_time_masks = self._all_time_masks_from(final_state)

                self.dial_embed = self._embed_dialogue_from(cell_output)

            if isinstance(self.featurizer, MaxHistoryTrackerFeaturizer):
                self.dial_embed = self.dial_embed[:, -1:, :]

            self.sim_op, _, _, _ = self._tf_sim(self.dial_embed, self.bot_embed, mask)

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
                          val_init_op,
                          batch_size_in,
                          loss: 'tf.Tensor',
                          mask,
                          dialogue_len,
                          ) -> None:
        """Train tf graph"""

        self.session.run(tf.global_variables_initializer())

        if self.evaluate_on_num_examples:
            logger.info(
                "Accuracy is updated every {} epochs"
                "".format(self.evaluate_every_num_epochs)
            )
        pbar = tqdm(range(self.epochs), desc="Epochs", disable=is_logging_disabled())

        train_acc = 0
        last_loss = 0
        for ep in pbar:

            batch_size = self._linearly_increasing_batch_size(ep)

            self.session.run(train_init_op, feed_dict={batch_size_in: batch_size})

            ep_loss = 0
            batches_per_epoch = 0
            while True:
                try:
                    _, batch_loss = self.session.run((self._train_op, loss),
                                                     feed_dict={self._is_training: True,
                                                                self._dialogue_len: dialogue_len})

                except tf.errors.OutOfRangeError:
                    break

                batches_per_epoch += 1
                ep_loss += batch_loss

            ep_loss /= batches_per_epoch

            if self.evaluate_on_num_examples and val_init_op is not None:
                if (ep == 0 or
                        (ep + 1) % self.evaluate_every_num_epochs == 0 or
                        (ep + 1) == self.epochs):
                    train_acc = self._output_training_stat_dataset(val_init_op, mask, dialogue_len)
                    last_loss = ep_loss

                pbar.set_postfix({
                    "loss": "{:.3f}".format(ep_loss),
                    "acc": "{:.3f}".format(train_acc)
                })
            else:
                pbar.set_postfix({
                    "loss": "{:.3f}".format(ep_loss)
                })

        if self.evaluate_on_num_examples:
            logger.info("Finished training embedding classifier, "
                        "loss={:.3f}, train accuracy={:.3f}"
                        "".format(last_loss, train_acc))

    def _output_training_stat_dataset(self, val_init_op, mask, dialogue_len) -> np.ndarray:
        """Output training statistics"""

        self.session.run(val_init_op)

        sim_, mask_ = self.session.run([self.sim_op, mask],
                                       feed_dict={self._is_training: False,
                                                  self._dialogue_len: dialogue_len})
        sim_ = sim_.reshape((-1, sim_.shape[-1]))
        mask_ = mask_.reshape((-1,))

        train_acc = np.sum((np.max(sim_, -1) == sim_.diagonal()) * mask_) / np.sum(mask_)

        return train_acc

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

            session_data = self._create_tf_session_data(
                domain, training_data.X, training_data.y
            )

            b = self._create_batch_b(session_data.Y, session_data.actions_for_Y)

            batch_loss_scales = self._scale_loss_by_count_actions(
                session_data.X,
                session_data.slots,
                session_data.previous_actions,
                session_data.actions_for_Y,
            )

            # fit to one extra example using updated trackers
            self.session.run(
                self._train_op,
                feed_dict={
                    self.a_in: session_data.X,
                    self.b_in: b,
                    self.c_in: session_data.slots,
                    self.b_prev_in: session_data.previous_actions,
                    self._dialogue_len: session_data.X.shape[1],
                    self._x_for_no_intent_in: session_data.x_for_no_intent,
                    self._y_for_no_action_in: session_data.y_for_no_action,
                    self._y_for_action_listen_in: session_data.y_for_action_listen,
                    self._is_training: True,
                    self._loss_scales: batch_loss_scales,
                },
            )

    def tf_feed_dict_for_prediction(self,
                                    tracker: DialogueStateTracker,
                                    domain: Domain) -> Dict:
        # noinspection PyPep8Naming
        data_X = self.featurizer.create_X([tracker], domain)
        session_data = self._create_tf_session_data(domain, data_X)
        # noinspection PyPep8Naming
        all_Y_d_x = np.stack([session_data.all_Y_d
                              for _ in range(session_data.X.shape[0])])

        return {self.a_in: session_data.X,
                self.b_in: all_Y_d_x,
                self.c_in: session_data.slots,
                self.b_prev_in: session_data.previous_actions,
                self._dialogue_len: session_data.X.shape[1]}

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

        # noinspection PyPep8Naming
        data_X = self.featurizer.create_X([tracker], domain)
        session_data = self._create_tf_session_data(domain, data_X)
        # noinspection PyPep8Naming
        all_Y_d_x = np.stack(
            [session_data.all_Y_d for _ in range(session_data.X.shape[0])]
        )
        # self.similarity_type = 'cosine'
        # mask = tf.sign(tf.reduce_max(self.a_in, -1) + 1)
        # self.sim_op, _, _ = self._tf_sim(self.dial_embed, self.bot_embed, mask)
        _sim = self.session.run(
            self.sim_op,
            feed_dict={
                self.a_in: session_data.X,
                self.b_in: all_Y_d_x,
                self.c_in: session_data.slots,
                self.b_prev_in: session_data.previous_actions,
                self._dialogue_len: session_data.X.shape[1],
            },
        )

        # TODO assume we used inner:
        self.similarity_type = "inner"

        result = _sim[0, -1, :]
        if self.similarity_type == "cosine":
            # clip negative values to zero
            result[result < 0] = 0
        elif self.similarity_type == "inner":
            # normalize result to [0, 1] with softmax but only over 3*num_neg+1 values
            low_ids = result.argsort()[::-1][4*self.num_neg+1:]
            result[low_ids] += -np.inf
            result = np.exp(result)
            result /= np.sum(result)

        return result.tolist()

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
        utils.create_dir_for_file(checkpoint)

        with self.graph.as_default():
            self._persist_tensor("intent_placeholder", self.a_in)
            self._persist_tensor("action_placeholder", self.b_in)
            self._persist_tensor("slots_placeholder", self.c_in)
            self._persist_tensor("prev_act_placeholder", self.b_prev_in)
            self._persist_tensor("dialogue_len", self._dialogue_len)
            self._persist_tensor("x_for_no_intent", self._x_for_no_intent_in)
            self._persist_tensor("y_for_no_action", self._y_for_no_action_in)
            self._persist_tensor("y_for_action_listen", self._y_for_action_listen_in)

            self._persist_tensor("similarity_op", self.sim_op)

            self._persist_tensor("alignment_history", self.alignment_history)

            self._persist_tensor("user_embed", self.user_embed)
            self._persist_tensor("bot_embed", self.bot_embed)
            self._persist_tensor("slot_embed", self.slot_embed)
            self._persist_tensor("dial_embed", self.dial_embed)

            self._persist_tensor("rnn_embed", self.rnn_embed)
            self._persist_tensor("attn_embed", self.attn_embed)
            self._persist_tensor("copy_attn_debug", self.copy_attn_debug)

            self._persist_tensor("all_time_masks", self.all_time_masks)

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
            dialogue_len = cls.load_tensor("dialogue_len")
            x_for_no_intent = cls.load_tensor("x_for_no_intent")
            y_for_no_action = cls.load_tensor("y_for_no_action")
            y_for_action_listen = cls.load_tensor("y_for_action_listen")

            sim_op = cls.load_tensor("similarity_op")

            alignment_history = cls.load_tensor("alignment_history")

            user_embed = cls.load_tensor("user_embed")
            bot_embed = cls.load_tensor("bot_embed")
            slot_embed = cls.load_tensor("slot_embed")
            dial_embed = cls.load_tensor("dial_embed")

            rnn_embed = cls.load_tensor("rnn_embed")
            attn_embed = cls.load_tensor("attn_embed")
            copy_attn_debug = cls.load_tensor("copy_attn_debug")

            all_time_masks = cls.load_tensor("all_time_masks")

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
            dialogue_len=dialogue_len,
            x_for_no_intent=x_for_no_intent,
            y_for_no_action=y_for_no_action,
            y_for_action_listen=y_for_action_listen,
            similarity_op=sim_op,
            alignment_history=alignment_history,
            user_embed=user_embed,
            bot_embed=bot_embed,
            slot_embed=slot_embed,
            dial_embed=dial_embed,
            rnn_embed=rnn_embed,
            attn_embed=attn_embed,
            copy_attn_debug=copy_attn_debug,
            all_time_masks=all_time_masks,
            attention_weights=attention_weights
        )
