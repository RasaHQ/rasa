from collections import namedtuple
import copy
import io
import logging
import os
import warnings

import numpy as np
import typing
from tqdm import tqdm
from typing import (
    Any, List, Optional, Text, Dict, Tuple, Union)

from rasa_core import utils
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.domain import Domain
from rasa_core.featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    LabelTokenizerSingleStateFeaturizer)
from rasa_core.policies.policy import Policy

import tensorflow as tf
from rasa_core.policies.tf_utils import (
    TimeAttentionWrapper,
    ChronoBiasLayerNormBasicLSTMCell)
from rasa_core.trackers import DialogueStateTracker

if typing.TYPE_CHECKING:
    from rasa_core.policies.tf_utils import TimeAttentionWrapperState

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger(__name__)

# namedtuple for all tf session related data
SessionData = namedtuple("SessionData", ("X", "Y", "slots",
                                         "previous_actions",
                                         "actions_for_Y",
                                         "x_for_no_intent",
                                         "y_for_no_action",
                                         "y_for_action_listen",
                                         "all_Y_d"))


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
        # number of units in rnn cell
        "rnn_size": 64,

        # training parameters
        # flag if to turn on layer normalization for lstm cell
        "layer_norm": True,
        # initial and final batch sizes - batch size will be
        # linearly increased for each epoch
        "batch_size": [8, 32],
        # number of epochs
        "epochs": 1,

        # embedding parameters
        # dimension size of embedding vectors
        "embed_dim": 20,
        # how similar the algorithm should try
        # to make embedding vectors for correct actions
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect actions
        "mu_neg": -0.2,  # should be -1.0 < ... < 1.0 for 'cosine'
        # the type of the similarity
        "similarity_type": 'cosine',  # string 'cosine' or 'inner'
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
        "evaluate_on_num_examples": 100  # large values may hurt performance
    }

    # end default properties (DOC MARKER - don't remove)

    @classmethod
    def _standard_featurizer(cls):
        return FullDialogueTrackerFeaturizer(
            LabelTokenizerSingleStateFeaturizer())

    def __init__(
        self,
        featurizer: Optional[FullDialogueTrackerFeaturizer] = None,
        encoded_all_actions: Optional[np.ndarray] = None,
        graph: Optional[tf.Graph] = None,
        session: Optional[tf.Session] = None,
        intent_placeholder: Optional[tf.Tensor] = None,
        action_placeholder: Optional[tf.Tensor] = None,
        slots_placeholder: Optional[tf.Tensor] = None,
        prev_act_placeholder: Optional[tf.Tensor] = None,
        dialogue_len: Optional[tf.Tensor] = None,
        x_for_no_intent: Optional[tf.Tensor] = None,
        y_for_no_action: Optional[tf.Tensor] = None,
        y_for_action_listen: Optional[tf.Tensor] = None,
        similarity_op: Optional[tf.Tensor] = None,
        alignment_history: Optional[tf.Tensor] = None,
        user_embed: Optional[tf.Tensor] = None,
        bot_embed: Optional[tf.Tensor] = None,
        slot_embed: Optional[tf.Tensor] = None,
        dial_embed: Optional[tf.Tensor] = None,
        rnn_embed: Optional[tf.Tensor] = None,
        attn_embed: Optional[tf.Tensor] = None,
        copy_attn_debug: Optional[tf.Tensor] = None,
        all_time_masks: Optional[tf.Tensor] = None,
        **kwargs: Any
    ) -> None:
        if featurizer:
            if not isinstance(featurizer, FullDialogueTrackerFeaturizer):
                raise TypeError("Passed tracker featurizer of type {}, "
                                "should be FullDialogueTrackerFeaturizer."
                                "".format(type(featurizer).__name__))
        super(EmbeddingPolicy, self).__init__(featurizer)

        # flag if to use the same embeddings for user and bot
        try:
            self.share_embedding = \
                self.featurizer.state_featurizer.use_shared_vocab
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

        # internal tf instances
        self._train_op = None
        self._is_training = None
        self._loss_scales = None

    # init helpers
    def _load_nn_architecture_params(self, config: Dict[Text, Any]) -> None:
        self.hidden_layer_sizes = {'a': config['hidden_layers_sizes_a'],
                                   'b': config['hidden_layers_sizes_b']}

        if self.share_embedding:
            if self.hidden_layer_sizes['a'] != self.hidden_layer_sizes['b']:
                raise ValueError("Due to sharing vocabulary "
                                 "in the featurizer, embedding weights "
                                 "are shared as well. "
                                 "So hidden_layers_sizes_a={} should be "
                                 "equal to hidden_layers_sizes_b={}"
                                 "".format(self.hidden_layer_sizes['a'],
                                           self.hidden_layer_sizes['b']))

        self.rnn_size = config['rnn_size']
        self.layer_norm = config['layer_norm']

        self.batch_size = config['batch_size']

        self.epochs = config['epochs']

    def _load_embedding_params(self, config: Dict[Text, Any]) -> None:
        self.embed_dim = config['embed_dim']
        self.mu_pos = config['mu_pos']
        self.mu_neg = config['mu_neg']
        self.similarity_type = config['similarity_type']
        self.num_neg = config['num_neg']
        self.use_max_sim_neg = config['use_max_sim_neg']

    def _load_regularization_params(self, config: Dict[Text, Any]) -> None:
        self.C2 = config['C2']
        self.C_emb = config['C_emb']
        self.scale_loss_by_action_counts = \
            config['scale_loss_by_action_counts']
        self.droprate = {"a": config['droprate_a'],
                         "b": config['droprate_b'],
                         "rnn": config['droprate_rnn']}

    def _load_attn_params(self, config: Dict[Text, Any]) -> None:
        self.sparse_attention = config['sparse_attention']
        self.attn_shift_range = config['attn_shift_range']
        self.attn_after_rnn = config['attn_after_rnn']
        self.attn_before_rnn = config['attn_before_rnn']

    def is_using_attention(self):
        return self.attn_after_rnn or self.attn_before_rnn

    def _load_visual_params(self, config: Dict[Text, Any]) -> None:
        self.evaluate_every_num_epochs = config['evaluate_every_num_epochs']
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs

        self.evaluate_on_num_examples = config['evaluate_on_num_examples']

    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        config = copy.deepcopy(self.defaults)
        config.update(kwargs)

        self._load_nn_architecture_params(config)
        self._load_embedding_params(config)
        self._load_regularization_params(config)
        self._load_attn_params(config)
        self._load_visual_params(config)

    # data helpers
    # noinspection PyPep8Naming
    def _create_X_slots_previous_actions(
        self,
        data_X: np.ndarray
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

        return np.stack([np.stack([self.encoded_all_actions[action_idx]
                                   for action_idx in action_ids])
                         for action_ids in actions_for_Y])

    # noinspection PyPep8Naming
    @staticmethod
    def _create_zero_vector(X: np.ndarray) -> np.ndarray:
        """Create zero vector of shape (1, X.shape[-1])."""

        return np.zeros((1, X.shape[-1]), X.dtype)

    def _create_y_for_action_listen(self, domain: 'Domain') -> np.ndarray:
        """Extract feature vector for action_listen"""
        action_listen_idx = domain.index_for_action(ACTION_LISTEN_NAME)
        return self.encoded_all_actions[action_listen_idx:
                                        action_listen_idx + 1]

    # noinspection PyPep8Naming
    def _create_all_Y_d(self, dialogue_len: int) -> np.ndarray:
        """Stack encoded_all_intents on top of each other

        to create candidates for training examples and
        to calculate training accuracy.
        """

        return np.stack([self.encoded_all_actions] * dialogue_len)

    # noinspection PyPep8Naming
    def _create_tf_session_data(self,
                                domain: 'Domain',
                                data_X: np.ndarray,
                                data_Y: Optional[np.ndarray] = None
                                ) -> SessionData:
        """Combine all tf session related data into a namedtuple"""

        X, slots, previous_actions = \
            self._create_X_slots_previous_actions(data_X)

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
        all_Y_d = self._create_all_Y_d(X.shape[1])

        return SessionData(
            X=X, Y=Y, slots=slots,
            previous_actions=previous_actions,
            actions_for_Y=actions_for_Y,
            x_for_no_intent=x_for_no_intent,
            y_for_no_action=y_for_no_action,
            y_for_action_listen=y_for_action_listen,
            all_Y_d=all_Y_d
        )

        # tf helpers:

    def _create_tf_nn(self,
                      x_in: tf.Tensor,
                      layer_sizes: List,
                      droprate: float,
                      layer_name_suffix: Text) -> tf.Tensor:
        """Create nn with hidden layers and name suffix."""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        x = tf.nn.relu(x_in)
        for i, layer_size in enumerate(layer_sizes):
            x = tf.layers.dense(inputs=x,
                                units=layer_size,
                                activation=tf.nn.relu,
                                kernel_regularizer=reg,
                                name='hidden_layer_{}_{}'
                                     ''.format(layer_name_suffix, i),
                                reuse=tf.AUTO_REUSE)
            x = tf.layers.dropout(x, rate=droprate,
                                  training=self._is_training)
        return x

    def _create_embed(self,
                      x: tf.Tensor,
                      layer_name_suffix: Text) -> tf.Tensor:
        """Create dense embedding layer with a name."""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        embed_x = tf.layers.dense(inputs=x,
                                  units=self.embed_dim,
                                  activation=None,
                                  kernel_regularizer=reg,
                                  name='embed_layer_{}'
                                       ''.format(layer_name_suffix),
                                  reuse=tf.AUTO_REUSE)
        return embed_x

    def _create_tf_user_embed(self, a_in: tf.Tensor) -> tf.Tensor:
        """Create embedding user vector."""

        layer_name_suffix = 'a_and_b' if self.share_embedding else 'a'

        a = self._create_tf_nn(
            a_in,
            self.hidden_layer_sizes['a'],
            self.droprate['a'],
            layer_name_suffix=layer_name_suffix
        )
        return self._create_embed(a, layer_name_suffix=layer_name_suffix)

    def _create_tf_bot_embed(self, b_in: tf.Tensor) -> tf.Tensor:
        """Create embedding bot vector."""

        layer_name_suffix = 'a_and_b' if self.share_embedding else 'b'

        b = self._create_tf_nn(
            b_in,
            self.hidden_layer_sizes['b'],
            self.droprate['b'],
            layer_name_suffix=layer_name_suffix
        )
        return self._create_embed(b, layer_name_suffix=layer_name_suffix)

    def _create_tf_no_intent_embed(self,
                                   x_for_no_intent_i: tf.Tensor) -> tf.Tensor:
        """Create embedding user vector for empty intent."""

        layer_name_suffix = 'a_and_b' if self.share_embedding else 'a'

        x_for_no_intent = self._create_tf_nn(
            x_for_no_intent_i,
            self.hidden_layer_sizes['a'],
            droprate=0,
            layer_name_suffix=layer_name_suffix
        )
        return tf.stop_gradient(
            self._create_embed(x_for_no_intent,
                               layer_name_suffix=layer_name_suffix))

    def _create_tf_no_action_embed(self,
                                   y_for_no_action_in: tf.Tensor) -> tf.Tensor:
        """Create embedding bot vector for empty action and action_listen."""

        layer_name_suffix = 'a_and_b' if self.share_embedding else 'b'

        y_for_no_action = self._create_tf_nn(
            y_for_no_action_in,
            self.hidden_layer_sizes['b'],
            droprate=0,
            layer_name_suffix=layer_name_suffix
        )
        return tf.stop_gradient(
            self._create_embed(y_for_no_action,
                               layer_name_suffix=layer_name_suffix))

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

        keep_prob = 1.0 - (self.droprate['rnn'] *
                           tf.cast(self._is_training, tf.float32))

        return ChronoBiasLayerNormBasicLSTMCell(
            num_units=self.rnn_size,
            layer_norm=self.layer_norm,
            forget_bias=fbias,
            input_bias=-fbias,
            dropout_keep_prob=keep_prob,
            out_layer_size=embed_layer_size
        )

    @staticmethod
    def _num_units(memory: tf.Tensor) -> int:
        return memory.shape[-1].value

    def _create_attn_mech(self,
                          memory: tf.Tensor,
                          real_length: tf.Tensor
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
            score_mask_value=0
        )

    def cell_input_fn(self, rnn_inputs: tf.Tensor, attention: tf.Tensor,
                      num_cell_input_memory_units: int) -> tf.Tensor:
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
                return tf.concat([rnn_inputs[:, :self.embed_dim] +
                                  attention[:, :num_cell_input_memory_units],
                                  rnn_inputs[:, self.embed_dim:]], -1)
            else:
                # in current implementation it cannot fall here,
                # but this Exception exists in case
                # attention before rnn is changed
                raise ValueError("Number of memory units {} is not "
                                 "equal to number of utter units {}. "
                                 "Please modify cell input function "
                                 "accordingly."
                                 "".format(num_cell_input_memory_units,
                                           self.embed_dim))
        else:
            return rnn_inputs

    def rnn_and_attn_inputs_fn(self,
                               inputs: tf.Tensor,
                               cell_state: tf.Tensor
                               ) -> Tuple[tf.Tensor, tf.Tensor]:
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
            attn_inputs = tf.concat([inputs[:, :self.embed_dim],
                                     cell_state.h], -1)
        else:
            attn_inputs = tf.concat([inputs[:, :self.embed_dim],
                                     cell_state], -1)

        # include slots in inputs but exclude previous action, since
        # rnn should get previous action from its hidden state
        rnn_inputs = inputs[:, :(self.embed_dim +
                                 self.embed_dim)]

        return rnn_inputs, attn_inputs

    def _create_attn_cell(self,
                          cell: tf.contrib.rnn.RNNCell,
                          embed_utter: tf.Tensor,
                          embed_prev_action: tf.Tensor,
                          real_length: tf.Tensor,
                          embed_for_no_intent: tf.Tensor,
                          embed_for_no_action: tf.Tensor,
                          embed_for_action_listen: tf.Tensor
                          ) -> tf.contrib.rnn.RNNCell:  # type:
        """Wrap cell in attention wrapper with given memory."""

        if self.attn_before_rnn:
            # create attention over previous user input
            num_memory_units_before_rnn = self._num_units(embed_utter)
            attn_mech = self._create_attn_mech(embed_utter, real_length)

            # create mask for empty user input not to pay attention to it
            ignore_mask = tf.reduce_all(tf.equal(tf.expand_dims(
                embed_for_no_intent, 0), embed_utter), -1)

            # do not use attention by location before rnn
            attn_shift_range = 0
        else:
            attn_mech = None
            ignore_mask = None
            num_memory_units_before_rnn = None
            attn_shift_range = None

        if self.attn_after_rnn:
            # create attention over previous bot actions
            attn_mech_after_rnn = self._create_attn_mech(embed_prev_action,
                                                         real_length)

            # create mask for empty bot action or action_listen
            # not to pay attention to them
            ignore_mask_listen = tf.logical_or(
                tf.reduce_all(tf.equal(
                    tf.expand_dims(embed_for_no_action, 0),
                    embed_prev_action), -1),
                tf.reduce_all(tf.equal(
                    tf.expand_dims(embed_for_action_listen, 0),
                    embed_prev_action), -1)
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
                self.cell_input_fn(inputs, attention,
                                   num_memory_units_before_rnn)),
            index_of_attn_to_copy=index_of_attn_to_copy,
            likelihood_fn=lambda emb_1, emb_2: (
                self._tf_sim(emb_1, emb_2, None)),
            tensor_not_to_copy=embed_for_action_listen,
            output_attention=True,
            alignment_history=True
        )

    def _create_tf_dial_embed(
            self,
            embed_utter: tf.Tensor,
            embed_slots: tf.Tensor,
            embed_prev_action: tf.Tensor,
            mask: tf.Tensor,
            embed_for_no_intent: tf.Tensor,
            embed_for_no_action: tf.Tensor,
            embed_for_action_listen: tf.Tensor
    ) -> Tuple[tf.Tensor, Union[tf.Tensor, 'TimeAttentionWrapperState']]:
        """Create rnn for dialogue level embedding."""

        cell_input = tf.concat([embed_utter, embed_slots,
                                embed_prev_action], -1)

        cell = self._create_rnn_cell()

        real_length = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

        if self.is_using_attention():
            cell = self._create_attn_cell(cell, embed_utter,
                                          embed_prev_action,
                                          real_length, embed_for_no_intent,
                                          embed_for_no_action,
                                          embed_for_action_listen)

        return tf.nn.dynamic_rnn(
            cell, cell_input,
            dtype=tf.float32,
            sequence_length=real_length,
            scope='rnn_decoder'
        )

    @staticmethod
    def _alignments_history_from(
        final_state: 'TimeAttentionWrapperState'
    ) -> tf.Tensor:
        """Extract alignments history form final rnn cell state."""

        alignments_from_state = final_state.alignment_history
        if not isinstance(alignments_from_state, tuple):
            alignments_from_state = [alignments_from_state]

        alignment_history = []
        for alignments in alignments_from_state:
            # reshape to (batch, time, memory_time)
            alignment_history.append(
                tf.transpose(alignments.stack(), [1, 0, 2]))

        return tf.concat(alignment_history, -1)

    @staticmethod
    def _all_time_masks_from(
        final_state: 'TimeAttentionWrapperState'
    ) -> tf.Tensor:
        """Extract all time masks form final rnn cell state."""

        # reshape to (batch, time, memory_time) and ignore last time
        # because time_mask is created for the next time step
        return tf.transpose(final_state.all_time_masks.stack(),
                            [1, 0, 2])[:, :-1, :]

    def _sims_rnn_to_max_from(self, cell_output: tf.Tensor) -> List[tf.Tensor]:
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

    def _embed_dialogue_from(self,
                             cell_output: tf.Tensor) -> tf.Tensor:
        """Extract or calculate dialogue level embedding from cell_output."""

        if self.attn_after_rnn:
            # embedding layer is inside rnn cell
            embed_dialogue = cell_output[:, :, :self.embed_dim]

            # extract additional debug tensors
            num_add = TimeAttentionWrapper.additional_output_size()
            self.rnn_embed = cell_output[
                :,
                :,
                self.embed_dim:(self.embed_dim + self.embed_dim)]
            self.attn_embed = cell_output[
                :,
                :,
                (self.embed_dim + self.embed_dim):-num_add]
        else:
            # add embedding layer to rnn cell output
            embed_dialogue = self._create_embed(
                cell_output[:, :, :self.rnn_size],
                layer_name_suffix='out'
            )
            if self.attn_before_rnn:
                # extract additional debug tensors
                self.attn_embed = cell_output[:, :, self.rnn_size:]

        return embed_dialogue

    def _tf_sim(self,
                embed_dialogue: tf.Tensor,
                embed_action: tf.Tensor,
                mask: Optional[tf.Tensor]
                ) -> Tuple[tf.Tensor, tf.Tensor]:
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

        if self.similarity_type == 'cosine':
            # normalize embedding vectors for cosine similarity
            embed_dialogue = tf.nn.l2_normalize(embed_dialogue, -1)
            embed_action = tf.nn.l2_normalize(embed_action, -1)

        if self.similarity_type in {'cosine', 'inner'}:

            if len(embed_dialogue.shape) == len(embed_action.shape):
                # calculate similarity between
                # two embedding vectors of the same size
                sim = tf.reduce_sum(embed_dialogue * embed_action, -1,
                                    keepdims=True)
                bin_sim = tf.where(sim > (self.mu_pos - self.mu_neg) / 2.0,
                                   tf.ones_like(sim),
                                   tf.zeros_like(sim))

                # output binary mask and similarity
                return bin_sim, sim

            else:
                # calculate similarity with several
                # embedded actions for the loss
                sim = tf.reduce_sum(tf.expand_dims(embed_dialogue, -2) *
                                    embed_action, -1)
                sim *= tf.expand_dims(mask, 2)

                sim_act = tf.reduce_sum(embed_action[:, :, :1, :] *
                                        embed_action[:, :, 1:, :], -1)
                sim_act *= tf.expand_dims(mask, 2)

                # output similarities between user input and bot actions
                # and similarities between bot actions
                return sim, sim_act

        else:
            raise ValueError("Wrong similarity type {}, "
                             "should be 'cosine' or 'inner'"
                             "".format(self.similarity_type))

    def _regularization_loss(self):
        # type: () -> Union[tf.Tensor, int]
        """Add regularization to the embed layer inside rnn cell."""

        if self.attn_after_rnn:
            return self.C2 * tf.add_n(
                [tf.nn.l2_loss(tf_var)
                 for tf_var in tf.trainable_variables()
                 if 'cell/out_layer/kernel' in tf_var.name]
            )
        else:
            return 0

    def _tf_loss(self,
                 sim: tf.Tensor,
                 sim_act: tf.Tensor,
                 sims_rnn_to_max: List[tf.Tensor],
                 mask: tf.Tensor
                 ) -> tf.Tensor:
        """Define loss."""

        # loss for maximizing similarity with correct action
        loss = tf.maximum(0., self.mu_pos - sim[:, :, 0])

        # loss for minimizing similarity with `num_neg` incorrect actions
        if self.use_max_sim_neg:
            # minimize only maximum similarity over incorrect actions
            max_sim_neg = tf.reduce_max(sim[:, :, 1:], -1)
            loss += tf.maximum(0., self.mu_neg + max_sim_neg)
        else:
            # minimize all similarities with incorrect actions
            max_margin = tf.maximum(0., self.mu_neg + sim[:, :, 1:])
            loss += tf.reduce_sum(max_margin, -1)

        if self.scale_loss_by_action_counts:
            # scale loss inverse proportionally to number of action counts
            loss *= self._loss_scales

        # penalize max similarity between intent embeddings
        loss_act = tf.maximum(0., tf.reduce_max(sim_act, -1))
        loss += loss_act * self.C_emb

        # maximize similarity returned by time attention wrapper
        for sim_to_add in sims_rnn_to_max:
            loss += tf.maximum(0., - sim_to_add + 1.)

        # mask loss for different length sequences
        loss *= mask
        # average the loss over sequence length
        loss = tf.reduce_sum(loss, -1) / tf.reduce_sum(mask, 1)

        # average the loss over the batch
        loss = (tf.reduce_mean(loss) +
                # add regularization losses
                self._regularization_loss() +
                tf.losses.get_regularization_loss())
        return loss

        # training methods

    def train(self,
              training_trackers: List[DialogueStateTracker],
              domain: Domain,
              **kwargs: Any
              ) -> None:
        """Train the policy on given training trackers."""

        logger.debug('Started training embedding policy.')

        # dealing with training data
        training_data = self.featurize_for_training(training_trackers,
                                                    domain,
                                                    **kwargs)
        # assume that characteristic time is the mean length of the dialogues
        self.characteristic_time = np.mean(training_data.true_length)
        if self.attn_shift_range is None:
            self.attn_shift_range = int(self.characteristic_time / 2)

        # encode all actions with policies' featurizer
        self.encoded_all_actions = \
            self.featurizer.state_featurizer.create_encoded_all_actions(
                domain)

        # check if number of negatives is less than number of actions
        logger.debug("Check if num_neg {} is smaller "
                     "than number of actions {}, "
                     "else set num_neg to the number of actions - 1"
                     "".format(self.num_neg, domain.num_actions))
        self.num_neg = min(self.num_neg, domain.num_actions - 1)

        # extract actual training data to feed to tf session
        session_data = self._create_tf_session_data(domain,
                                                    training_data.X,
                                                    training_data.y)

        self.graph = tf.Graph()

        with self.graph.as_default():
            dialogue_len = None  # use dynamic time for rnn
            # create placeholders
            self.a_in = tf.placeholder(
                dtype=tf.float32,
                shape=(None, dialogue_len,
                       session_data.X.shape[-1]),
                name='a'
            )
            self.b_in = tf.placeholder(
                dtype=tf.float32,
                shape=(None, dialogue_len,
                       None, session_data.Y.shape[-1]),
                name='b'
            )
            self.c_in = tf.placeholder(
                dtype=tf.float32,
                shape=(None, dialogue_len,
                       session_data.slots.shape[-1]),
                name='slt'
            )
            self.b_prev_in = tf.placeholder(
                dtype=tf.float32,
                shape=(None, dialogue_len,
                       session_data.Y.shape[-1]),
                name='b_prev'
            )
            self._dialogue_len = tf.placeholder(
                dtype=tf.int32,
                shape=(),
                name='dialogue_len'
            )
            self._x_for_no_intent_in = tf.placeholder(
                dtype=tf.float32,
                shape=(1, session_data.X.shape[-1]),
                name='x_for_no_intent'
            )
            self._y_for_no_action_in = tf.placeholder(
                dtype=tf.float32,
                shape=(1, session_data.Y.shape[-1]),
                name='y_for_no_action'
            )
            self._y_for_action_listen_in = tf.placeholder(
                dtype=tf.float32,
                shape=(1, session_data.Y.shape[-1]),
                name='y_for_action_listen'
            )
            self._is_training = tf.placeholder_with_default(False, shape=())

            self._loss_scales = tf.placeholder(dtype=tf.float32,
                                               shape=(None, dialogue_len))

            # create embedding vectors
            self.user_embed = self._create_tf_user_embed(self.a_in)
            self.bot_embed = self._create_tf_bot_embed(self.b_in)
            self.slot_embed = self._create_embed(self.c_in,
                                                 layer_name_suffix='slt')

            embed_prev_action = self._create_tf_bot_embed(self.b_prev_in)
            embed_for_no_intent = self._create_tf_no_intent_embed(
                self._x_for_no_intent_in)
            embed_for_no_action = self._create_tf_no_action_embed(
                self._y_for_no_action_in)
            embed_for_action_listen = self._create_tf_no_action_embed(
                self._y_for_action_listen_in)

            # mask different length sequences
            # if there is at least one `-1` it should be masked
            mask = tf.sign(tf.reduce_max(self.a_in, -1) + 1)

            # get rnn output
            cell_output, final_state = self._create_tf_dial_embed(
                self.user_embed, self.slot_embed, embed_prev_action, mask,
                embed_for_no_intent, embed_for_no_action,
                embed_for_action_listen
            )
            # process rnn output
            if self.is_using_attention():
                self.alignment_history = \
                    self._alignments_history_from(final_state)

                self.all_time_masks = self._all_time_masks_from(final_state)

            sims_rnn_to_max = self._sims_rnn_to_max_from(cell_output)
            self.dial_embed = self._embed_dialogue_from(cell_output)

            # calculate similarities
            self.sim_op, sim_act = self._tf_sim(self.dial_embed,
                                                self.bot_embed, mask)
            # construct loss
            loss = self._tf_loss(self.sim_op, sim_act, sims_rnn_to_max, mask)

            # define which optimizer to use
            self._train_op = tf.train.AdamOptimizer(
                learning_rate=0.001, epsilon=1e-16).minimize(loss)
            # train tensorflow graph
            self.session = tf.Session()

            self._train_tf(session_data, loss, mask)

    # training helpers
    def _linearly_increasing_batch_size(self, epoch: int) -> int:
        """Linearly increase batch size with every epoch.

        The idea comes from https://arxiv.org/abs/1711.00489.
        """

        if not isinstance(self.batch_size, list):
            return int(self.batch_size)

        if self.epochs > 1:
            return int(self.batch_size[0] +
                       epoch * (self.batch_size[1] -
                                self.batch_size[0]) / (self.epochs - 1))
        else:
            return int(self.batch_size[0])

    def _create_batch_b(self,
                        batch_pos_b: np.ndarray,
                        intent_ids: np.ndarray
                        ) -> np.ndarray:
        """Create batch of actions.

        The first is correct action
        and the rest are wrong actions sampled randomly.
        """

        batch_pos_b = batch_pos_b[:, :, np.newaxis, :]

        # sample negatives
        batch_neg_b = np.zeros((batch_pos_b.shape[0],
                                batch_pos_b.shape[1],
                                self.num_neg,
                                batch_pos_b.shape[-1]),
                               dtype=int)
        for b in range(batch_pos_b.shape[0]):
            for h in range(batch_pos_b.shape[1]):
                # create negative indexes out of possible ones
                # except for correct index of b
                negative_indexes = [
                    i
                    for i in range(self.encoded_all_actions.shape[0])
                    if i != intent_ids[b, h]
                ]

                negs = np.random.choice(negative_indexes, size=self.num_neg)

                batch_neg_b[b, h] = self.encoded_all_actions[negs]

        return np.concatenate([batch_pos_b, batch_neg_b], -2)

    # noinspection PyPep8Naming
    def _scale_loss_by_count_actions(self,
                                     X: np.ndarray,
                                     slots: np.ndarray,
                                     previous_actions: np.ndarray,
                                     actions_for_Y: np.ndarray
                                     ) -> Union[np.ndarray, List[List]]:
        """Calculate inverse proportionality of repeated actions."""

        if self.scale_loss_by_action_counts:
            full_X = np.concatenate([X, slots, previous_actions,
                                     actions_for_Y[:, :, np.newaxis]], -1)
            full_X = full_X.reshape((-1, full_X.shape[-1]))

            _, i, c = np.unique(full_X, return_inverse=True,
                                return_counts=True, axis=0)

            counts = c[i].reshape((X.shape[0], X.shape[1]))

            # do not include [-1 -1 ... -1 0] in averaging
            # and smooth it by taking sqrt
            return np.maximum(np.sqrt(np.mean(c[1:]) / counts), 1)
        else:
            return [[None]]

    def _train_tf(self,
                  session_data: SessionData,
                  loss: tf.Tensor,
                  mask: tf.Tensor) -> None:
        """Train tf graph."""

        self.session.run(tf.global_variables_initializer())

        if self.evaluate_on_num_examples:
            logger.info("Accuracy is updated every {} epochs"
                        "".format(self.evaluate_every_num_epochs))
        pbar = tqdm(range(self.epochs), desc="Epochs")
        train_acc = 0
        last_loss = 0
        for ep in pbar:
            # randomize training data for the current epoch
            ids = np.random.permutation(session_data.X.shape[0])

            # calculate batch size for the current epoch
            batch_size = self._linearly_increasing_batch_size(ep)
            # calculate number of batches in the current epoch
            batches_per_epoch = (
                session_data.X.shape[0] // batch_size +
                int(session_data.X.shape[0] % batch_size > 0))

            # collect average loss over the batches
            ep_loss = 0
            for i in range(batches_per_epoch):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                batch_ids = ids[start_idx:end_idx]

                # get randomized data for current batch
                batch_a = session_data.X[batch_ids]
                batch_pos_b = session_data.Y[batch_ids]
                actions_for_b = session_data.actions_for_Y[batch_ids]

                # add negatives - incorrect bot actions predictions
                batch_b = self._create_batch_b(batch_pos_b, actions_for_b)

                batch_c = session_data.slots[batch_ids]
                batch_b_prev = session_data.previous_actions[batch_ids]

                # calculate how much the loss from each action
                # should be scaled based on action rarity
                batch_loss_scales = self._scale_loss_by_count_actions(
                    batch_a, batch_c, batch_b_prev, actions_for_b)

                # minimize and calculate loss
                _loss, _ = self.session.run(
                    [loss, self._train_op],
                    feed_dict={
                        self.a_in: batch_a,
                        self.b_in: batch_b,
                        self.c_in: batch_c,
                        self.b_prev_in: batch_b_prev,
                        self._dialogue_len: session_data.X.shape[1],
                        self._x_for_no_intent_in:
                            session_data.x_for_no_intent,
                        self._y_for_no_action_in:
                            session_data.y_for_no_action,
                        self._y_for_action_listen_in:
                            session_data.y_for_action_listen,
                        self._is_training: True,
                        self._loss_scales: batch_loss_scales
                    }
                )
                # collect average loss over the batches
                ep_loss += _loss / batches_per_epoch

            # calculate train accuracy
            if self.evaluate_on_num_examples:
                if ((ep + 1) == 1 or
                        (ep + 1) % self.evaluate_every_num_epochs == 0 or
                        (ep + 1) == self.epochs):
                    train_acc = self._calc_train_acc(session_data, mask)
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
            logger.info("Finished training embedding policy, "
                        "loss={:.3f}, train accuracy={:.3f}"
                        "".format(last_loss, train_acc))

    def _calc_train_acc(self,
                        session_data: SessionData,
                        mask: tf.Tensor) -> np.float32:
        """Calculate training accuracy."""

        # choose n examples to calculate train accuracy
        n = self.evaluate_on_num_examples
        ids = np.random.permutation(len(session_data.X))[:n]
        # noinspection PyPep8Naming
        all_Y_d_x = np.stack([session_data.all_Y_d
                              for _ in range(session_data.X[ids].shape[0])])

        _sim, _mask = self.session.run(
            [self.sim_op, mask],
            feed_dict={
                self.a_in: session_data.X[ids],
                self.b_in: all_Y_d_x,
                self.c_in: session_data.slots[ids],
                self.b_prev_in: session_data.previous_actions[ids],
                self._dialogue_len: session_data.X.shape[1],
                self._x_for_no_intent_in:
                    session_data.x_for_no_intent,
                self._y_for_no_action_in:
                    session_data.y_for_no_action,
                self._y_for_action_listen_in:
                    session_data.y_for_action_listen
            }
        )
        return np.sum((np.argmax(_sim, -1) ==
                       session_data.actions_for_Y[ids]) *
                      _mask) / np.sum(_mask)

    def continue_training(self,
                          training_trackers: List[DialogueStateTracker],
                          domain: Domain,
                          **kwargs: Any) -> None:
        """Continue training an already trained policy."""

        batch_size = kwargs.get("batch_size", 5)
        epochs = kwargs.get("epochs", 50)

        for _ in range(epochs):
            training_data = self._training_data_for_continue_training(
                batch_size, training_trackers, domain)

            session_data = self._create_tf_session_data(domain,
                                                        training_data.X,
                                                        training_data.y)

            b = self._create_batch_b(session_data.Y,
                                     session_data.actions_for_Y)

            batch_loss_scales = self._scale_loss_by_count_actions(
                session_data.X,
                session_data.slots,
                session_data.previous_actions,
                session_data.actions_for_Y
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
                    self._x_for_no_intent_in:
                        session_data.x_for_no_intent,
                    self._y_for_no_action_in:
                        session_data.y_for_no_action,
                    self._y_for_action_listen_in:
                        session_data.y_for_action_listen,
                    self._is_training: True,
                    self._loss_scales: batch_loss_scales
                }
            )

    def predict_action_probabilities(self,
                                     tracker: DialogueStateTracker,
                                     domain: Domain) -> List[float]:
        """Predict the next action the bot should take.

        Return the list of probabilities for the next actions.
        """

        if self.session is None:
            logger.error("There is no trained tf.session: "
                         "component is either not trained or "
                         "didn't receive enough training data")
            return [0.0] * domain.num_actions

        # noinspection PyPep8Naming
        data_X = self.featurizer.create_X([tracker], domain)
        session_data = self._create_tf_session_data(domain, data_X)
        # noinspection PyPep8Naming
        all_Y_d_x = np.stack([session_data.all_Y_d
                              for _ in range(session_data.X.shape[0])])

        _sim = self.session.run(
            self.sim_op,
            feed_dict={
                self.a_in: session_data.X,
                self.b_in: all_Y_d_x,
                self.c_in: session_data.slots,
                self.b_prev_in: session_data.previous_actions,
                self._dialogue_len: session_data.X.shape[1],
                self._x_for_no_intent_in:
                    session_data.x_for_no_intent,
                self._y_for_no_action_in:
                    session_data.y_for_no_action,
                self._y_for_action_listen_in:
                    session_data.y_for_action_listen
            }
        )

        result = _sim[0, -1, :]
        if self.similarity_type == 'cosine':
            # clip negative values to zero
            result[result < 0] = 0
        elif self.similarity_type == 'inner':
            # normalize result to [0, 1] with softmax
            result = np.exp(result)
            result /= np.sum(result)

        return result.tolist()

    def _persist_tensor(self, name: Text, tensor: tf.Tensor) -> None:
        if tensor is not None:
            self.graph.clear_collection(name)
            self.graph.add_to_collection(name, tensor)

    def persist(self, path: Text) -> None:
        """Persists the policy to a storage."""

        if self.session is None:
            warnings.warn("Method `persist(...)` was called "
                          "without a trained model present. "
                          "Nothing to persist then!")
            return

        self.featurizer.persist(path)

        file_name = 'tensorflow_embedding.ckpt'
        checkpoint = os.path.join(path, file_name)
        utils.create_dir_for_file(checkpoint)

        with self.graph.as_default():
            self._persist_tensor('intent_placeholder',
                                 self.a_in)
            self._persist_tensor('action_placeholder',
                                 self.b_in)
            self._persist_tensor('slots_placeholder',
                                 self.c_in)
            self._persist_tensor('prev_act_placeholder',
                                 self.b_prev_in)
            self._persist_tensor('dialogue_len',
                                 self._dialogue_len)
            self._persist_tensor('x_for_no_intent',
                                 self._x_for_no_intent_in)
            self._persist_tensor('y_for_no_action',
                                 self._y_for_no_action_in)
            self._persist_tensor('y_for_action_listen',
                                 self._y_for_action_listen_in)

            self._persist_tensor('similarity_op', self.sim_op)

            self._persist_tensor('alignment_history',
                                 self.alignment_history)

            self._persist_tensor('user_embed', self.user_embed)
            self._persist_tensor('bot_embed', self.bot_embed)
            self._persist_tensor('slot_embed', self.slot_embed)
            self._persist_tensor('dial_embed', self.dial_embed)

            self._persist_tensor('rnn_embed', self.rnn_embed)
            self._persist_tensor('attn_embed', self.attn_embed)
            self._persist_tensor('copy_attn_debug', self.copy_attn_debug)

            self._persist_tensor('all_time_masks', self.all_time_masks)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        dump_path = os.path.join(path, file_name + ".encoded_all_actions.pkl")
        with io.open(dump_path, 'wb') as f:
            pickle.dump(self.encoded_all_actions, f)

    @staticmethod
    def load_tensor(name: Text) -> Optional[tf.Tensor]:
        tensor_list = tf.get_collection(name)
        return tensor_list[0] if tensor_list else None

    @classmethod
    def load(cls, path: Text) -> 'EmbeddingPolicy':
        """Loads a policy from the storage.

            **Needs to load its featurizer**"""

        if not os.path.exists(path):
            raise Exception("Failed to load dialogue model. Path {} "
                            "doesn't exist".format(os.path.abspath(path)))

        featurizer = TrackerFeaturizer.load(path)

        file_name = 'tensorflow_embedding.ckpt'
        checkpoint = os.path.join(path, file_name)

        if not os.path.exists(checkpoint + '.meta'):
            return cls(featurizer=featurizer)

        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            saver = tf.train.import_meta_graph(checkpoint + '.meta')

            saver.restore(sess, checkpoint)

            a_in = cls.load_tensor('intent_placeholder')
            b_in = cls.load_tensor('action_placeholder')
            c_in = cls.load_tensor('slots_placeholder')
            b_prev_in = cls.load_tensor('prev_act_placeholder')
            dialogue_len = cls.load_tensor('dialogue_len')
            x_for_no_intent = cls.load_tensor('x_for_no_intent')
            y_for_no_action = cls.load_tensor('y_for_no_action')
            y_for_action_listen = cls.load_tensor('y_for_action_listen')

            sim_op = cls.load_tensor('similarity_op')

            alignment_history = cls.load_tensor('alignment_history')

            user_embed = cls.load_tensor('user_embed')
            bot_embed = cls.load_tensor('bot_embed')
            slot_embed = cls.load_tensor('slot_embed')
            dial_embed = cls.load_tensor('dial_embed')

            rnn_embed = cls.load_tensor('rnn_embed')
            attn_embed = cls.load_tensor('attn_embed')
            copy_attn_debug = cls.load_tensor('copy_attn_debug')

            all_time_masks = cls.load_tensor('all_time_masks')

        encoded_actions_file = os.path.join(
            path, "{}.encoded_all_actions.pkl".format(file_name))

        with io.open(encoded_actions_file, 'rb') as f:
            encoded_all_actions = pickle.load(f)

        return cls(featurizer=featurizer,
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
                   all_time_masks=all_time_masks)
