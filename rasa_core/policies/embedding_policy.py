from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

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
    Any, List, Optional, Text)

from rasa_core import utils
from rasa_core.actions.action import ACTION_LISTEN_NAME
from rasa_core.featurizers import (
    TrackerFeaturizer, FullDialogueTrackerFeaturizer,
    LabelTokenizerSingleStateFeaturizer)
from rasa_core.policies import Policy

if typing.TYPE_CHECKING:
    from rasa_core.domain import Domain
    from rasa_core.trackers import DialogueStateTracker

try:
    import cPickle as pickle
except ImportError:
    import pickle

try:
    import tensorflow as tf
except ImportError:
    tf = None

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
    SUPPORTS_ONLINE_TRAINING = True

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
        "attn_before_rnn": False,
        # flag to use attention over prev bot actions
        # and copy it to output bypassing rnn
        "attn_after_rnn": False,

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

    @classmethod
    def _standard_featurizer(cls):
        return FullDialogueTrackerFeaturizer(
                LabelTokenizerSingleStateFeaturizer())

    @staticmethod
    def _check_tensorflow():
        if tf is None:
            raise ImportError("Failed to import `tensorflow`. "
                              "Please install `tensorflow`. "
                              "For example with `pip install tensorflow`.")

    def __init__(
            self,
            featurizer=None,  # type: Optional[FullDialogueTrackerFeaturizer]
            encoded_all_actions=None,  # type: Optional[np.ndarray]
            session=None,  # type: Optional[tf.Session]
            graph=None,  # type: Optional[tf.Graph]
            intent_placeholder=None,  # type: Optional[tf.Tensor]
            action_placeholder=None,  # type: Optional[tf.Tensor]
            slots_placeholder=None,  # type: Optional[tf.Tensor]
            prev_act_placeholder=None,  # type: Optional[tf.Tensor]
            dialogue_len=None,  # type: Optional[tf.Tensor]
            x_for_no_intent=None,  # type: Optional[tf.Tensor]
            y_for_no_action=None,  # type: Optional[tf.Tensor]
            y_for_action_listen=None,  # type: Optional[tf.Tensor]
            similarity_op=None,  # type: Optional[tf.Tensor]
            alignment_history=None,  # type: Optional[tf.Tensor]
            user_embed=None,  # type: Optional[tf.Tensor]
            bot_embed=None,  # type: Optional[tf.Tensor]
            slot_embed=None,  # type: Optional[tf.Tensor]
            dial_embed=None,  # type: Optional[tf.Tensor]
            rnn_embed=None,  # type: Optional[tf.Tensor]
            attn_embed=None,  # type: Optional[tf.Tensor]
            copy_attn_debug=None  # type: Optional[tf.Tensor]
    ):
        # type: (...) -> None
        self._check_tensorflow()
        if featurizer:
            if not isinstance(featurizer, FullDialogueTrackerFeaturizer):
                raise TypeError("Passed tracker featurizer of type {}, "
                                "should be FullDialogueTrackerFeaturizer."
                                "".format(type(featurizer).__name__))
        super(EmbeddingPolicy, self).__init__(featurizer)

        # flag if to use the same embeddings for user and bot
        try:
            self.share_embedding = \
                self.featurizer.state_featurizer.share_vocab
        except AttributeError:
            self.share_embedding = False

        self._load_params()

        # chrono initialization for forget bias
        self.characteristic_time = None

        # encode all actions with numbers
        # persist this array for prediction time
        self.encoded_all_actions = encoded_all_actions

        # tf related instances
        self.session = session
        self.graph = graph
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

        # internal tf instances
        self._train_op = None
        self._is_training = None
        self._loss_scales = None

        logger.info('---------------')
        logger.info(tf.get_default_graph())
        logger.info(len(tf.get_default_graph().get_operations()))
        logger.info(tf.get_default_session())
        logger.info(self.graph)
        logger.info(len(self.graph.get_operations()))
        logger.info(self.session)
        logger.info(self.session.graph)
        logger.info('---------------')

    # init helpers
    def _load_nn_architecture_params(self, config):

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

    def _load_embedding_params(self, config):
        self.embed_dim = config['embed_dim']
        self.mu_pos = config['mu_pos']
        self.mu_neg = config['mu_neg']
        self.similarity_type = config['similarity_type']
        self.num_neg = config['num_neg']
        self.use_max_sim_neg = config['use_max_sim_neg']

    def _load_regularization_params(self, config):
        self.C2 = config['C2']
        self.C_emb = config['C_emb']
        self.scale_loss_by_action_counts = \
            config['scale_loss_by_action_counts']
        self.droprate = {"a": config['droprate_a'],
                         "b": config['droprate_b'],
                         "rnn": config['droprate_rnn']}

    def _load_attn_params(self, config):
        self.sparse_attention = config['sparse_attention']
        self.attn_shift_range = config['attn_shift_range']
        self.attn_after_rnn = config['attn_after_rnn']
        self.attn_before_rnn = config['attn_before_rnn']

    def is_using_attention(self):
        return self.attn_after_rnn or self.attn_before_rnn

    def _load_visual_params(self, config):
        self.evaluate_every_num_epochs = config['evaluate_every_num_epochs']
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs

        self.evaluate_on_num_examples = config['evaluate_on_num_examples']

    def _load_params(self, **kwargs):
        config = copy.deepcopy(self.defaults)
        config.update(kwargs)

        self._load_nn_architecture_params(config)
        self._load_embedding_params(config)
        self._load_regularization_params(config)
        self._load_attn_params(config)
        self._load_visual_params(config)

    # data helpers
    # noinspection PyPep8Naming
    def _create_X_slots_previous_actions(self, data_X):
        """Extract feature vectors for user input (X), slots and
            previously executed actions from training data."""

        featurizer = self.featurizer.state_featurizer
        slot_start = featurizer.user_feature_len
        previous_start = slot_start + featurizer.slot_feature_len

        X = data_X[:, :, :slot_start]
        slots = data_X[:, :, slot_start:previous_start]
        previous_actions = data_X[:, :, previous_start:]

        return X, slots, previous_actions

    # noinspection PyPep8Naming
    @staticmethod
    def _actions_for_Y(data_Y):
        """Prepare Y data for training: extract actions indices."""
        return data_Y.argmax(axis=-1)

    # noinspection PyPep8Naming
    def _action_features_for_Y(self, actions_for_Y):
        """Prepare Y data for training: features for action labels."""
        return np.stack([np.stack([self.encoded_all_actions[action_idx]
                                   for action_idx in action_ids])
                         for action_ids in actions_for_Y])

    # noinspection PyPep8Naming
    @staticmethod
    def _create_zero_vector(X):
        """Create zero vector of shape (1, X.shape[-1])"""
        return np.zeros((1, X.shape[-1]), X.dtype)

    def _create_y_for_action_listen(self, domain):
        """Extract feature vector for action_listen"""
        action_listen_idx = domain.index_for_action(ACTION_LISTEN_NAME)
        return self.encoded_all_actions[action_listen_idx:
                                        action_listen_idx + 1]

    # noinspection PyPep8Naming
    def _create_all_Y_d(self, dialogue_len):
        """Stack encoded_all_intents on top of each other
            to create candidates for training examples
            to calculate training accuracy"""
        return np.stack([self.encoded_all_actions] * dialogue_len)

    # noinspection PyPep8Naming
    def _create_tf_session_data(self, domain, data_X, data_Y=None):
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
    def _create_tf_nn(self, x_in, layer_sizes, droprate, name):
        """Create nn with hidden layers and name"""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        x = tf.nn.relu(x_in)
        for i, layer_size in enumerate(layer_sizes):
            x = tf.layers.dense(inputs=x,
                                units=layer_size,
                                activation=tf.nn.relu,
                                kernel_regularizer=reg,
                                name='hidden_layer_{}_{}'.format(name, i),
                                reuse=tf.AUTO_REUSE)
            x = tf.layers.dropout(x, rate=droprate,
                                  training=self._is_training)
        return x

    def _create_embed(self, x, name):
        """Create dense embedding layer with a name."""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        emb_x = tf.layers.dense(inputs=x,
                                units=self.embed_dim,
                                activation=None,
                                kernel_regularizer=reg,
                                name='embed_layer_{}'.format(name),
                                reuse=tf.AUTO_REUSE)
        return emb_x

    def _create_tf_user_embed(self, a_in):
        """Create embedding user vector"""

        name = 'a_and_b' if self.share_embedding else 'a'

        a = self._create_tf_nn(
                a_in,
                self.hidden_layer_sizes['a'],
                self.droprate['a'],
                name=name
        )
        return self._create_embed(a, name=name)

    def _create_tf_bot_embed(self, b_in):
        """Create embedding bot vector"""

        name = 'a_and_b' if self.share_embedding else 'b'

        b = self._create_tf_nn(
                b_in,
                self.hidden_layer_sizes['b'],
                self.droprate['b'],
                name=name
        )
        return self._create_embed(b, name=name)

    def _create_tf_no_intent_embed(self, x_for_no_intent_i):
        """Create embedding user vector for empty intent"""

        name = 'a_and_b' if self.share_embedding else 'a'

        x_for_no_intent = self._create_tf_nn(
                x_for_no_intent_i,
                self.hidden_layer_sizes['a'],
                droprate=0,
                name=name
        )
        return tf.stop_gradient(
                self._create_embed(x_for_no_intent, name=name))

    def _create_tf_no_action_embed(self, y_for_no_action_in):
        """Create embedding bot vector for empty action and action_listen"""

        name = 'a_and_b' if self.share_embedding else 'b'

        y_for_no_action = self._create_tf_nn(
                y_for_no_action_in,
                self.hidden_layer_sizes['b'],
                droprate=0,
                name=name
        )
        return tf.stop_gradient(
                self._create_embed(y_for_no_action, name=name))

    def _create_rnn_cell(self):
        """Create one rnn cell"""

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
    def num_mem_units(memory):
        return memory.shape[-1].value

    def _create_attn_mech(self, memory, real_length):
        attn_mech = tf.contrib.seq2seq.BahdanauAttention(
                num_units=self.num_mem_units(memory),
                memory=memory,
                memory_sequence_length=real_length,
                normalize=True,
                probability_fn=tf.identity,
                # we only attend to memory up to a current time
                # it does not affect alignments, but
                # is important for interpolation gate
                score_mask_value=0
        )
        return attn_mech

    def _create_attn_cell(self, cell, emb_utter, emb_prev_act,
                          real_length, emb_for_no_intent,
                          emb_for_no_action, emb_for_action_listen):
        """Wrap cell in attention wrapper with given memory"""

        if self.attn_before_rnn:
            # create attention over previous user input
            num_mem_units = self.num_mem_units(emb_utter)
            attn_mech = self._create_attn_mech(emb_utter, real_length)

            # create mask for empty user input not to pay attention to it
            ignore_mask = tf.reduce_all(tf.equal(tf.expand_dims(
                    emb_for_no_intent, 0), emb_utter), -1)

            # do not use attention by location before rnn
            attn_shift_range = 0
        else:
            attn_mech = None
            ignore_mask = None
            num_mem_units = 0
            attn_shift_range = None

        if self.attn_after_rnn:
            # create attantion over previous bot actions
            attn_mech_after_rnn = self._create_attn_mech(emb_prev_act,
                                                         real_length)

            # create mask for empty bot action or action_listen
            # not to pay attention to them
            ignore_mask_listen = tf.logical_or(
                    tf.reduce_all(tf.equal(tf.expand_dims(
                            emb_for_no_action, 0), emb_prev_act), -1),
                    tf.reduce_all(tf.equal(tf.expand_dims(
                            emb_for_action_listen, 0), emb_prev_act), -1)
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

        num_utter_units = self.num_mem_units(emb_utter)

        def cell_input_fn(inputs, attention):
            """Combine rnn inputs and attention into cell input"""
            if num_mem_units > 0:
                if num_mem_units == num_utter_units:
                    # since attention can contain additional
                    # attention mechanisms, only attention
                    # from previous user input is used as an input
                    # for rnn cell and only if memory before rnn
                    # is the same size as emb_utter
                    return tf.concat([inputs[:, :num_utter_units] +
                                      attention[:, :num_utter_units],
                                      inputs[:, num_utter_units:]], -1)
                else:
                    # in current implementation it cannot fall here,
                    # but this Exception exists in case
                    # attention before rnn is changed
                    raise ValueError("Number of memory units {} is not "
                                     "equal to number of utter units {}. "
                                     "Please modify cell input function "
                                     "accordingly.".format(num_mem_units,
                                                           num_utter_units))
            else:
                return inputs

        # noinspection PyUnusedLocal
        def attn_input_fn(inputs, cell_state):
            """Construct rnn input and attention mechanism input"""

            # the hidden state and slots are not included,
            # in hope that algorithm would learn correct attention
            # regardless of the hidden state of an rnn and slots
            prev_out_for_attn = tf.concat([inputs[:, :num_utter_units],
                                           inputs[:, (num_utter_units +
                                                      num_utter_units):]], 1)

            # include slots in inputs but exclude previous action, since
            # rnn should get previous action from its hidden state
            inputs = inputs[:, :(num_utter_units +
                                 num_utter_units)]

            return inputs, prev_out_for_attn

        attn_cell = TimeAttentionWrapper(
                cell=cell,
                attention_mechanism=attn_mech,
                dialogue_len=self._dialogue_len,
                attn_shift_range=attn_shift_range,
                sparse_attention=self.sparse_attention,
                cell_input_fn=cell_input_fn,
                attn_input_fn=attn_input_fn,
                index_of_attn_to_copy=index_of_attn_to_copy,
                similarity_fn=lambda emb_1, emb_2: (
                    self._tf_sim(emb_1, emb_2, None)),
                ignore_mask=ignore_mask,
                emb_for_action_listen=emb_for_action_listen,
                output_attention=True,
                alignment_history=True
        )
        return attn_cell

    def _alignments_history_from(self, final_state):
        """Extract alignments history form final rnn cell state"""

        if not self.is_using_attention():
            return None

        alignments_from_state = final_state.alignment_history
        if not isinstance(alignments_from_state, tuple):
            alignments_from_state = [alignments_from_state]

        alignment_history = []
        for alignments in alignments_from_state:
            # reshape to (batch, time, memory_time)
            alignment_history.append(
                    tf.transpose(alignments.stack(), [1, 0, 2]))

        return tf.concat(alignment_history, -1)

    def _sim_rnn_to_max_from(self, cell_output):
        """Save intermediate tensors for debug purposes"""

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

    def _emb_dial_from(self, cell_output):
        """Extract or calculate dialogue level embedding from cell_output"""

        if self.attn_after_rnn:
            # embedding layer is inside rnn cell
            emb_dial = cell_output[:, :, :self.embed_dim]

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
            emb_dial = self._create_embed(cell_output[:, :, :self.rnn_size],
                                          name='out')

            if self.attn_before_rnn:
                # extract additional debug tensors
                self.attn_embed = cell_output[:, :, self.rnn_size:]

        return emb_dial

    def _create_tf_dial_embed(self, emb_utter, emb_slots, emb_prev_act, mask,
                              emb_for_no_intent, emb_for_no_action,
                              emb_for_action_listen):
        """Create rnn for dialogue level embedding"""

        cell_input = tf.concat([emb_utter, emb_slots, emb_prev_act], -1)

        cell = self._create_rnn_cell()

        real_length = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

        if self.is_using_attention():
            cell = self._create_attn_cell(cell, emb_utter, emb_prev_act,
                                          real_length, emb_for_no_intent,
                                          emb_for_no_action,
                                          emb_for_action_listen)

        return tf.nn.dynamic_rnn(
                cell, cell_input,
                dtype=tf.float32,
                sequence_length=real_length,
                scope='rnn_decoder'
        )

    def _tf_sim(self, emb_dial, emb_act, mask):
        """Define similarity
            this method has two roles:
            - calculate similarity between
                two embedding vectors of the same size
                and output binary mask and similarity
            - calculate similarity with several embedded actions for the loss
                and output similarities between user input and bot actions
                and similarities between bot actions

            They are kept in the same helper method,
            because it is necessary for them to be mathematically identical
        """

        if self.similarity_type == 'cosine':
            # normalize embedding vectors for cosine similarity
            emb_dial = tf.nn.l2_normalize(emb_dial, -1)
            emb_act = tf.nn.l2_normalize(emb_act, -1)

        if self.similarity_type in {'cosine', 'inner'}:

            if len(emb_dial.shape) == len(emb_act.shape):
                # calculate similarity between
                # two embedding vectors of the same size
                sim = tf.reduce_sum(emb_dial * emb_act, -1, keepdims=True)
                bin_sim = tf.where(sim > (self.mu_pos - self.mu_neg) / 2.0,
                                   tf.ones_like(sim),
                                   tf.zeros_like(sim))

                # output binary mask and similarity
                return bin_sim, sim

            else:
                # calculate similarity with several
                # embedded actions for the loss
                sim = tf.reduce_sum(tf.expand_dims(emb_dial, -2) *
                                    emb_act, -1)
                sim *= tf.expand_dims(mask, 2)

                sim_act = tf.reduce_sum(emb_act[:, :, :1, :] *
                                        emb_act[:, :, 1:, :], -1)
                sim_act *= tf.expand_dims(mask, 2)

                # output similarities between user input and bot actions
                # and similarities between bot actions
                return sim, sim_act

        else:
            raise ValueError("Wrong similarity type {}, "
                             "should be 'cosine' or 'inner'"
                             "".format(self.similarity_type))

    def _regularization_loss(self):
        """Add regularization to the embed layer inside rnn cell"""
        if self.attn_after_rnn:
            return self.C2 * tf.add_n(
                    [tf.nn.l2_loss(tf_var)
                     for tf_var in tf.trainable_variables()
                     if 'cell/out_layer/kernel' in tf_var.name]
            )
        else:
            return 0

    def _tf_loss(self, sim, sim_act, sim_rnn_to_max, mask):
        """Define loss"""

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
        for sim_to_add in sim_rnn_to_max:
            loss += tf.maximum(0., 1. - sim_to_add)

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
              training_trackers,  # type: List[DialogueStateTracker]
              domain,  # type: Domain
              **kwargs  # type: **Any
              ):
        # type: (...) -> None
        """Trains the policy on given training trackers."""

        logger.debug('Started training embedding policy.')

        logger.info('---------------')
        logger.info(tf.get_default_graph())
        logger.info(len(tf.get_default_graph().get_operations()))
        logger.info(tf.get_default_session())
        logger.info(self.graph)
        logger.info(len(self.graph.get_operations()))
        logger.info(self.session)
        logger.info(self.session.graph)
        logger.info('---------------')

        if kwargs:
            logger.debug("Config is updated with {}".format(kwargs))
            self._load_params(**kwargs)

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
            self.slot_embed = self._create_embed(self.c_in, name='slt')

            emb_prev_act = self._create_tf_bot_embed(self.b_prev_in)
            emb_for_no_intent = self._create_tf_no_intent_embed(
                    self._x_for_no_intent_in)
            emb_for_no_action = self._create_tf_no_action_embed(
                    self._y_for_no_action_in)
            emb_for_action_listen = self._create_tf_no_action_embed(
                    self._y_for_action_listen_in)

            # mask different length sequences
            # if there is at least one `-1` it should be masked
            mask = tf.sign(tf.reduce_max(self.a_in, -1) + 1)

            # get rnn output
            cell_output, final_state = self._create_tf_dial_embed(
                    self.user_embed, self.slot_embed, emb_prev_act, mask,
                    emb_for_no_intent, emb_for_no_action,
                    emb_for_action_listen
            )
            # process rnn output
            self.alignment_history = \
                self._alignments_history_from(final_state)

            sim_rnn_to_max = self._sim_rnn_to_max_from(cell_output)
            self.dial_embed = self._emb_dial_from(cell_output)

            # calculate similarities
            self.sim_op, sim_act = self._tf_sim(self.dial_embed,
                                                self.bot_embed, mask)
            # construct loss
            loss = self._tf_loss(self.sim_op, sim_act, sim_rnn_to_max, mask)

            # define which optimizer to use
            self._train_op = tf.train.AdamOptimizer(
                    learning_rate=0.001, epsilon=1e-16).minimize(loss)
            # train tensorflow graph
            self.session = tf.Session()

            self._train_tf(session_data, loss, mask)

        logger.info('---------------')
        logger.info(tf.get_default_graph())
        logger.info(len(tf.get_default_graph().get_operations()))
        logger.info(tf.get_default_session())
        logger.info(self.graph)
        logger.info(len(self.graph.get_operations()))
        logger.info(self.session)
        logger.info(self.session.graph)
        logger.info('---------------')

    # training helpers
    def _linearly_increasing_batch_size(self, epoch):
        """Linearly increase batch size with every epoch.
            The idea comes from https://arxiv.org/abs/1711.00489"""
        if not isinstance(self.batch_size, list):
            return int(self.batch_size)

        if self.epochs > 1:
            return int(self.batch_size[0] +
                       epoch * (self.batch_size[1] -
                                self.batch_size[0]) / (self.epochs - 1))
        else:
            return int(self.batch_size[0])

    def _create_batch_b(self, batch_pos_b, intent_ids):
        """Create batch of actions, where the first is correct action
            and the rest are wrong actions sampled randomly"""

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
                        i for i in range(self.encoded_all_actions.shape[0])
                        if i != intent_ids[b, h]
                ]

                negs = np.random.choice(negative_indexes, size=self.num_neg)

                batch_neg_b[b, h] = self.encoded_all_actions[negs]

        return np.concatenate([batch_pos_b, batch_neg_b], -2)

    # noinspection PyPep8Naming
    def _scale_loss_by_count_actions(self, X, slots,
                                     previous_actions, actions_for_Y):
        """Count number of repeated actions and
            output inverse proportionality"""
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

    def _train_tf(self, session_data, loss, mask):
        """Train tf graph"""

        self.session.run(tf.global_variables_initializer())

        if self.evaluate_on_num_examples:
            logger.info("Accuracy is updated every {} epochs"
                        "".format(self.evaluate_every_num_epochs))
        pbar = tqdm(range(self.epochs), desc="Epochs")
        train_acc = 0
        last_loss = 0
        for ep in pbar:
            ids = np.random.permutation(session_data.X.shape[0])

            batch_size = self._linearly_increasing_batch_size(ep)
            batches_per_epoch = (
                    session_data.X.shape[0] // batch_size +
                    int(session_data.X.shape[0] % batch_size > 0))

            ep_loss = 0
            for i in range(batches_per_epoch):
                start_idx = i * batch_size
                end_idx = (i + 1) * batch_size
                batch_ids = ids[start_idx:end_idx]

                batch_a = session_data.X[batch_ids]
                batch_pos_b = session_data.Y[batch_ids]
                actions_for_b = session_data.actions_for_Y[batch_ids]
                # add negatives
                batch_b = self._create_batch_b(batch_pos_b, actions_for_b)

                batch_c = session_data.slots[batch_ids]
                batch_b_prev = session_data.previous_actions[batch_ids]

                batch_loss_scales = self._scale_loss_by_count_actions(
                        batch_a, batch_c, batch_b_prev, actions_for_b)

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
                ep_loss += _loss / batches_per_epoch

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

    def _calc_train_acc(self, session_data, mask):
        """Calculate training accuracy"""

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

    def continue_training(self, training_trackers, domain, **kwargs):
        # type: (List[DialogueStateTracker], Domain, **Any) -> None
        """Continues training an already trained policy."""

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

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        """Predicts the next action the bot should take
            after seeing the tracker.

            Returns the list of probabilities for the next actions"""

        logger.info('---------------')
        logger.info(tf.get_default_graph())
        logger.info(len(tf.get_default_graph().get_operations()))
        logger.info(tf.get_default_session())
        logger.info(self.graph)
        logger.info(len(self.graph.get_operations()))
        logger.info(self.session)
        logger.info(self.session.graph)
        logger.info('---------------')

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

    def _persist_tensor(self, name, tensor):
        if tensor is not None:
            self.graph.clear_collection(name)
            self.graph.add_to_collection(name, tensor)

    def persist(self, path):
        # type: (Text) -> None
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

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with io.open(os.path.join(
                path,
                file_name + ".encoded_all_actions.pkl"), 'wb') as f:
            pickle.dump(self.encoded_all_actions, f)

    @staticmethod
    def load_tensor(name):
        tensor_list = tf.get_collection(name)
        return tensor_list[0] if tensor_list else None

    @classmethod
    def load(cls, path):
        # type: (Text) -> EmbeddingPolicy
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

        encoded_actions_file = os.path.join(
                path, "{}.encoded_all_actions.pkl".format(file_name))

        with io.open(encoded_actions_file, 'rb') as f:
            encoded_all_actions = pickle.load(f)

        return cls(featurizer=featurizer,
                   encoded_all_actions=encoded_all_actions,
                   session=sess,
                   graph=graph,
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
                   copy_attn_debug=copy_attn_debug)


# Attentional interface
class TimedNTM(object):
    """Timed Neural Turing Machine
      paper:
        https://arxiv.org/pdf/1410.5401.pdf
      implementation inspired by:
        https://github.com/carpedm20/NTM-tensorflow/blob/master/ntm_cell.py

      ::param int attn_shift_range:
        a time range within which to attend to the memory by location
      ::param bool sparse_attention:
        a flag if use sparsemax instead of softmax for probs or gate if None
    """

    def __init__(self, attn_shift_range, sparse_attention, name):
        # interpolation gate
        self.name = 'timed_ntm_' + name

        self.inter_gate = tf.layers.Dense(
                units=1,
                activation=tf.sigmoid,
                name=self.name + '/inter_gate'
        )
        # if use sparsemax instead of softmax for probs
        self.sparse_attention = sparse_attention

        # shift weighting if range is provided
        if attn_shift_range:
            self.shift_weight = tf.layers.Dense(
                    units=2 * attn_shift_range + 1,
                    activation=tf.nn.softmax,
                    name=self.name + '/shift_weight'
            )
        else:
            self.shift_weight = None

        # sharpening parameter
        self.gamma_sharp = tf.layers.Dense(
                units=1,
                activation=lambda a: tf.nn.softplus(a) + 1,
                bias_initializer=tf.constant_initializer(1),
                name=self.name + '/gamma_sharp'
        )

    def __call__(self, cell_output, scores, scores_state, ignore_mask):
        # apply exponential moving average with interpolation gate weight
        # to scores from previous time which are equal to probs at this point
        # different from original NTM where it is applied after softmax
        i_g = self.inter_gate(cell_output)

        # scores limited by time
        scores = tf.concat([i_g * scores[:, :-1] + (1 - i_g) * scores_state,
                            scores[:, -1:]], 1)
        next_scores_state = scores

        # create probabilities for attention
        if self.sparse_attention:
            probs = tf.contrib.sparsemax.sparsemax(scores)
        else:
            probs = tf.nn.softmax(scores)

        if self.shift_weight is not None:
            s_w = self.shift_weight(cell_output)

            # we want to go back in time during convolution
            conv_probs = tf.reverse(probs, axis=[1])

            # preare probs for tf.nn.depthwise_conv2d
            # [in_width, in_channels=batch]
            conv_probs = tf.transpose(conv_probs, [1, 0])
            # [batch=1, in_height=1, in_width=time+1, in_channels=batch]
            conv_probs = conv_probs[tf.newaxis, tf.newaxis, :, :]

            # [filter_height=1, filter_width=2*attn_shift_range+1,
            #   in_channels=batch, channel_multiplier=1]
            conv_s_w = tf.transpose(s_w, [1, 0])
            conv_s_w = conv_s_w[tf.newaxis, :, :, tf.newaxis]

            # perform 1d convolution
            # [batch=1, out_height=1, out_width=time+1, out_channels=batch]
            conv_probs = tf.nn.depthwise_conv2d_native(conv_probs, conv_s_w,
                                                       [1, 1, 1, 1], 'SAME')
            conv_probs = conv_probs[0, 0, :, :]
            conv_probs = tf.transpose(conv_probs, [1, 0])

            probs = tf.reverse(conv_probs, axis=[1])

        # Sharpening
        g_sh = self.gamma_sharp(cell_output)

        powed_probs = tf.pow(probs, g_sh)
        probs = powed_probs / (
                tf.reduce_sum(powed_probs, 1, keepdims=True) + 1e-32)

        # set probs for no intents and action_listens to zero
        if ignore_mask is not None:
            probs = tf.concat([tf.where(ignore_mask,
                                        tf.zeros_like(probs[:, :-1]),
                                        probs[:, :-1]),
                               probs[:, -1:]], 1)
        return probs, next_scores_state


def _compute_time_attention(attention_mechanism, cell_output, attention_state,
                            # time is added to calculate time attention
                            time, timed_ntm, ignore_mask, attention_layer):
    """Computes the attention and alignments limited by time
        for a given attention_mechanism.

        Modified helper method form tensorflow."""

    scores, _ = attention_mechanism(cell_output, state=attention_state)

    # take only scores form current and past times
    timed_scores = scores[:, :time + 1]
    timed_scores_state = attention_state[:, :time]
    if ignore_mask is not None:
        timed_ignore_mask = ignore_mask[:, :time]
    else:
        timed_ignore_mask = None

    # pass these scores to NTM
    probs, next_scores_state = timed_ntm(cell_output, timed_scores,
                                         timed_scores_state,
                                         timed_ignore_mask)

    # concatenate probs with zeros to get new alignments
    zeros = tf.zeros_like(scores)
    # remove current time from attention
    alignments = tf.concat([probs[:, :-1], zeros[:, time:]], 1)

    # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
    expanded_alignments = tf.expand_dims(alignments, 1)

    # Context is the inner product of alignments and values along the
    # memory time dimension.
    # alignments shape is
    #   [batch_size, 1, memory_time]
    # attention_mechanism.values shape is
    #   [batch_size, memory_time, memory_size]
    # the batched matmul is over memory_time, so the output shape is
    #   [batch_size, 1, memory_size].
    # we then squeeze out the singleton dim.
    context = tf.matmul(expanded_alignments, attention_mechanism.values)
    context = tf.squeeze(context, [1])

    if attention_layer is not None:
        attention = attention_layer(tf.concat([cell_output, context], 1))
    else:
        attention = context

    # return current time to attention
    alignments = tf.concat([probs, zeros[:, time + 1:]], 1)
    next_attention_state = tf.concat([next_scores_state,
                                      zeros[:, time + 1:]], 1)
    return attention, alignments, next_attention_state


class TimeAttentionWrapperState(namedtuple(
                "TimeAttentionWrapperState",
                tf.contrib.seq2seq.AttentionWrapperState._fields +
                ("all_hidden_cell_states",))  # added
    ):
    """Modified  from tensorflow's tf.contrib.seq2seq.AttentionWrapperState
        see there for description of the parameters"""

    def clone(self, **kwargs):
        """Copied  from tensorflow's tf.contrib.seq2seq.AttentionWrapperState
            see there for description of the parameters"""

        def with_same_shape(old, new):
            """Check and set new tensor's shape."""
            if isinstance(old, tf.Tensor) and isinstance(new, tf.Tensor):
                return tf.contrib.framework.with_same_shape(old, new)
            return new

        return tf.contrib.framework.nest.map_structure(
                with_same_shape,
                self,
                super(TimeAttentionWrapperState, self)._replace(**kwargs)
        )


class TimeAttentionWrapper(tf.contrib.seq2seq.AttentionWrapper):
    """Custom AttentionWrapper that takes into account time
        when calculating attention.
        Attention is calculated before calling rnn cell.

        Modified from tensorflow's tf.contrib.seq2seq.AttentionWrapper.

        Additional args:
            attn_shift_range: Python int (`0` by default),
                a time range within which to attend to the memory by location
                by Neural Turing Machine.
            sparse_attention: Python bool,
                flag to use sparsemax (if `True`) instead of
                softmax (if `False`, default) for probabilities
            attn_to_copy_fn: (optional) a `callable`.
                a function that picks which part of attention tensor
                to use for copying to output, the default is `None`, which
                turns off copying mechanism.
                Copy inpired by: https://arxiv.org/pdf/1603.06393.pdf
            output_attention: Python bool.  If `True`, the output at each
                time step is the concatenated cell outputs, attention values
                and ..., used in copy mechanism.

        See the super class for other arguments description"""

    def __init__(self, cell,
                 attention_mechanism,
                 dialogue_len,
                 attn_shift_range=None,
                 sparse_attention=False,
                 attention_layer_size=None,
                 alignment_history=False,
                 attn_input_fn=None,
                 cell_input_fn=None,
                 index_of_attn_to_copy=None,
                 similarity_fn=None,
                 ignore_mask=None,
                 emb_for_action_listen=None,
                 output_attention=False,
                 initial_cell_state=None,
                 name=None):
        super(TimeAttentionWrapper, self).__init__(
                cell,
                attention_mechanism,
                attention_layer_size,
                alignment_history,
                cell_input_fn,
                output_attention,
                initial_cell_state,
                name
        )
        self._attn_input_fn = attn_input_fn

        self._dialogue_len = dialogue_len
        if not isinstance(attn_shift_range, list):
            attn_shift_range = [attn_shift_range]
        self._timed_ntms = [TimedNTM(attn_shift_range[0],
                                     sparse_attention,
                                     name='0')]
        if self._is_multi:
            for i in range(1, len(attention_mechanism)):
                if len(attn_shift_range) < i + 1:
                    attn_shift_range.append(attn_shift_range[-1])
                self._timed_ntms.append(TimedNTM(attn_shift_range[i],
                                                 sparse_attention,
                                                 name=str(i)))
        self._sim_fn = similarity_fn
        self._emb_for_action_listen = emb_for_action_listen
        if not isinstance(ignore_mask, list):
            self._ignore_mask = [ignore_mask]
        else:
            self._ignore_mask = ignore_mask

        self._index_of_attn_to_copy = index_of_attn_to_copy

    @staticmethod
    def additional_output_size():
        """Number of additional outputs:
          similarities:
              sim_attn_to_max, sim_state_to_max
          debugging info:
              current_time_prob, bin_sim_listen, bin_sim

          **Method should be static**
        """
        return 2 + 3

    @property
    def output_size(self):
        if self._output_attention:
            if self._index_of_attn_to_copy is not None:
                # output both raw rnn cell_output and
                # cell_output with copied attention
                # together with attention vector itself
                # and additional output
                return (2 * self._cell.output_size +
                        self._attention_layer_size +
                        self.additional_output_size())
            else:
                return self._cell.output_size + self._attention_layer_size
        else:
            return self._cell.output_size

    @property
    def state_size(self):
        """The `state_size` property of `TimeAttentionWrapper`.
        Returns:
          A `TimeAttentionWrapperState` tuple containing shapes used by this
          object.
        """

        # use AttentionWrapperState from superclass
        state_size = super(TimeAttentionWrapper, self).state_size

        all_hidden_cell_states = (self._cell.state_size
                                  if self._index_of_attn_to_copy is not None
                                  else None)

        return TimeAttentionWrapperState(
                cell_state=state_size.cell_state,
                time=state_size.time,
                attention=state_size.attention,
                alignments=state_size.alignments,
                attention_state=state_size.attention_state,
                alignment_history=state_size.alignment_history,
                all_hidden_cell_states=all_hidden_cell_states)

    def zero_state(self, batch_size, dtype):
        """Modified  from tensorflow's zero_state
            see there for description of the parameters"""

        # use AttentionWrapperState from superclass
        zero_state = super(TimeAttentionWrapper,
                           self).zero_state(batch_size, dtype)

        with tf.name_scope(type(self).__name__ + "ZeroState",
                           values=[batch_size]):
            if self._index_of_attn_to_copy is not None:
                # store all cell states into a tensor array to allow
                # copy mechanism to go back in time
                if isinstance(self._cell.state_size,
                              tf.contrib.rnn.LSTMStateTuple):
                    all_hidden_cell_states = tf.contrib.rnn.LSTMStateTuple(
                            tf.TensorArray(dtype, size=self._dialogue_len + 1,
                                           dynamic_size=False,
                                           clear_after_read=False
                                           ).write(0, zero_state.cell_state.c),
                            tf.TensorArray(dtype, size=self._dialogue_len + 1,
                                           dynamic_size=False,
                                           clear_after_read=False
                                           ).write(0, zero_state.cell_state.h)
                    )
                else:
                    all_hidden_cell_states = tf.TensorArray(
                            dtype, size=0,
                            dynamic_size=False,
                            clear_after_read=False
                    ).write(0, zero_state.cell_state)
            else:
                # do not waste resources on storing history
                all_hidden_cell_states = None

            return TimeAttentionWrapperState(
                    cell_state=zero_state.cell_state,
                    time=zero_state.time,
                    attention=zero_state.attention,
                    alignments=zero_state.alignments,
                    attention_state=zero_state.attention_state,
                    alignment_history=zero_state.alignment_history,
                    all_hidden_cell_states=all_hidden_cell_states
            )

    # helper methods for copy mechanism
    def _get_memory_probs(self, all_alignments, time):
        """Helper method to  get memory_probs from all_alignments"""

        memory_probs = tf.stop_gradient(all_alignments[
                self._index_of_attn_to_copy][:, :time])
        # set memory_probs for action_listens to zeros in history
        memory_probs = tf.where(
                self._ignore_mask[self._index_of_attn_to_copy][:, :time],
                tf.zeros_like(memory_probs),
                memory_probs
        )
        # binarize memory_probs
        memory_probs_max = tf.reduce_max(memory_probs, axis=1, keepdims=True)
        memory_probs_max = tf.where(memory_probs_max > 0.1,
                                    memory_probs_max, -memory_probs_max)

        return tf.where(tf.equal(memory_probs, memory_probs_max),
                        tf.ones_like(memory_probs),
                        tf.zeros_like(memory_probs))

    @staticmethod
    def _history_alignments(memory_probs):
        """Helper method to apply binary mask to memory_probs"""

        current_time_prob = 1 - tf.reduce_sum(memory_probs, 1, keepdims=True)
        return tf.concat([memory_probs, current_time_prob], 1)

    @staticmethod
    def _apply_alignments_to_history(alignments, history_states, state):
        """Helper method to apply attention probabilities to  rnn history"""

        expanded_alignments = tf.stop_gradient(tf.expand_dims(alignments, 1))

        history_states = tf.concat([history_states,
                                    tf.expand_dims(state, 1)], 1)

        # Context is the inner product of alignments and values along the
        # memory time dimension.
        # expanded_alignments shape is
        #   [batch_size, 1, memory_time]
        # history_states shape is
        #   [batch_size, memory_time, memory_size]
        # the batched matmul is over memory_time, so the output shape is
        #   [batch_size, 1, memory_size].
        # we then squeeze out the singleton dim.

        return tf.squeeze(tf.matmul(expanded_alignments, history_states), [1])

    def _prev_emb_action(self, state, alignments, time):
        """Helper method to get previous embedding vector from history"""

        # get all previous embedding vectors
        # from appropriate attention mechanism limited by current time
        prev_emb_acts = tf.stop_gradient(self._attention_mechanisms[
                self._index_of_attn_to_copy].values[:, :time, :])

        # multiply by alignments to get one vector from one time step
        return self._apply_alignments_to_history(alignments,
                                                 prev_emb_acts,
                                                 state)

    def _additional_sim(self, emb_vector, prev_emb_act, current_time_prob):
        """Helper method to create additional similarities to maximize"""

        _, sim_add = self._sim_fn(emb_vector,
                                  tf.stop_gradient(prev_emb_act))
        return tf.where(current_time_prob < 0.5,
                        sim_add, tf.ones_like(sim_add))

    def _new_hidden_state(self, prev_all_hidden_cell_states,
                          new_state, alignments, time):
        """Helper method to look into rnn history"""

        # reshape to (batch, time, memory_time) and
        # do not include current time because
        # we do not want to pay attention to it,
        # but we need to read it instead of
        # adding conditional flow if time == 0
        prev_cell_states = tf.transpose(
                prev_all_hidden_cell_states.gather(
                        tf.range(0, time + 1)), [1, 0, 2]
        )[:, :-1, :]

        return self._apply_alignments_to_history(alignments,
                                                 prev_cell_states,
                                                 new_state)

    def _new_next_cell_state(self, prev_all_hidden_cell_states,
                             next_cell_state, new_cell_output,
                             alignments, time):
        """Helper method to recalculate new next_cell_state"""

        if isinstance(next_cell_state, tf.contrib.rnn.LSTMStateTuple):
            next_cell_state_c = self._new_hidden_state(
                prev_all_hidden_cell_states.c,
                next_cell_state.c,
                alignments,
                time
            )
            next_cell_state_h = self._new_hidden_state(
                prev_all_hidden_cell_states.h,
                new_cell_output,
                alignments,
                time
            )
            return tf.contrib.rnn.LSTMStateTuple(next_cell_state_c,
                                                 next_cell_state_h)
        else:
            return self._new_hidden_state(prev_all_hidden_cell_states,
                                          alignments, new_cell_output, time)

    @staticmethod
    def _all_hidden_cell_states(prev_all_hidden_cell_states,
                                next_cell_state, time):
        """Helper method to calculate all_hidden_cell_states tensor array"""

        if isinstance(next_cell_state, tf.contrib.rnn.LSTMStateTuple):
            return tf.contrib.rnn.LSTMStateTuple(
                prev_all_hidden_cell_states.c.write(time + 1,
                                                    next_cell_state.c),
                prev_all_hidden_cell_states.h.write(time + 1,
                                                    next_cell_state.h)
            )
        else:
            return prev_all_hidden_cell_states.write(time + 1,
                                                     next_cell_state)

    def call(self, inputs, state):
        """Perform a step of attention-wrapped RNN.

        The order is changed:
        - Step 1: Calculate output for attention based on the previous output
          and current input
        - Step 2: Score the output with `attention_mechanism`.
        - Step 3: Calculate the alignments by passing the score through the
          `normalizer` and limit them by time.
        - Step 4: Calculate the context vector as the inner product between the
          alignments and the attention_mechanism's values (memory).
        - Step 5: Calculate the attention output by concatenating the cell
        output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).
        - Step 6: Mix the `inputs` and `attention` output via
          `cell_input_fn`.
        - Step 7: Call the wrapped `cell` with this input and its previous
        state.

        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time
          step.
          state: An instance of `TimeAttentionWrapperState` containing
            tensors from the previous time step.

        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:

          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `TimeAttentionWrapperState`
             containing the state calculated at this time step.

        Raises:
          TypeError: If `state` is not an instance of `TimeAttentionWrapperState`.
        """
        if not isinstance(state, TimeAttentionWrapperState):
            raise TypeError("Expected state to be instance of "
                            "TimeAttentionWrapperState. "
                            "Received type {} instead.".format(type(state)))

        # Step 1: Calculate attention based on
        #          the previous output and current input
        cell_state = state.cell_state

        inputs, prev_out_for_attn = self._attn_input_fn(inputs, cell_state)

        cell_batch_size = (
                prev_out_for_attn.shape[0].value or
                tf.shape(prev_out_for_attn)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the query (decoder output).  "
                "Are you using "
                "the BeamSearchDecoder?  "
                "You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with tf.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            prev_out_for_attn = tf.identity(
                    prev_out_for_attn, name="checked_prev_out_for_attn")

        if self._is_multi:
            previous_attention_state = state.attention_state
            previous_alignment_history = state.alignment_history
        else:
            previous_attention_state = [state.attention_state]
            previous_alignment_history = [state.alignment_history]

        all_alignments = []
        all_attentions = []
        all_attention_states = []
        maybe_all_histories = []

        for i, attention_mechanism in enumerate(self._attention_mechanisms):
            (attention, alignments,
             next_attention_state) = _compute_time_attention(
                    attention_mechanism, prev_out_for_attn,
                    previous_attention_state[i],
                    # time is added to calculate time attention
                    state.time, self._timed_ntms[i],
                    self._ignore_mask[i],
                    self._attention_layers[i]
                    if self._attention_layers else None)

            alignment_history = previous_alignment_history[i].write(
                    state.time, alignments) if self._alignment_history else ()

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        attention = tf.concat(all_attentions, 1)

        # Step 6: Calculate the true inputs to the cell based on the
        #          calculated attention value.
        cell_inputs = self._cell_input_fn(inputs, attention)

        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        prev_all_hidden_cell_states = state.all_hidden_cell_states

        if self._index_of_attn_to_copy is not None:
            # get relevant previous bot actions from history
            attn_emb_prev_act = all_attentions[self._index_of_attn_to_copy]
            # copy them to current output
            cell_output_with_attn = cell_output + attn_emb_prev_act

            memory_probs = self._get_memory_probs(all_alignments, state.time)

            # check that we do not pay attention to `action_listen`
            bin_sim_listen, _ = self._sim_fn(cell_output_with_attn,
                                             self._emb_for_action_listen)
            # recalculate probs
            memory_probs *= 1 - bin_sim_listen

            history_alignments = self._history_alignments(memory_probs)

            # get previous embedding vector from history
            prev_emb_act = self._prev_emb_action(cell_output_with_attn,
                                                 history_alignments,
                                                 state.time)

            # check that similarity between current embedding is close to
            # the one in the history to which we pay attention to
            bin_sim, _ = self._sim_fn(cell_output_with_attn, prev_emb_act)
            # recalculate probs
            memory_probs *= bin_sim

            history_alignments = self._history_alignments(memory_probs)
            current_time_prob = history_alignments[:, -1:]

            # create additional similarities to maximize
            sim_attn_to_max = self._additional_sim(
                    attn_emb_prev_act,
                    prev_emb_act,
                    current_time_prob
            )
            sim_state_to_max = self._additional_sim(
                    cell_output + tf.stop_gradient(attn_emb_prev_act),
                    prev_emb_act,
                    current_time_prob
            )

            # recalculate new next_cell_state based on history_alignments
            next_cell_state = self._new_next_cell_state(
                    prev_all_hidden_cell_states,
                    next_cell_state,
                    cell_output_with_attn,
                    history_alignments,
                    state.time
            )

            all_hidden_cell_states = self._all_hidden_cell_states(
                    prev_all_hidden_cell_states,
                    next_cell_state,
                    state.time
            )

            if self._output_attention:
                # concatenate cell outputs, attention and copy_attn_debug
                output = tf.concat([cell_output_with_attn,
                                    cell_output,
                                    attention,
                                    sim_attn_to_max, sim_state_to_max,
                                    bin_sim_listen, bin_sim,
                                    current_time_prob], 1)
            else:
                output = cell_output_with_attn

        else:
            # do not waste resources on storing history
            all_hidden_cell_states = None

            if self._output_attention:
                output = tf.concat([cell_output, attention], 1)
            else:
                output = cell_output

        next_state = TimeAttentionWrapperState(
                time=state.time + 1,
                cell_state=next_cell_state,
                attention=attention,
                attention_state=self._item_or_tuple(all_attention_states),
                alignments=self._item_or_tuple(all_alignments),
                alignment_history=self._item_or_tuple(maybe_all_histories),
                all_hidden_cell_states=all_hidden_cell_states
        )
        return output, next_state


class ChronoBiasLayerNormBasicLSTMCell(tf.contrib.rnn.LayerNormBasicLSTMCell):
    """Custom LayerNormBasicLSTMCell that allows chrono initialization
        of gate biases.

        See super class for description."""

    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 input_bias=0.0,
                 activation=tf.tanh,
                 layer_norm=True,
                 norm_gain=1.0,
                 norm_shift=0.0,
                 dropout_keep_prob=1.0,
                 dropout_prob_seed=None,
                 out_layer_size=None,
                 reuse=None):
        super(ChronoBiasLayerNormBasicLSTMCell, self).__init__(
                num_units,
                forget_bias=forget_bias,
                activation=activation,
                layer_norm=layer_norm,
                norm_gain=norm_gain,
                norm_shift=norm_shift,
                dropout_keep_prob=dropout_keep_prob,
                dropout_prob_seed=dropout_prob_seed,
                reuse=reuse
        )
        self._input_bias = input_bias
        self._out_layer_size = out_layer_size

    @property
    def output_size(self):
        return self._out_layer_size or self._num_units

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units,
                                             self.output_size)

    @staticmethod
    def _dense_layer(args, layer_size):
        """Optional out projection layer"""
        proj_size = args.get_shape()[-1]
        dtype = args.dtype
        weights = tf.get_variable("kernel",
                                  [proj_size, layer_size],
                                  dtype=dtype)
        bias = tf.get_variable("bias",
                               [layer_size],
                               dtype=dtype)
        out = tf.nn.bias_add(tf.matmul(args, weights), bias)
        return out

    def call(self, inputs, state):
        """LSTM cell with layer normalization and recurrent dropout."""
        c, h = state
        args = tf.concat([inputs, h], 1)
        concat = self._linear(args)
        dtype = args.dtype

        i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=1)
        if self._layer_norm:
            i = self._norm(i, "input", dtype=dtype)
            j = self._norm(j, "transform", dtype=dtype)
            f = self._norm(f, "forget", dtype=dtype)
            o = self._norm(o, "output", dtype=dtype)

        g = self._activation(j)
        if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
            g = tf.nn.dropout(g, self._keep_prob, seed=self._seed)

        new_c = (
                c * tf.sigmoid(f + self._forget_bias) +
                g * tf.sigmoid(i + self._input_bias))  # added input_bias

        # do not do layer normalization on the new c,
        # because there are no trainable weights
        # if self._layer_norm:
        #     new_c = self._norm(new_c, "state", dtype=dtype)

        new_h = self._activation(new_c) * tf.sigmoid(o)

        # added dropout to the hidden state h
        if (not isinstance(self._keep_prob, float)) or self._keep_prob < 1:
            new_h = tf.nn.dropout(new_h, self._keep_prob, seed=self._seed)

        # add postprocessing of the output
        if self._out_layer_size is not None:
            with tf.variable_scope('out_layer'):
                new_h = self._dense_layer(new_h, self._out_layer_size)

        new_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)
        return new_h, new_state
