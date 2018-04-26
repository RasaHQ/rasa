from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import os
import logging
import warnings
import typing
from tqdm import tqdm

from typing import \
    Any, List, Optional, Text

import numpy as np
import copy
from rasa_core.policies import Policy
from rasa_core.featurizers import \
    TrackerFeaturizer, FullDialogueTrackerFeaturizer, \
    LabelTokenizerSingleStateFeaturizer
from rasa_core import utils

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


class EmbeddingPolicy(Policy):
    SUPPORTS_ONLINE_TRAINING = True

    @classmethod
    def _standard_featurizer(cls):
        return FullDialogueTrackerFeaturizer(
                    LabelTokenizerSingleStateFeaturizer())

    defaults = {
        # nn architecture
        "num_hidden_layers_a": 0,
        "hidden_layer_size_a": [],
        "num_hidden_layers_b": 0,
        "hidden_layer_size_b": [],
        "rnn_size": 64,
        "batch_size": 16,
        "epochs": 1,

        # embedding parameters
        "embed_dim": 10,
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        "mu_neg": -0.2,  # should be -1.0 < ... < 1.0 for 'cosine'
        "similarity_type": 'cosine',  # string 'cosine' or 'inner'
        "num_neg": 20,
        "use_max_sim_neg": True,  # flag which loss function to use

        # regularization
        "C2": 0.001,
        "C_emb": 0.8,

        "droprate_a": 0.1,
        "droprate_b": 0.1,
        "droprate_c": 0.3,
        "droprate_rnn": 0.1,
        "droprate_out": 0.1,

        "nuke_slots_ones_in_epochs": 3,  # set to None or 0 to disable

        # attention parameters
        "attn_bias_for_last_input": 0.5,  # set to 1 to turn off attention

        # visualization of accuracy
        "calc_acc_ones_in_epochs": 50,  # small values affect performance
        "calc_acc_on_num_examples": 100  # large values affect performance
    }

    @staticmethod
    def _check_hidden_layer_sizes(num_layers, layer_size, name=''):
        num_layers = int(num_layers)

        if num_layers < 0:
            logger.error("num_hidden_layers_{} = {} < 0."
                         "Set it to 0".format(name, num_layers))
            num_layers = 0

        if isinstance(layer_size, list) and len(layer_size) != num_layers:
            if len(layer_size) == 0:
                raise ValueError("hidden_layer_size_{} = {} "
                                 "is an empty list, "
                                 "while num_hidden_layers_{} = {} > 0"
                                 "".format(name, layer_size,
                                           name, num_layers))

            logger.error("The length of hidden_layer_size_{} = {} "
                         "does not correspond to num_hidden_layers_{} "
                         "= {}. Set hidden_layer_size_{} to "
                         "the first element = {} for all layers"
                         "".format(name, len(layer_size),
                                   name, num_layers,
                                   name, layer_size[0]))

            layer_size = layer_size[0]

        if not isinstance(layer_size, list):
            layer_size = [layer_size for _ in range(num_layers)]

        return num_layers, layer_size

    def _load_nn_architecture_params(self, config):
        self.num_hidden_layers_a = config['num_hidden_layers_a']
        self.hidden_layer_size_a = config['hidden_layer_size_a']
        (self.num_hidden_layers_a,
         self.hidden_layer_size_a) = self._check_hidden_layer_sizes(
                                            self.num_hidden_layers_a,
                                            self.hidden_layer_size_a,
                                            name='a')

        self.num_hidden_layers_b = config['num_hidden_layers_b']
        self.hidden_layer_size_b = config['hidden_layer_size_b']
        (self.num_hidden_layers_b,
         self.hidden_layer_size_b) = self._check_hidden_layer_sizes(
                                            self.num_hidden_layers_b,
                                            self.hidden_layer_size_b,
                                            name='b')
        if self.share_embedding:
            if self.num_hidden_layers_a != self.num_hidden_layers_b or \
                    self.hidden_layer_size_a != self.hidden_layer_size_b:
                logger.debug("Due to sharing vocabulary in featurizer, "
                             "embedding weights are shared as well. "
                             "So num_hidden_layers_b and "
                             "hidden_layer_size_b are set to the ones "
                             "for `a`.")
                self.num_hidden_layers_b = self.num_hidden_layers_a
                self.hidden_layer_size_b = self.hidden_layer_size_a

        self.rnn_size = config['rnn_size']

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

        self.droprate = dict()
        self.droprate['a'] = config['droprate_a']
        self.droprate['b'] = config['droprate_b']
        self.droprate['c'] = config['droprate_c']
        self.droprate['rnn'] = config['droprate_rnn']
        self.droprate['out'] = config['droprate_out']

        self.nuke_slots_ones_in_epochs = config['nuke_slots_ones_in_epochs']

    def _load_attn_params(self, config):
        self.attn_bias_for_last_input = max(
                config['attn_bias_for_last_input'], 0.0)

    def _load_visual_params(self, config):
        self.calc_acc_ones_in_epochs = config['calc_acc_ones_in_epochs']
        if self.calc_acc_ones_in_epochs < 1:
            self.calc_acc_ones_in_epochs = self.epochs
        self.calc_acc_on_num_examples = config['calc_acc_on_num_examples']

    def _load_params(self, **kwargs):
        config = copy.deepcopy(self.defaults)
        config.update(kwargs)
        # nn architecture parameters
        self._load_nn_architecture_params(config)
        # embedding parameters
        self._load_embedding_params(config)
        # regularization
        self._load_regularization_params(config)
        # attention parameters
        self._load_attn_params(config)
        # visualization of accuracy
        self._load_visual_params(config)

    @staticmethod
    def _check_tensorflow():
        if tf is None:
            raise ImportError(
                'Failed to import `tensorflow`. '
                'Please install `tensorflow`. '
                'For example with `pip install tensorflow`.')

    def __init__(
            self,
            featurizer=None,  # type: Optional[FullDialogueTrackerFeaturizer]
            fbias=None,  # type: Optional[np.ndarray]
            encoded_all_actions=None,  # type: Optional[np.ndarray]
            session=None,  # type: Optional[tf.Session]
            graph=None,  # type: Optional[tf.Graph]
            intent_placeholder=None,  # type: Optional[tf.Tensor]
            action_placeholder=None,  # type: Optional[tf.Tensor]
            slots_placeholder=None,  # type: Optional[tf.Tensor]
            similarity_op=None,  # type: Optional[tf.Tensor]
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
        self.mean_time = None
        self.fbias = fbias

        # encode all actions with numbers
        self.encoded_all_actions = encoded_all_actions

        # tf related instances
        self.session = session
        self.graph = graph
        self.a_in = intent_placeholder
        self.b_in = action_placeholder
        self.c_in = slots_placeholder
        self.sim_op = similarity_op
        self.alignment_history = None
        # for continue training
        self.train_op = None
        self.is_training = None

    # data helpers:
    def _create_all_Y_d(self, dialogue_len):
        """Stack encoded_all_intents on top of each other
            to create candidates for training examples
            to calculate training accuracy"""
        all_Y_d = np.stack([self.encoded_all_actions
                            for _ in range(dialogue_len)])
        return all_Y_d

    def _create_X_slots(self, data_X):

        slot_start = \
            self.featurizer.state_featurizer.user_feature_len
        prev_start = slot_start + \
            self.featurizer.state_featurizer.slot_feature_len

        # do not include prev actions
        X = data_X[:, :, :slot_start]
        slots = data_X[:, :, slot_start:prev_start]

        return X, slots

    def _create_Y_actions_for_X(self, data_Y):
        """Prepare data for training"""

        actions_for_X = data_Y.argmax(axis=-1)

        Y = np.stack([np.stack([self.encoded_all_actions[action_idx]
                                for action_idx in action_ids])
                      for action_ids in actions_for_X])

        return Y, actions_for_X

    # tf helpers:
    def _create_tf_nn(self, x_in, num_layers, layer_size, droprate, name):
        """Create embed nn for layer with name"""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        x = tf.nn.relu(x_in)
        for i in range(num_layers):
            x = tf.layers.dense(inputs=x,
                                units=layer_size[i],
                                activation=tf.nn.relu,
                                kernel_regularizer=reg,
                                name='hidden_layer_{}_{}'.format(name, i),
                                reuse=tf.AUTO_REUSE)
            x = tf.layers.dropout(x, rate=droprate,
                                  training=self.is_training)
        return x

    def _create_embed(self, x, name):
        reg = tf.contrib.layers.l2_regularizer(self.C2)
        emb_x = tf.layers.dense(inputs=x,
                                units=self.embed_dim,
                                kernel_regularizer=reg,
                                name='embed_layer_{}'.format(name),
                                reuse=tf.AUTO_REUSE)
        return emb_x

    def _create_rnn_cell(self):
        """Create one rnn cell"""

        # chrono initialization for forget bias
        # assuming that characteristic time is average dialogue length
        fbias = np.log(
                (self.mean_time - 2) *
                np.random.random(self.rnn_size) + 1)

        keep_prob = 1.0 - (self.droprate['rnn'] *
                           tf.cast(self.is_training, tf.float32))
        cell = ChronoLayerNormBasicLSTMCell(
                num_units=self.rnn_size,
                layer_norm=False,
                dropout_keep_prob=keep_prob,
                forget_bias=fbias,
                input_bias=-fbias
        )
        return cell

    @staticmethod
    def _create_attn_cell(cell, memory,
                          real_length, attn_bias_for_last_input):
        """Create wrap cell in attention cell with given memory"""

        num_units = int(memory.shape[-1])
        attn_mech = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, memory=memory,
                memory_sequence_length=real_length,
                normalize=True,
                probability_fn=tf.identity
        )

        def cell_input_fn(inputs, attention):
            res = tf.concat([attention[:, :num_units],
                             inputs[:, num_units:]], -1)
            return res

        attn_cell = TimeAttentionWrapper(
                cell, attn_mech,
                attn_bias_for_last_input=attn_bias_for_last_input,
                cell_input_fn=cell_input_fn,
                output_attention=False,
                alignment_history=False
        )
        return attn_cell

    def _create_rnn(self, emb_utter, emb_slots, real_length):
        """Create rnn"""

        cell_input = tf.concat([emb_utter, emb_slots], -1)

        cell = self._create_rnn_cell()

        if self.attn_bias_for_last_input < 1.0:
            cell = self._create_attn_cell(cell, emb_utter,
                                          real_length,
                                          self.attn_bias_for_last_input)

        cell_output, _ = tf.nn.dynamic_rnn(
                cell, cell_input,
                dtype=tf.float32,
                sequence_length=real_length,
                scope='rnn_decoder_{}'.format(0)
        )
        return cell_output

    def _tf_sim(self, emb_dial, emb_act, mask):
        """Define similarity"""

        if self.similarity_type == 'cosine':
            emb_act = tf.nn.l2_normalize(emb_act, -1)
            emb_dial = tf.nn.l2_normalize(emb_dial, -1)

        if self.similarity_type in {'cosine', 'inner'}:
            sim = tf.reduce_sum(tf.expand_dims(emb_dial, -2) * emb_act, -1)
            sim *= tf.expand_dims(mask, 2)

            sim_act = tf.reduce_sum(emb_act[:, :, 0:1, :] *
                                    emb_act[:, :, 1:, :], -1)
            sim_act *= tf.expand_dims(mask, 2)

            return sim, sim_act
        else:
            raise ValueError("Wrong similarity type {}, "
                             "should be 'cosine' or 'inner'"
                             "".format(self.similarity_type))

    def _tf_loss(self, sim, sim_act, mask):
        """Define loss"""

        if self.use_max_sim_neg:
            max_sim_neg = tf.reduce_max(sim[:, :, 1:], -1)
            loss = (tf.maximum(0., self.mu_pos - sim[:, :, 0]) +
                    tf.maximum(0., self.mu_neg + max_sim_neg)) * mask

        else:
            # create an array for mu
            mu = self.mu_neg * np.ones(self.num_neg + 1)
            mu[0] = self.mu_pos
            mu = mu[np.newaxis, np.newaxis, :]

            factors = tf.concat([-1 * tf.ones([1, 1, 1]),
                                 tf.ones([1, 1, tf.shape(sim)[-1] - 1])],
                                axis=-1)

            max_margin = tf.maximum(0., mu + factors * sim)

            loss = tf.reduce_sum(max_margin, -1) * mask

        # penalize max similarity between intent embeddings
        loss_act = tf.maximum(0., tf.reduce_max(sim_act, -1))
        loss += loss_act * self.C_emb

        loss = tf.reduce_sum(loss, -1) / tf.reduce_sum(mask, 1)
        # add regularization losses
        loss = tf.reduce_mean(loss) + tf.losses.get_regularization_loss()
        return loss

    def _create_tf_graph(self, a_in, b_in, c_in):
        """Create tf graph for training"""

        if self.share_embedding:
            name_a = 'a_and_b'
            name_b = 'a_and_b'
        else:
            name_a = 'a'
            name_b = 'b'

        a = self._create_tf_nn(a_in,
                               self.num_hidden_layers_a,
                               self.hidden_layer_size_a,
                               self.droprate['a'],
                               name=name_a)
        a = tf.layers.dropout(a, rate=self.droprate['a'],
                              training=self.is_training)
        emb_utter = self._create_embed(a, name=name_a)

        b = self._create_tf_nn(b_in,
                               self.num_hidden_layers_b,
                               self.hidden_layer_size_b,
                               self.droprate['b'],
                               name=name_b)
        shape_b = tf.shape(b)
        b = tf.layers.dropout(b, rate=self.droprate['b'],
                              training=self.is_training,
                              noise_shape=[shape_b[0],
                                           shape_b[1],
                                           1, shape_b[-1]])
        emb_act = self._create_embed(b, name=name_b)

        c = c_in  # no hidden layers for slots
        c = tf.layers.dropout(c, rate=self.droprate['c'],
                              training=self.is_training)
        emb_slots = self._create_embed(c, name='c')

        # if there is at least one `-1` it should be masked
        mask = tf.sign(tf.reduce_max(a_in, -1) + 1)
        real_length = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

        # create rnn
        cell_output = self._create_rnn(emb_utter, emb_slots, real_length)

        cell_output = tf.layers.dropout(cell_output,
                                        rate=self.droprate['out'],
                                        training=self.is_training)
        emb_dial = self._create_embed(cell_output, name='out')

        sim, sim_act = self._tf_sim(emb_dial, emb_act, mask)
        loss = self._tf_loss(sim, sim_act, mask)

        return sim, loss, mask

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
                negative_indexes = [i for i in range(
                                        self.encoded_all_actions.shape[0])
                                    if i != intent_ids[b, h]]

                negs = np.random.choice(negative_indexes, size=self.num_neg)

                batch_neg_b[b, h] = self.encoded_all_actions[negs]

        return np.concatenate([batch_pos_b, batch_neg_b], -2)

    def _train_tf(self, X, Y, slots, actions_for_X, all_Y_d, loss, mask):
        """Train tf graph"""
        self.session.run(tf.global_variables_initializer())

        batches_per_epoch = (len(X) // self.batch_size +
                             int(len(X) % self.batch_size > 0))

        if self.calc_acc_on_num_examples:
            logger.info("Accuracy is updated every {} epochs"
                        "".format(self.calc_acc_ones_in_epochs))
        pbar = tqdm(range(self.epochs), desc="Epochs", initial=1)
        train_acc = 0
        last_loss = 0
        for ep in pbar:
            ids = np.random.permutation(X.shape[0])

            ep_loss = 0
            for i in range(batches_per_epoch):
                start_idx = i * self.batch_size
                end_idx = (i + 1) * self.batch_size

                batch_a = X[ids[start_idx:end_idx]]

                batch_pos_b = Y[ids[start_idx:end_idx]]
                actions_for_b = actions_for_X[ids[start_idx:end_idx]]
                # add negatives
                batch_b = self._create_batch_b(batch_pos_b, actions_for_b)

                batch_c = slots[ids[start_idx:end_idx]]

                if self.nuke_slots_ones_in_epochs:
                    if (ep + 1) % self.nuke_slots_ones_in_epochs == 0:
                        batch_c[:] = 0

                sess_out = self.session.run(
                        {'loss': loss, 'train_op': self.train_op},
                        feed_dict={self.a_in: batch_a,
                                   self.b_in: batch_b,
                                   self.c_in: batch_c,
                                   self.is_training: True}
                )
                ep_loss += sess_out.get('loss') / batches_per_epoch

            if self.calc_acc_on_num_examples:
                if (ep + 1) == 1 or \
                        (ep + 1) % self.calc_acc_ones_in_epochs == 0 or \
                        (ep + 1) == self.epochs:
                    train_acc = self._calc_train_acc(X, slots, actions_for_X,
                                                     all_Y_d, mask)
                    last_loss = ep_loss

                pbar.set_postfix({
                    "loss": "{:.3f}".format(ep_loss),
                    "acc": "{:.3f}".format(train_acc)
                })
            else:
                pbar.set_postfix({
                    "loss": "{:.3f}".format(ep_loss)
                })

        if self.calc_acc_on_num_examples:
            logger.info("loss={:.3f}, accuracy={:.3f}"
                        "".format(last_loss, train_acc))

    def _calc_train_acc(self, X, slots,
                        actions_for_X, all_Y_d, mask):
        """Calculate training accuracy"""
        # choose n examples to calculate train accuracy
        n = self.calc_acc_on_num_examples
        ids = np.random.permutation(len(X))[:n]
        all_Y_d_x = np.stack([all_Y_d for _ in range(X[ids].shape[0])])

        _sim, _mask = self.session.run(
                [self.sim_op, mask],
                feed_dict={self.a_in: X[ids],
                           self.b_in: all_Y_d_x,
                           self.c_in: slots[ids],
                           self.is_training: False}
        )

        train_acc = np.sum((np.argmax(_sim, -1) ==
                            actions_for_X[ids]) * _mask)
        train_acc /= np.sum(_mask)
        return train_acc

    def train(self,
              training_trackers,  # type: List[DialogueStateTracker]
              domain,  # type: Domain
              **kwargs  # type: **Any
              ):
        # type: (...) -> None
        """Trains the policy on given training trackers."""

        logger.debug('Started to train embedding policy.')

        if kwargs:
            logger.debug("Config is updated with {}".format(kwargs))
            self._load_params(**kwargs)

        # dealing with training data
        training_data = self.featurize_for_training(training_trackers,
                                                    domain,
                                                    **kwargs)
        self.mean_time = np.mean(training_data.true_length)

        self.encoded_all_actions = \
            self.featurizer.state_featurizer.create_encoded_all_actions(
                domain)

        # check if number of negatives is less than number of actions
        logger.debug("Check if num_neg {} is smaller "
                     "than number of actions {}, "
                     "else set num_neg to the number of actions - 1"
                     "".format(self.num_neg, domain.num_actions))
        self.num_neg = min(self.num_neg, domain.num_actions - 1)

        X, slots = self._create_X_slots(training_data.X)
        Y, actions_for_X = self._create_Y_actions_for_X(training_data.y)

        all_Y_d = self._create_all_Y_d(X.shape[1])

        self.graph = tf.Graph()
        with self.graph.as_default():
            dialogue_len = None  # use dynamic time for rnn
            self.a_in = tf.placeholder(tf.float32,
                                       (None, dialogue_len,
                                        X.shape[-1]),
                                       name='a')
            self.b_in = tf.placeholder(tf.float32,
                                       (None, dialogue_len,
                                        None, Y.shape[-1]),
                                       name='b')
            self.c_in = tf.placeholder(tf.float32,
                                       (None, dialogue_len,
                                        slots.shape[-1]),
                                       name='c')
            self.is_training = tf.placeholder_with_default(False, shape=())

            self.sim_op, loss, mask = self._create_tf_graph(self.a_in,
                                                            self.b_in,
                                                            self.c_in)

            self.train_op = tf.train.AdamOptimizer().minimize(loss)

            # train tensorflow graph
            self.session = tf.Session()

            self._train_tf(X, Y, slots, actions_for_X, all_Y_d,
                           loss, mask)

    def continue_training(self, training_trackers, domain, **kwargs):
        # type: (List[DialogueStateTracker], Domain, **Any) -> None
        """Continues training an already trained policy."""

        batch_size = kwargs.get('batch_size', 5)
        epochs = kwargs.get('epochs', 50)

        num_samples = batch_size - 1
        num_prev_examples = len(training_trackers) - 1
        for _ in range(epochs):
            sampled_idx = np.random.choice(range(num_prev_examples),
                                           replace=False,
                                           size=min(num_samples,
                                                    num_prev_examples))
            trackers = [training_trackers[i]
                        for i in sampled_idx] + training_trackers[-1:]
            training_data = self.featurize_for_training(trackers,
                                                        domain)
            batch_a, batch_c = self._create_X_slots(training_data.X)
            batch_pos_b, actions_for_b = self._create_Y_actions_for_X(
                                            training_data.y)

            batch_b = self._create_batch_b(batch_pos_b, actions_for_b)

            # fit to one extra example using updated trackers
            self.session.run(self.train_op,
                             feed_dict={self.a_in: batch_a,
                                        self.b_in: batch_b,
                                        self.c_in: batch_c,
                                        self.is_training: True})

    def predict_action_probabilities(self, tracker, domain):
        # type: (DialogueStateTracker, Domain) -> List[float]
        """Predicts the next action the bot should take
        after seeing the tracker.

        Returns the list of probabilities for the next actions"""
        if self.session is None:
            logger.error("There is no trained tf.session: "
                         "component is either not trained or "
                         "didn't receive enough training data")
            return [0.0] * domain.num_actions

        data_X = self.featurizer.create_X([tracker], domain)

        X, slots = self._create_X_slots(data_X)
        all_Y_d = self._create_all_Y_d(X.shape[1])
        all_Y_d_x = np.stack([all_Y_d for _ in range(X.shape[0])])

        _sim = self.session.run(
            self.sim_op, feed_dict={self.a_in: X,
                                    self.b_in: all_Y_d_x,
                                    self.c_in: slots}
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

    def persist(self, path):
        # type: (Text) -> None
        """Persists the policy to a storage."""

        if self.session is None:
            warnings.warn("Persist called without a trained model present. "
                          "Nothing to persist then!")
            return

        self.featurizer.persist(path)

        file_name = 'tensorflow_embedding.ckpt'
        checkpoint = os.path.join(path, file_name)
        utils.create_dir_for_file(checkpoint)

        with self.graph.as_default():
            self.graph.clear_collection('intent_placeholder')
            self.graph.add_to_collection('intent_placeholder', self.a_in)

            self.graph.clear_collection('action_placeholder')
            self.graph.add_to_collection('action_placeholder', self.b_in)

            self.graph.clear_collection('slots_placeholder')
            self.graph.add_to_collection('slots_placeholder', self.c_in)

            self.graph.clear_collection('similarity_op')
            self.graph.add_to_collection('similarity_op', self.sim_op)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with io.open(os.path.join(
                path,
                file_name + ".encoded_all_actions.pkl"), 'wb') as f:
            pickle.dump(self.encoded_all_actions, f)

    @classmethod
    def load(cls, path):
        # type: (Text) -> EmbeddingPolicy
        """Loads a policy from the storage.

        Needs to load its featurizer"""

        if os.path.exists(path):
            featurizer = TrackerFeaturizer.load(path)

            file_name = 'tensorflow_embedding.ckpt'
            checkpoint = os.path.join(path, file_name)

            if os.path.exists(checkpoint + '.meta'):
                graph = tf.Graph()
                with graph.as_default():
                    sess = tf.Session()
                    saver = tf.train.import_meta_graph(checkpoint + '.meta')

                    saver.restore(sess, checkpoint)

                    a_in = tf.get_collection('intent_placeholder')[0]
                    b_in = tf.get_collection('action_placeholder')[0]
                    c_in = tf.get_collection('slots_placeholder')[0]
                    sim_op = tf.get_collection('similarity_op')[0]

                with io.open(os.path.join(
                        path,
                        file_name + ".encoded_all_actions.pkl"), 'rb') as f:
                    encoded_all_actions = pickle.load(f)

                return cls(featurizer,
                           encoded_all_actions=encoded_all_actions,
                           session=sess,
                           graph=graph,
                           intent_placeholder=a_in,
                           action_placeholder=b_in,
                           slots_placeholder=c_in,
                           similarity_op=sim_op)
            else:
                return cls(featurizer=featurizer)

        else:
            raise Exception("Failed to load dialogue model. Path {} "
                            "doesn't exist".format(os.path.abspath(path)))


# modified tensorflow attention wrapper
def _compute_time_attention(attention_mechanism, cell_output, attention_state,
                            # time is added to calculate time attention
                            time, attn_bias_for_last_input,
                            attention_layer):
    """Computes the attention and alignments limited by time
    for a given attention_mechanism."""

    alignments, next_attention_state = attention_mechanism(
      cell_output, state=attention_state)

    alpha = attn_bias_for_last_input/(1.0 - attn_bias_for_last_input)
    time_weight = alpha * tf.cast(time, tf.float32) + 1.0
    time_bias = tf.log(time_weight)
    probs = tf.nn.softmax(tf.concat([alignments[:, :time],
                                     alignments[:, time:time+1] +
                                     time_bias], 1))
    # `time_weight = a * time + 1` was not chosen arbitrary
    #
    # if we assume that we have simple stories where each action
    # is followed by `action_listen` then:
    #
    # if `time_weight = 1`, the initial `probs[:, time]`
    # will be `1/(time+1)`
    # and the initial `sum of probs` for 0-intent (before `action_listen`)
    # will be `1/2 for any time`
    #
    # if `time_weight = time`, the initial `probs[:, time]`
    # will be `1/2 for any time`
    # and the initial `sum of probs` for 0-intent (before `action_listen`)
    # will be `(1/4)(time+1)/time`
    #
    # if `time_weight = time + 2`, the initial `probs[:, time]`
    # will be `(1/2)(time+2)/(time+1)`
    # and the initial `sum of probs` for 0-intent (before `action_listen`)
    # will be `1/4 for any time`

    # we do not want initial `probs` to be independent of time
    # to allow different times to be descriptive, but
    # we want `lim(probs) = const` when `time -> inf`
    # to avoid vanishing alignments for large time

    # if `time_weight = a * time + 1`, the initial `probs[:, time]`
    # will be `(a/(a+1))(time+1/a)/(time+1/(a+1))`
    # and the initial `sum of probs` for 0-intent (before `action_listen`)
    # will be `(1/(2a+2))(time+1/a)/(time+1/(a+1))`

    zeros = tf.zeros_like(alignments)
    alignments = tf.concat([probs, zeros[:, time+1:]], 1)

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

    return attention, alignments, next_attention_state


class TimeAttentionWrapper(tf.contrib.seq2seq.AttentionWrapper):
    """Custom AttentionWrapper that takes into account time
        when calculating attention.
        Attention is calculated before calling rnn cell."""

    def __init__(self,
                 cell,
                 attention_mechanism,
                 attention_layer_size=None,
                 attn_bias_for_last_input=0.0,
                 alignment_history=False,
                 cell_input_fn=None,
                 output_attention=True,
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
        self._attn_bias_for_last_input = attn_bias_for_last_input

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
        - Step 5: Calculate the attention output by concatenating the cell output
          and context through the attention layer (a linear layer with
          `attention_layer_size` outputs).
        - Step 6: Mix the `inputs` and `attention` output via
          `cell_input_fn`.
        - Step 7: Call the wrapped `cell` with this input and its previous state.

        Args:
          inputs: (Possibly nested tuple of) Tensor, the input at this time step.
          state: An instance of `AttentionWrapperState` containing
            tensors from the previous time step.

        Returns:
          A tuple `(attention_or_cell_output, next_state)`, where:

          - `attention_or_cell_output` depending on `output_attention`.
          - `next_state` is an instance of `AttentionWrapperState`
             containing the state calculated at this time step.

        Raises:
          TypeError: If `state` is not an instance of `AttentionWrapperState`.
        """
        if not isinstance(state, tf.contrib.seq2seq.AttentionWrapperState):
            raise TypeError("Expected state to be instance of AttentionWrapperState. "
                            "Received type %s instead." % type(state))

        # Step 1: Calculate attention based on
        #          the previous output and current input
        if isinstance(state.cell_state, tf.contrib.rnn.LSTMStateTuple):
            (_, h) = state.cell_state
            # the hidden state is not included, in hope that algorithm
            # would learn correct attention at least to some tokens
            # regardless of the hidden state of lstm memory
            out_for_attn = tf.concat([h, inputs], 1)
        else:
            out_for_attn = tf.concat([state.cell_state, inputs], 1)

        cell_batch_size = (
                out_for_attn.shape[0].value or tf.shape(out_for_attn)[0])
        error_message = (
                "When applying AttentionWrapper %s: " % self.name +
                "Non-matching batch sizes between the memory "
                "(encoder output) and the query (decoder output).  Are you using "
                "the BeamSearchDecoder?  You may need to tile your memory input via "
                "the tf.contrib.seq2seq.tile_batch function with argument "
                "multiple=beam_width.")
        with tf.control_dependencies(
                self._batch_size_checks(cell_batch_size, error_message)):
            out_for_attn = tf.identity(
                out_for_attn, name="checked_out_for_attn")

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
            attention, alignments, next_attention_state = _compute_time_attention(
                attention_mechanism, out_for_attn, previous_attention_state[i],
                # time is added to calculate time attention
                state.time, self._attn_bias_for_last_input,
                self._attention_layers[i] if self._attention_layers else None)

            alignment_history = previous_alignment_history[i].write(
                state.time, alignments) if self._alignment_history else ()

            all_attention_states.append(next_attention_state)
            all_alignments.append(alignments)
            all_attentions.append(attention)
            maybe_all_histories.append(alignment_history)

        attention = tf.concat(all_attentions, 1)

        # Step 6: Calculate the true inputs to the cell based on the
        #          previous attention value.
        cell_inputs = self._cell_input_fn(inputs, attention)
        cell_state = state.cell_state
        cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

        next_state = tf.contrib.seq2seq.AttentionWrapperState(
            time=state.time + 1,
            cell_state=next_cell_state,
            attention=attention,
            attention_state=self._item_or_tuple(all_attention_states),
            alignments=self._item_or_tuple(all_alignments),
            alignment_history=self._item_or_tuple(maybe_all_histories))

        if self._output_attention:
            # does not make too much sense now
            # because rnn cell is called after attention
            return attention, next_state
        else:
            return cell_output, next_state


class ChronoLayerNormBasicLSTMCell(tf.contrib.rnn.LayerNormBasicLSTMCell):
    """Custom LayerNormBasicLSTMCell that allows chrono initialization
        of gate biases."""

    def __init__(self,
                 num_units,
                 forget_bias=1.0,
                 input_bias=0.0,
                 input_size=None,
                 activation=tf.tanh,
                 layer_norm=True,
                 norm_gain=1.0,
                 norm_shift=0.0,
                 dropout_keep_prob=1.0,
                 dropout_prob_seed=None,
                 reuse=None):
        super(ChronoLayerNormBasicLSTMCell, self).__init__(
                num_units,
                forget_bias=forget_bias,
                input_size=input_size,
                activation=activation,
                layer_norm=layer_norm,
                norm_gain=norm_gain,
                norm_shift=norm_shift,
                dropout_keep_prob=dropout_keep_prob,
                dropout_prob_seed=dropout_prob_seed,
                reuse=reuse
        )
        self._input_bias = input_bias

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
        if self._layer_norm:
            new_c = self._norm(new_c, "state", dtype=dtype)
        new_h = self._activation(new_c) * tf.sigmoid(o)

        new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)
        return new_h, new_state
