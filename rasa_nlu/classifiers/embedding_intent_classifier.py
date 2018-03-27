from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import os

import typing
from future.utils import PY3
from typing import List, Text, Any, Optional, Dict

from rasa_nlu.classifiers import INTENT_RANKING_LENGTH
from rasa_nlu.components import Component
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Metadata
from rasa_nlu.training_data import Message
from rasa_nlu.training_data import TrainingData
import numpy as np
import tensorflow as tf

try:
    import cPickle as pickle
except:
    import pickle

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import tensorflow as tf

class EmbeddingIntentClassifier(Component):
    """Intent classifier using supervised embeddings

    Based on the starspace idea from: https://arxiv.org/abs/1709.03856"""

    name = "intent_classifier_tensorflow_embedding"

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    defaults = {
        # nn architecture
        "num_hidden_layers_a": 2,
        "num_hidden_layers_b": 1,
        "hidden_layer_size": 256,
        "batch_size": 32,
        "epochs": 100,

        # embedding parameters
        "embed_dim": 10,
        "mu_pos": 0.8,
        "mu_neg": -0.4,
        "similarity_type": "cosine",
        "num_neg": 10,

        # regularization
        "C2": 0.002,
        "C_emb": 0.8,
        "droprate": 0.2,

        # flag if to tokenize intents
        "intent_tokenization_flag": False,
        "intent_split_symbol": "_"
    }

    def __init__(self, component_config=None,
                 session=None, embedding_placeholder=None, graph=None,
                 intent_placeholder=None, similarity_op=None,
                 intent_dict=None, intent_token_dict=None):

        # type: () -> None
        """Declare instant variables with default values"""

        super(EmbeddingIntentClassifier, self).__init__(component_config)

        # nn architecture
        self.num_hidden_layers_a = self.component_config['num_hidden_layers_a']
        self.num_hidden_layers_b = self.component_config['num_hidden_layers_b']
        self.hidden_layer_size = self.component_config['hidden_layer_size']
        self.batch_size = self.component_config['batch_size']
        self.epochs = self.component_config['epochs']

        # embedding parameters
        self.embed_dim = self.component_config['embed_dim']
        self.mu_pos = self.component_config['mu_pos']
        self.mu_neg = self.component_config['mu_neg']
        self.similarity_type = self.component_config['similarity_type']
        self.num_neg = self.component_config['num_neg']

        # regularization
        self.C2 = self.component_config['C2']  # l2
        self.C_emb = self.component_config['C_emb']
        self.droprate = self.component_config['droprate']  # dropout

        # flag if tokenize intents
        self.intent_tokenization_flag = self.component_config['intent_tokenization_flag']
        self.intent_split_symbol = self.component_config['intent_split_symbol']

        # mu is different for pos and neg intents, will be created in train()
        self.mu = None

        # transform intents to numbers
        self.intent_dict = intent_dict  # encode intents with numbers
        self.intent_token_dict = intent_token_dict  # encode words in intents with numbers

        # tf related instances
        self.session = session
        self.graph = graph if graph is not None else tf.Graph()
        self.embedding_placeholder = embedding_placeholder
        self.intent_placeholder = intent_placeholder
        self.similarity_op = similarity_op

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["tensorflow"]

    def train(self, training_data, cfg, **kwargs):
        # type: (TrainingData, RasaNLUModelConfig, **Any) -> None
        """Train the embedding intent classifier on a data set."""

        # transform intents to numbers
        # if intent_tokenization = False these dicts are identical
        self.intent_dict, self.intent_token_dict = self._create_intent_dicts(training_data)

        # check if number of negatives is less than number of intents
        logger.debug("Check if num_neg {} is smaller than number of intents {}, "
                     "else set num_neg to the number of intents".format(
                      self.num_neg, len(self.intent_dict)))
        self.num_neg = min(self.num_neg, len(self.intent_dict) - 1)

        # create an array for mu
        self.mu = self.mu_neg * np.ones(self.num_neg + 1)
        self.mu[0] = self.mu_pos

        X = np.stack([example.get("text_features")
                      for example in training_data.intent_examples])

        # due to tokenization of the intents, special array is created
        # that stores the number of the whole intent from intent_dict
        intents_for_X = np.zeros(X.shape[0], dtype=int)
        for i, example in enumerate(training_data.intent_examples):
            intents_for_X[i] = self.intent_dict[example.get("intent")]

        # array that holds bag of words for tokenized intents
        # if intent_tokenization = False this is one-hot vector
        Y = np.zeros([X.shape[0], len(self.intent_token_dict)])
        for i, example in enumerate(training_data.intent_examples):
            if self.intent_tokenization_flag and self.intent_split_symbol:
                for t in example.get("intent").split(self.intent_split_symbol):
                    Y[i, self.intent_token_dict[t]] = 1
            else:
                Y[i, self.intent_dict[example.get("intent")]] = 1

        # the matrix that encodes intents as bag of words in rows
        # if intent_tokenization = False this is identity matrix
        encoded_all_intents = self._create_encoded_intents()

        # stack encoded_all_intents on top of each other
        # to create candidates for training examples
        # to calculate training accuracy
        all_Y = np.stack([encoded_all_intents for _ in range(X.shape[0])])

        with self.graph.as_default():
            a_in = tf.placeholder(tf.float32, (None, X.shape[-1]), name='a')
            b_in = tf.placeholder(tf.float32, (None, None, Y.shape[-1]),
                                  name='b')

            self.embedding_placeholder = a_in
            self.intent_placeholder = b_in

            is_training = tf.placeholder_with_default(False, shape=())

            sim, sim_emb = self._tf_sim(a_in, b_in, is_training)
            self.similarity_op = sim
            loss = self._tf_loss(sim, sim_emb)

            train_op = tf.train.AdamOptimizer().minimize(loss)
            init = tf.global_variables_initializer()

            # train tensorflow graph
            sess = tf.Session()
            self.session = sess

            sess.run(init)

            batches_per_epoch = (len(X) // self.batch_size +
                                 int(len(X) % self.batch_size > 0))
            for ep in range(self.epochs):
                indices = np.random.permutation(len(X))
                sess_out = {}
                for i in range(batches_per_epoch):
                    end_idx = (i + 1) * self.batch_size
                    start_idx = i * self.batch_size
                    batch_a = X[indices[start_idx:end_idx]]
                    batch_pos_b = Y[indices[start_idx:end_idx]]
                    intents_for_b = intents_for_X[indices[start_idx:end_idx]]
                    # add negatives
                    batch_b = self._create_batch_b(batch_pos_b, intents_for_b,
                                                   encoded_all_intents)

                    sess_out = sess.run({'loss': loss, 'train_op': train_op},
                                        feed_dict={a_in: batch_a,
                                                   b_in: batch_b,
                                                   is_training: True})

                if (ep+1) % 10 == 0:
                    train_sim = sess.run(sim, feed_dict={a_in: X,
                                                         b_in: all_Y,
                                                         is_training: False})

                    train_acc = np.mean(np.argmax(train_sim, -1) == intents_for_X)
                    logger.debug("epoch {} / {}: loss {}, "
                                 "train accuracy : {:.3f}".format(
                                  (ep+1), self.epochs,
                                  sess_out.get('loss'), train_acc))

    def _create_intent_dicts(self, training_data):
        """Create intent dictionary"""

        intent_dict = {}
        intent_token_dict = {}
        for example in training_data.intent_examples:
            intent = example.get("intent")
            if intent not in intent_dict:
                intent_dict[intent] = len(intent_dict)

            # split intents if the flag is true
            # and if split_symbol is not empty string
            if self.intent_tokenization_flag and self.intent_split_symbol:
                for t in intent.split(self.intent_split_symbol):
                    if t not in intent_token_dict:
                        intent_token_dict[t] = len(intent_token_dict)

        if not (self.intent_tokenization_flag and self.intent_split_symbol):
            intent_token_dict = intent_dict

        return intent_dict, intent_token_dict

    def _create_encoded_intents(self):
        """Create matrix with intents encoded in rows as bag of words,
        if intent_tokenization_flag = False this is identity matrix"""

        if self.intent_tokenization_flag and self.intent_split_symbol:
            encoded_all_intents = np.zeros((len(self.intent_dict),
                                            len(self.intent_token_dict)))
            for key, value in self.intent_dict.items():
                for t in key.split(self.intent_split_symbol):
                    encoded_all_intents[value, self.intent_token_dict[t]] = 1
        else:
            encoded_all_intents = np.eye(len(self.intent_dict))

        return encoded_all_intents

    def _create_batch_b(self, batch_pos_b, intent_ids, encoded_all_intents):
        """Create batch of intents, where the first is correct intent
            and the rest are wrong intents sampled randomly"""
        batch_pos_b = batch_pos_b[:, np.newaxis, :]

        # sample negatives
        batch_neg_b = np.zeros((batch_pos_b.shape[0], self.num_neg,
                                batch_pos_b.shape[-1]))
        for b in range(batch_pos_b.shape[0]):
            # create negative indexes out of possible ones
            # except for correct index of b
            negative_indexes = [i for i in range(encoded_all_intents.shape[0])
                                if i != intent_ids[b]]
            negs = np.random.choice(negative_indexes, size=self.num_neg)

            batch_neg_b[b] = encoded_all_intents[negs]

        return np.concatenate([batch_pos_b, batch_neg_b], 1)

    def _tf_sim(self, a_in, b_in, is_training):
        """Define tensorflow graph to calculate similarity"""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        # embed sentences
        a = a_in
        for _ in range(self.num_hidden_layers_a):
            a = tf.layers.dense(inputs=a,
                                units=self.hidden_layer_size,
                                activation=tf.nn.relu,
                                kernel_regularizer=reg)
            a = tf.layers.dropout(a, rate=self.droprate, training=is_training)

        a = tf.layers.dense(inputs=a,
                            units=self.embed_dim,
                            kernel_regularizer=reg)
        # embed intents
        b = b_in
        for _ in range(self.num_hidden_layers_b):
            b = tf.layers.dense(inputs=b,
                                units=self.hidden_layer_size,
                                activation=tf.nn.relu,
                                kernel_regularizer=reg)
            b = tf.layers.dropout(b, rate=self.droprate, training=is_training)

        b = tf.layers.dense(inputs=b,
                            units=self.embed_dim,
                            kernel_regularizer=reg)

        if self.similarity_type == "cosine":
            a = tf.nn.l2_normalize(a, -1)
            b = tf.nn.l2_normalize(b, -1)

        if self.similarity_type == "inner" or self.similarity_type == "cosine":
            sim = tf.reduce_sum(tf.expand_dims(a, 1) * b, -1)

            # similarity between intent embeddings
            sim_emb = tf.reduce_sum(b[:, 0:1, :] * b[:, 1:, :], -1)

            return sim, sim_emb
        else:
            raise Exception("ERROR: wrong similarity type {}, "
                            "should be 'cosine' or 'inner'"
                            "".format(self.similarity_type))

    def _tf_loss(self, sim, sim_emb):
        """Define loss"""

        factors = tf.concat([-1 * tf.ones([1, 1]),
                             tf.ones([1, tf.shape(sim)[1] - 1])], 1)
        max_margin = tf.maximum(0., self.mu + factors * sim)

        loss = (tf.reduce_mean(tf.reduce_sum(max_margin, 1)) +
                # penalize max similarity between intent embeddings
                tf.reduce_mean(tf.maximum(0., tf.reduce_max(sim_emb, 1))) * self.C_emb +
                # add regularization losses
                tf.losses.get_regularization_loss())
        return loss

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Return the most likely intent and its similarity to the input."""

        # get features (bag of words) for a message
        X = message.get("text_features").reshape(1, -1)

        # the matrix that encodes intents as bag of words in rows
        # if intent_tokenization = False this is identity matrix
        encoded_all_intents = self._create_encoded_intents()

        # stack encoded_all_intents on top of each other
        # in order to make the appropriate size b_in tensor for tf graph
        all_Y = np.stack([encoded_all_intents for _ in range(X.shape[0])])

        inv_intent_dict = {value: key
                           for key, value in self.intent_dict.items()}

        # load tf graph and session
        a_in = self.embedding_placeholder
        b_in = self.intent_placeholder

        sim = self.similarity_op
        sess = self.session

        message_sim = sess.run(sim, feed_dict={a_in: X,
                                               b_in: all_Y})
        message_sim = message_sim.flatten()  # sim is a matrix

        intent_ids = message_sim.argsort()[::-1]
        message_sim[::-1].sort()

        # transform sim to python list for JSON serializing
        message_sim = message_sim.tolist()

        if intent_ids.size > 0:
            intent = {"name": inv_intent_dict[intent_ids[0]],
                      "confidence": message_sim[0]}

            ranking = list(zip(list(intent_ids), message_sim))
            ranking = ranking[:INTENT_RANKING_LENGTH]
            intent_ranking = [{"name": inv_intent_dict[intent_idx],
                               "confidence": score}
                              for intent_idx, score in ranking]
        else:
            intent = {"name": None, "confidence": 0.0}
            intent_ranking = []

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[EmbeddingIntentClassifier]
             **kwargs  # type: **Any
             ):
        # type: (...) -> EmbeddingIntentClassifier

        meta = model_metadata.for_component(cls.name)

        if model_dir and meta.get("classifier_file"):
            file_name = meta.get("classifier_file")
            checkpoint = os.path.join(model_dir, file_name)
            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session()
                saver = tf.train.import_meta_graph(checkpoint + '.meta')

                saver.restore(sess, checkpoint)

                embedding_placeholder = tf.get_collection(
                    'embedding_placeholder')[0]
                intent_placeholder = tf.get_collection(
                    'intent_placeholder')[0]
                similarity_op = tf.get_collection(
                    'similarity_op')[0]

            intent_token_dict = pickle.load(open(os.path.join(
                model_dir, cls.name + "_intent_token_dict.pkl"), 'rb'))
            intent_dict = pickle.load(open(os.path.join(
                model_dir, cls.name + "_intent_dict.pkl"), 'rb'))

            return EmbeddingIntentClassifier(
                        session=sess, embedding_placeholder=embedding_placeholder,
                        graph=graph, intent_placeholder=intent_placeholder,
                        intent_token_dict=intent_token_dict, intent_dict=intent_dict,
                        similarity_op=similarity_op)

        else:
            raise Exception("Failed to load nlu model. Path {} "
                            "doesn't exist".format(os.path.abspath(model_dir)))

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""

        checkpoint = os.path.join(model_dir, self.name + ".ckpt")

        try:
            os.makedirs(os.path.dirname(checkpoint))
        except OSError as e:
            # be happy if someone already created the path
            import errno
            if e.errno != errno.EEXIST:
                raise
        with self.graph.as_default():
            self.graph.clear_collection('embedding_placeholder')
            self.graph.add_to_collection('embedding_placeholder',
                                         self.embedding_placeholder)

            self.graph.clear_collection('intent_placeholder')
            self.graph.add_to_collection('intent_placeholder',
                                         self.intent_placeholder)

            self.graph.clear_collection('similarity_op')
            self.graph.add_to_collection('similarity_op',
                                         self.similarity_op)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

            pickle.dump(self.intent_token_dict, open(os.path.join(
                model_dir, self.name + "_intent_token_dict.pkl"), 'wb'))
            pickle.dump(self.intent_dict,  open(os.path.join(
                model_dir, self.name + "_intent_dict.pkl"), 'wb'))

        return {"classifier_file": self.name + ".ckpt"}
