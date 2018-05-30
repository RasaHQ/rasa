from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import io
import logging
import os
from tqdm import tqdm

import typing
from typing import List, Text, Any, Optional, Dict

from rasa_nlu.classifiers import INTENT_RANKING_LENGTH
from rasa_nlu.components import Component
import numpy as np

try:
    import cPickle as pickle
except ImportError:
    import pickle

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import tensorflow as tf
    from rasa_nlu.config import RasaNLUModelConfig
    from rasa_nlu.training_data import TrainingData
    from rasa_nlu.model import Metadata
    from rasa_nlu.training_data import Message

try:
    import tensorflow as tf
except ImportError:
    tf = None


class EmbeddingIntentClassifier(Component):
    """Intent classifier using supervised embeddings.

    The embedding intent classifier embeds user inputs
    and intent labels into the same space.
    Supervised embeddings are trained by maximizing similarity between them.
    It also provides rankings of the labels that did not "win".

    The embedding intent classifier needs to be preceded by
    a featurizer in the pipeline.
    This featurizer creates the features used for the embeddings.
    It is recommended to use ``intent_featurizer_count_vectors`` that
    can be optionally preceded by ``nlp_spacy`` and ``tokenizer_spacy``.

    Based on the starspace idea from: https://arxiv.org/abs/1709.03856.
    However, in this implementation the `mu` parameter is treated differently
    and additional hidden layers are added together with dropout."""

    name = "intent_classifier_tensorflow_embedding"

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    defaults = {
        # nn architecture
        "num_hidden_layers_a": 2,
        "hidden_layer_size_a": [256, 128],
        "num_hidden_layers_b": 0,
        "hidden_layer_size_b": [],
        "batch_size": [64, 256],
        "epochs": 300,

        # embedding parameters
        "embed_dim": 20,
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        "mu_neg": -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
        "similarity_type": 'cosine',  # string 'cosine' or 'inner'
        "num_neg": 20,
        "use_max_sim_neg": True,  # flag which loss function to use

        # regularization
        "C2": 0.002,
        "C_emb": 0.8,
        "droprate": 0.2,

        # flag if tokenize intents
        "intent_tokenization_flag": False,
        "intent_split_symbol": '_',

        # visualization of accuracy
        "evaluate_every_num_epochs": 10,  # small values may hurt performance
        "evaluate_on_num_examples": 1000  # large values may hurt performance
    }

    @classmethod
    def required_packages(cls):
        # type: () -> List[Text]
        return ["tensorflow"]

    def _load_nn_architecture_params(self):
        self.num_hidden_layers_a = self.component_config['num_hidden_layers_a']
        self.hidden_layer_size_a = self.component_config['hidden_layer_size_a']
        self.num_hidden_layers_b = self.component_config['num_hidden_layers_b']
        self.hidden_layer_size_b = self.component_config['hidden_layer_size_b']
        self.batch_size = self.component_config['batch_size']
        if not isinstance(self.batch_size, list):
            self.batch_size = [self.batch_size, self.batch_size]
        self.epochs = self.component_config['epochs']

    def _load_embedding_params(self):
        self.embed_dim = self.component_config['embed_dim']
        self.mu_pos = self.component_config['mu_pos']
        self.mu_neg = self.component_config['mu_neg']
        self.similarity_type = self.component_config['similarity_type']
        self.num_neg = self.component_config['num_neg']
        self.use_max_sim_neg = self.component_config['use_max_sim_neg']

    def _load_regularization_params(self):
        self.C2 = self.component_config['C2']
        self.C_emb = self.component_config['C_emb']
        self.droprate = self.component_config['droprate']

    def _load_flag_if_tokenize_intents(self):
        self.intent_tokenization_flag = self.component_config[
                                            'intent_tokenization_flag']
        self.intent_split_symbol = self.component_config[
                                        'intent_split_symbol']
        if self.intent_tokenization_flag and not self.intent_split_symbol:
            logger.warning("intent_split_symbol was not specified, "
                           "so intent tokenization will be ignored")
            self.intent_tokenization_flag = False

    def _load_visual_params(self):
        self.evaluate_every_num_epochs = self.component_config[
                                            'evaluate_every_num_epochs']
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs
        self.evaluate_on_num_examples = self.component_config[
                                            'evaluate_on_num_examples']

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

    @staticmethod
    def _check_tensorflow():
        if tf is None:
            raise ImportError(
                'Failed to import `tensorflow`. '
                'Please install `tensorflow`. '
                'For example with `pip install tensorflow`.')

    def __init__(self,
                 component_config=None,  # type: Optional[Dict[Text, Any]]
                 inv_intent_dict=None,  # type: Optional[Dict[int, Text]]
                 encoded_all_intents=None,  # type: Optional[np.ndarray]
                 session=None,  # type: Optional[tf.Session]
                 graph=None,  # type: Optional[tf.Graph]
                 message_placeholder=None,  # type: Optional[tf.Tensor]
                 intent_placeholder=None,  # type: Optional[tf.Tensor]
                 similarity_op=None,   # type: Optional[tf.Tensor]
                 word_embed=None,  # type: Optional[tf.Tensor]
                 intent_embed=None  # type: Optional[tf.Tensor]
                 ):
        # type: (...) -> None
        """Declare instant variables with default values"""
        self._check_tensorflow()
        super(EmbeddingIntentClassifier, self).__init__(component_config)

        # nn architecture parameters
        self._load_nn_architecture_params()
        # embedding parameters
        self._load_embedding_params()
        # regularization
        self._load_regularization_params()
        # flag if tokenize intents
        self._load_flag_if_tokenize_intents()
        # visualization of accuracy
        self._load_visual_params()

        # check if hidden_layer_sizes are valid
        (self.num_hidden_layers_a,
         self.hidden_layer_size_a) = self._check_hidden_layer_sizes(
                                        self.num_hidden_layers_a,
                                        self.hidden_layer_size_a,
                                        name='a')
        (self.num_hidden_layers_b,
         self.hidden_layer_size_b) = self._check_hidden_layer_sizes(
                                        self.num_hidden_layers_b,
                                        self.hidden_layer_size_b,
                                        name='b')

        # transform numbers to intents
        self.inv_intent_dict = inv_intent_dict
        # encode all intents with numbers
        self.encoded_all_intents = encoded_all_intents

        # tf related instances
        self.session = session
        self.graph = graph
        self.a_in = message_placeholder
        self.b_in = intent_placeholder
        self.sim_op = similarity_op

        # persisted embeddings
        self.word_embed = word_embed
        self.intent_embed = intent_embed

    # training data helpers:
    @staticmethod
    def _create_intent_dict(training_data):
        """Create intent dictionary"""

        distinct_intents = set([example.get("intent")
                               for example in training_data.intent_examples])
        return {intent: idx
                for idx, intent in enumerate(sorted(distinct_intents))}

    @staticmethod
    def _create_intent_token_dict(intents, intent_split_symbol):
        """Create intent token dictionary"""

        distinct_tokens = set([token
                               for intent in intents
                               for token in intent.split(
                                        intent_split_symbol)])
        return {token: idx
                for idx, token in enumerate(sorted(distinct_tokens))}

    def _create_encoded_intents(self, intent_dict):
        """Create matrix with intents encoded in rows as bag of words,
        if intent_tokenization_flag = False this is identity matrix"""

        if self.intent_tokenization_flag:
            intent_token_dict = self._create_intent_token_dict(
                list(intent_dict.keys()), self.intent_split_symbol)

            encoded_all_intents = np.zeros((len(intent_dict),
                                            len(intent_token_dict)))
            for key, idx in intent_dict.items():
                for t in key.split(self.intent_split_symbol):
                    encoded_all_intents[idx, intent_token_dict[t]] = 1

            return encoded_all_intents
        else:
            return np.eye(len(intent_dict))

    # data helpers:
    def _create_all_Y(self, size):
        # stack encoded_all_intents on top of each other
        # to create candidates for training examples
        # to calculate training accuracy
        all_Y = np.stack([self.encoded_all_intents for _ in range(size)])

        return all_Y

    def _prepare_data_for_training(self, training_data, intent_dict):
        """Prepare data for training"""

        X = np.stack([e.get("text_features")
                      for e in training_data.intent_examples])

        intents_for_X = np.array([intent_dict[e.get("intent")]
                                  for e in training_data.intent_examples])

        Y = np.stack([self.encoded_all_intents[intent_idx]
                      for intent_idx in intents_for_X])

        return X, Y, intents_for_X

    # tf helpers:
    def _create_tf_embed_nn(self, x_in, is_training,
                            num_layers, layer_size, name):
        """Create embed nn for layer with name"""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        x = x_in
        for i in range(num_layers):
            x = tf.layers.dense(inputs=x,
                                units=layer_size[i],
                                activation=tf.nn.relu,
                                kernel_regularizer=reg,
                                name='hidden_layer_{}_{}'.format(name, i))
            x = tf.layers.dropout(x, rate=self.droprate, training=is_training)

        x = tf.layers.dense(inputs=x,
                            units=self.embed_dim,
                            kernel_regularizer=reg,
                            name='embed_layer_{}'.format(name))
        return x

    def _create_tf_embed(self, a_in, b_in, is_training):
        """Create tf graph for training"""

        emb_a = self._create_tf_embed_nn(a_in, is_training,
                                         self.num_hidden_layers_a,
                                         self.hidden_layer_size_a,
                                         name='a')
        emb_b = self._create_tf_embed_nn(b_in, is_training,
                                         self.num_hidden_layers_b,
                                         self.hidden_layer_size_b,
                                         name='b')
        return emb_a, emb_b

    def _tf_sim(self, a, b):
        """Define similarity"""

        if self.similarity_type == 'cosine':
            a = tf.nn.l2_normalize(a, -1)
            b = tf.nn.l2_normalize(b, -1)

        if self.similarity_type == 'cosine' or self.similarity_type == 'inner':
            sim = tf.reduce_sum(tf.expand_dims(a, 1) * b, -1)

            # similarity between intent embeddings
            sim_emb = tf.reduce_sum(b[:, 0:1, :] * b[:, 1:, :], -1)

            return sim, sim_emb
        else:
            raise ValueError("Wrong similarity type {}, "
                             "should be 'cosine' or 'inner'"
                             "".format(self.similarity_type))

    def _tf_loss(self, sim, sim_emb):
        """Define loss"""

        if self.use_max_sim_neg:
            max_sim_neg = tf.reduce_max(sim[:, 1:], -1)
            loss = tf.reduce_mean(tf.maximum(0., self.mu_pos - sim[:, 0]) +
                                  tf.maximum(0., self.mu_neg + max_sim_neg))
        else:
            # create an array for mu
            mu = self.mu_neg * np.ones(self.num_neg + 1)
            mu[0] = self.mu_pos

            factors = tf.concat([-1 * tf.ones([1, 1]),
                                 tf.ones([1, tf.shape(sim)[1] - 1])], 1)
            max_margin = tf.maximum(0., mu + factors * sim)
            loss = tf.reduce_mean(tf.reduce_sum(max_margin, -1))

        max_sim_emb = tf.maximum(0., tf.reduce_max(sim_emb, -1))

        loss = (loss +
                # penalize max similarity between intent embeddings
                tf.reduce_mean(max_sim_emb) * self.C_emb +
                # add regularization losses
                tf.losses.get_regularization_loss())
        return loss

    # training helpers:
    def _create_batch_b(self, batch_pos_b, intent_ids):
        """Create batch of intents, where the first is correct intent
            and the rest are wrong intents sampled randomly"""

        batch_pos_b = batch_pos_b[:, np.newaxis, :]

        # sample negatives
        batch_neg_b = np.zeros((batch_pos_b.shape[0], self.num_neg,
                                batch_pos_b.shape[-1]))
        for b in range(batch_pos_b.shape[0]):
            # create negative indexes out of possible ones
            # except for correct index of b
            negative_indexes = [i for i in range(
                                    self.encoded_all_intents.shape[0])
                                if i != intent_ids[b]]
            negs = np.random.choice(negative_indexes, size=self.num_neg)

            batch_neg_b[b] = self.encoded_all_intents[negs]

        return np.concatenate([batch_pos_b, batch_neg_b], 1)

    def _linearly_increasing_batch_size(self, ep):
        if self.epochs > 1:
            return int(self.batch_size[0] +
                       ep * (self.batch_size[1] - self.batch_size[0]) /
                       (self.epochs - 1))
        else:
            return int(self.batch_size[0])

    def _train_tf(self, X, Y, intents_for_X,
                  loss, is_training, train_op):
        """Train tf graph"""
        self.session.run(tf.global_variables_initializer())

        if self.evaluate_on_num_examples:
            logger.info("Accuracy is updated every {} epochs"
                        "".format(self.evaluate_every_num_epochs))

        pbar = tqdm(range(self.epochs), desc="Epochs")
        train_acc = 0
        last_loss = 0
        for ep in pbar:
            indices = np.random.permutation(len(X))

            batch_size = self._linearly_increasing_batch_size(ep)
            batches_per_epoch = (len(X) // batch_size +
                                 int(len(X) % batch_size > 0))

            ep_loss = 0
            for i in range(batches_per_epoch):
                end_idx = (i + 1) * batch_size
                start_idx = i * batch_size
                batch_a = X[indices[start_idx:end_idx]]
                batch_pos_b = Y[indices[start_idx:end_idx]]
                intents_for_b = intents_for_X[indices[start_idx:end_idx]]
                # add negatives
                batch_b = self._create_batch_b(batch_pos_b, intents_for_b)

                sess_out = self.session.run(
                        {'loss': loss, 'train_op': train_op},
                        feed_dict={self.a_in: batch_a,
                                   self.b_in: batch_b,
                                   is_training: True}
                )
                ep_loss += sess_out.get('loss') / batches_per_epoch

            if self.evaluate_on_num_examples:
                if (ep == 0 or
                        (ep + 1) % self.evaluate_every_num_epochs == 0 or
                        (ep + 1) == self.epochs):
                    train_acc = self._output_training_stat(X, intents_for_X,
                                                           is_training)
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

    def _output_training_stat(self, X, intents_for_X, is_training):
        """Output training statistics"""
        n = self.evaluate_on_num_examples
        ids = np.random.permutation(len(X))[:n]
        all_Y = self._create_all_Y(X[ids].shape[0])

        train_sim = self.session.run(self.sim_op,
                                     feed_dict={self.a_in: X[ids],
                                                self.b_in: all_Y,
                                                is_training: False})

        train_acc = np.mean(np.argmax(train_sim, -1) == intents_for_X[ids])
        return train_acc

    def train(self, training_data, cfg=None, **kwargs):
        # type: (TrainingData, Optional[RasaNLUModelConfig], **Any) -> None
        """Train the embedding intent classifier on a data set."""

        intent_dict = self._create_intent_dict(training_data)
        if len(intent_dict) < 2:
            logger.error("Can not train an intent classifier. "
                         "Need at least 2 different classes. "
                         "Skipping training of intent classifier.")
            return

        self.inv_intent_dict = {v: k for k, v in intent_dict.items()}
        self.encoded_all_intents = self._create_encoded_intents(
                                        intent_dict)

        X, Y, intents_for_X = self._prepare_data_for_training(
                                training_data, intent_dict)

        # check if number of negatives is less than number of intents
        logger.debug("Check if num_neg {} is smaller than "
                     "number of intents {}, "
                     "else set num_neg to the number of intents - 1"
                     "".format(self.num_neg,
                               self.encoded_all_intents.shape[0]))
        self.num_neg = min(self.num_neg,
                           self.encoded_all_intents.shape[0] - 1)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.a_in = tf.placeholder(tf.float32, (None, X.shape[-1]),
                                       name='a')
            self.b_in = tf.placeholder(tf.float32, (None, None, Y.shape[-1]),
                                       name='b')

            is_training = tf.placeholder_with_default(False, shape=())

            (self.word_embed,
             self.intent_embed) = self._create_tf_embed(self.a_in, self.b_in,
                                                        is_training)

            self.sim_op, sim_emb = self._tf_sim(self.word_embed,
                                                self.intent_embed)
            loss = self._tf_loss(self.sim_op, sim_emb)

            train_op = tf.train.AdamOptimizer().minimize(loss)

            # train tensorflow graph
            self.session = tf.Session()

            self._train_tf(X, Y, intents_for_X,
                           loss, is_training, train_op)

    # process helpers
    def _calculate_message_sim(self, X, all_Y):
        """Load tf graph and calculate message similarities"""

        message_sim = self.session.run(self.sim_op,
                                       feed_dict={self.a_in: X,
                                                  self.b_in: all_Y})
        message_sim = message_sim.flatten()  # sim is a matrix

        intent_ids = message_sim.argsort()[::-1]
        message_sim[::-1].sort()

        if self.similarity_type == 'cosine':
            # clip negative values to zero
            message_sim[message_sim < 0] = 0
        elif self.similarity_type == 'inner':
            # normalize result to [0, 1] with softmax
            message_sim = np.exp(message_sim)
            message_sim /= np.sum(message_sim)

        # transform sim to python list for JSON serializing
        return intent_ids, message_sim.tolist()

    def process(self, message, **kwargs):
        # type: (Message, **Any) -> None
        """Return the most likely intent and its similarity to the input."""

        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        if self.session is None:
            logger.error("There is no trained tf.session: "
                         "component is either not trained or "
                         "didn't receive enough training data")

        else:
            # get features (bag of words) for a message
            X = message.get("text_features").reshape(1, -1)

            # stack encoded_all_intents on top of each other
            # to create candidates for test examples
            all_Y = self._create_all_Y(X.shape[0])

            # load tf graph and session
            intent_ids, message_sim = self._calculate_message_sim(X, all_Y)

            # if X contains all zeros do not predict some label
            if X.any() and intent_ids.size > 0:
                intent = {"name": self.inv_intent_dict[intent_ids[0]],
                          "confidence": message_sim[0]}

                ranking = list(zip(list(intent_ids), message_sim))
                ranking = ranking[:INTENT_RANKING_LENGTH]
                intent_ranking = [{"name": self.inv_intent_dict[intent_idx],
                                   "confidence": score}
                                  for intent_idx, score in ranking]

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def persist(self, model_dir):
        # type: (Text) -> Dict[Text, Any]
        """Persist this model into the passed directory.
        Return the metadata necessary to load the model again."""
        if self.session is None:
            return {"classifier_file": None}

        checkpoint = os.path.join(model_dir, self.name + ".ckpt")

        try:
            os.makedirs(os.path.dirname(checkpoint))
        except OSError as e:
            # be happy if someone already created the path
            import errno
            if e.errno != errno.EEXIST:
                raise
        with self.graph.as_default():
            self.graph.clear_collection('message_placeholder')
            self.graph.add_to_collection('message_placeholder',
                                         self.a_in)

            self.graph.clear_collection('intent_placeholder')
            self.graph.add_to_collection('intent_placeholder',
                                         self.b_in)

            self.graph.clear_collection('similarity_op')
            self.graph.add_to_collection('similarity_op',
                                         self.sim_op)

            self.graph.clear_collection('word_embed')
            self.graph.add_to_collection('word_embed',
                                         self.word_embed)
            self.graph.clear_collection('intent_embed')
            self.graph.add_to_collection('intent_embed',
                                         self.intent_embed)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with io.open(os.path.join(
                model_dir,
                self.name + "_inv_intent_dict.pkl"), 'wb') as f:
            pickle.dump(self.inv_intent_dict, f)
        with io.open(os.path.join(
                model_dir,
                self.name + "_encoded_all_intents.pkl"), 'wb') as f:
            pickle.dump(self.encoded_all_intents, f)

        return {"classifier_file": self.name + ".ckpt"}

    @classmethod
    def load(cls,
             model_dir=None,  # type: Text
             model_metadata=None,  # type: Metadata
             cached_component=None,  # type: Optional[Component]
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

                a_in = tf.get_collection('message_placeholder')[0]
                b_in = tf.get_collection('intent_placeholder')[0]

                sim_op = tf.get_collection('similarity_op')[0]

                word_embed = tf.get_collection('word_embed')[0]
                intent_embed = tf.get_collection('intent_embed')[0]

            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_inv_intent_dict.pkl"), 'rb') as f:
                inv_intent_dict = pickle.load(f)
            with io.open(os.path.join(
                    model_dir,
                    cls.name + "_encoded_all_intents.pkl"), 'rb') as f:
                encoded_all_intents = pickle.load(f)

            return EmbeddingIntentClassifier(
                    component_config=meta,
                    inv_intent_dict=inv_intent_dict,
                    encoded_all_intents=encoded_all_intents,
                    session=sess,
                    graph=graph,
                    message_placeholder=a_in,
                    intent_placeholder=b_in,
                    similarity_op=sim_op,
                    word_embed=word_embed,
                    intent_embed=intent_embed
            )

        else:
            logger.warning("Failed to load nlu model. Maybe path {} "
                           "doesn't exist"
                           "".format(os.path.abspath(model_dir)))
            return EmbeddingIntentClassifier(component_config=meta)
