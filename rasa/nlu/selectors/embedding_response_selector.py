import io
import logging
import numpy as np
import os
import pickle
import typing
from typing import Any, Dict, Optional, Text

from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.utils import train_utils

from rasa.nlu.classifiers.embedding_intent_classifier import EmbeddingIntentClassifier
from rasa.constants import (
    DEFAULT_OPEN_UTTERANCE_TYPE_KEY,
    DEFAULT_OPEN_UTTERANCE_TYPE_KEY_RANKING,
)
from rasa.core.actions.action import RESPOND_PREFIX

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from tensorflow import Graph, Session, Tensor
    from rasa.nlu.config import RasaNLUModelConfig
    from rasa.nlu.training_data import TrainingData
    from rasa.nlu.model import Metadata
    from rasa.nlu.training_data import Message

try:
    import tensorflow as tf

    # avoid warning println on contrib import - remove for tf 2
    tf.contrib._warning = None
except ImportError:
    tf = None


class ResponseSelector(EmbeddingIntentClassifier):
    """Response selector using supervised embeddings.

        The response selector embeds user inputs
        and candidate response into the same space.
        Supervised embeddings are trained by maximizing similarity between them.
        It also provides rankings of the response that did not "win".

        The supervised response selector needs to be preceded by
        a featurizer in the pipeline.
        This featurizer creates the features used for the embeddings.
        It is recommended to use ``CountVectorsFeaturizer`` that
        can be optionally preceded by ``SpacyNLP`` and ``SpacyTokenizer``.

        Based on the starspace idea from: https://arxiv.org/abs/1709.03856.
        However, in this implementation the `mu` parameter is treated differently
        and additional hidden layers are added together with dropout.
        """

    provides = ["response", "response_ranking"]

    requires = ["text_features"]

    name = "ResponseSelector"

    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer for input words
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_a": [256, 128],
        # sizes of hidden layers before the embedding layer for intent labels
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_b": [],
        # training parameters
        # initial and final batch sizes - batch size will be
        # linearly increased for each epoch
        "batch_size": [64, 256],
        # how to create batches
        "batch_strategy": "balanced",  # string 'sequence' or 'balanced'
        # number of epochs
        "epochs": 300,
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
        # to make embedding vectors for correct intent labels
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect intent labels
        "mu_neg": -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
        # flag: if true, only minimize the maximum similarity for
        # incorrect intent labels
        "use_max_sim_neg": True,
        # scale loss inverse proportionally to confidence of correct prediction
        "scale_loss": True,
        # regularization parameters
        # the scale of L2 regularization
        "C2": 0.002,
        # the scale of how critical the algorithm should be of minimizing the
        # maximum similarity between embeddings of different intent labels
        "C_emb": 0.8,
        # dropout rate for rnn
        "droprate": 0.2,
        "label_tokenization_flag": False,
        # delimiter string to split the labels
        "label_split_symbol": "_",
        # visualization of accuracy
        # how often to calculate training accuracy
        "evaluate_every_num_epochs": 20,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        "evaluate_on_num_examples": 0,  # large values may hurt performance,
        # Tensorboard config
        "summary_dir": os.path.join(os.getcwd(), "tb_logs"),
        # selector config
        "response_type": None,
    }
    # end default properties (DOC MARKER - don't remove)

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        inv_label_dict: Optional[Dict[int, Text]] = None,
        session: Optional["tf.Session"] = None,
        graph: Optional["tf.Graph"] = None,
        message_placeholder: Optional["tf.Tensor"] = None,
        label_placeholder: Optional["tf.Tensor"] = None,
        sim_all: Optional["tf.Tensor"] = None,
        pred_confidence: Optional["tf.Tensor"] = None,
        similarity_op: Optional["tf.Tensor"] = None,
        message_embed: Optional["tf.Tensor"] = None,
        label_embed: Optional["tf.Tensor"] = None,
        all_labels_embed: Optional["tf.Tensor"] = None,
    ) -> None:
        super(ResponseSelector, self).__init__(
            component_config,
            inv_label_dict,
            session,
            graph,
            message_placeholder,
            label_placeholder,
            sim_all,
            pred_confidence,
            similarity_op,
            message_embed,
            label_embed,
            all_labels_embed,
        )

    def _load_tb_params(self, config: Dict[Text, Any]) -> None:
        self.summary_dir = config["summary_dir"]

    def _load_selector_params(self, config: Dict[Text, Any]):
        self.response_type = config["response_type"]

    def _load_params(self) -> None:
        super(ResponseSelector, self)._load_params()
        self._load_tb_params(self.component_config)
        self._load_selector_params(self.component_config)

    def process(self, message: "Message", **kwargs: Any) -> None:
        """Return the most likely intent and its similarity to the input."""

        label = {"name": None, "confidence": 0.0}
        label_ranking = []

        if self.session is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data"
            )

        else:
            # get features (bag of words) for a message
            # noinspection PyPep8Naming
            X = message.get("text_features").reshape(1, -1)

            # load tf graph and session
            label_ids, message_sim = self._calculate_message_sim(X)

            # if X contains all zeros do not predict some label
            if X.any() and label_ids.size > 0:
                label = {
                    "name": self.inverted_label_dict[label_ids[0]],
                    "confidence": message_sim[0],
                }

                ranking = list(zip(list(label_ids), message_sim))
                ranking = ranking[:LABEL_RANKING_LENGTH]
                label_ranking = [
                    {"name": self.inverted_label_dict[label_idx], "confidence": score}
                    for label_idx, score in ranking
                ]

        if self.response_type:
            response_key_for_tracker = "{0}{1}_response".format(
                RESPOND_PREFIX, self.response_type
            )
            response_ranking_key_for_tracker = "{0}{1}_response_ranking".format(
                RESPOND_PREFIX, self.response_type
            )
        else:
            response_key_for_tracker = DEFAULT_OPEN_UTTERANCE_TYPE_KEY
            response_ranking_key_for_tracker = DEFAULT_OPEN_UTTERANCE_TYPE_KEY_RANKING

        message.set(response_key_for_tracker, label, add_to_output=True)
        message.set(response_ranking_key_for_tracker, label_ranking, add_to_output=True)

    # training data helpers:
    @staticmethod
    def _create_label_dict(
        training_data: "TrainingData", attribute: Text = "intent"
    ) -> Dict[Text, int]:
        """Create intent dictionary"""

        distinct_labels = set(
            [
                example.get(attribute)
                for example in training_data.intent_examples
                if example.get(attribute)
            ]
        )
        return {response: idx for idx, response in enumerate(sorted(distinct_labels))}

    @staticmethod
    def _find_example_for_label(label, examples, label_type="intent"):
        for ex in examples:
            if ex.get(label_type) == label:
                return ex

    # @staticmethod
    def _create_encoded_labels(
        self,
        label_dict: Dict[Text, int],
        training_data: "TrainingData",
        attribute: Text = "intent",
        attribute_features: Text = "intent_features",
    ) -> np.ndarray:
        """Create matrix with intents encoded in rows as bag of words.
        """

        encoded_all_labels = []

        for label_name, idx in label_dict.items():
            encoded_all_labels.insert(
                idx,
                self._find_example_for_label(
                    label_name, training_data.intent_examples, attribute
                ).get(attribute_features),
            )

        return np.array(encoded_all_labels)

    # noinspection PyPep8Naming
    def _create_session_data(
        self,
        training_data: "TrainingData",
        label_dict: Dict[Text, int],
        attribute: Text = "intent",
    ) -> "train_utils.SessionData":
        """Prepare data for training"""

        X = np.stack(
            [
                e.get("text_features")
                for e in training_data.intent_examples
                if e.get(attribute)
            ]
        )

        label_ids = np.array(
            [
                label_dict[e.get(attribute)]
                for e in training_data.intent_examples
                if e.get(attribute)
            ]
        )

        Y = np.stack([self._encoded_all_label_ids[label] for label in label_ids])

        return train_utils.SessionData(X=X, Y=Y, label_ids=label_ids)

    def train(
        self,
        training_data: "TrainingData",
        config: Optional["RasaNLUModelConfig"] = None,
        **kwargs: Any
    ) -> None:
        """Train the embedding intent classifier on a data set."""

        if self.response_type:
            training_data = training_data.filter_by_intent(self.response_type)

        label_dict = self._create_label_dict(training_data, attribute="response")

        if len(label_dict) < 2:
            logger.error(
                "Can not train a response selector. "
                "Need at least 2 different classes. "
                "Skipping training of response selector."
            )
            return

        self.inverted_label_dict = {
            v: k for k, v in label_dict.items()
        }  # idx: response
        self._encoded_all_label_ids = self._create_encoded_labels(
            label_dict,
            training_data,
            attribute="response",
            attribute_features="response_features",
        )

        # check if number of negatives is less than number of intents
        logger.debug(
            "Check if num_neg {} is smaller than "
            "number of intents {}, "
            "else set num_neg to the number of intents - 1"
            "".format(self.num_neg, self._encoded_all_label_ids.shape[0])
        )

        # noinspection PyAttributeOutsideInit
        self.num_neg = min(self.num_neg, self._encoded_all_label_ids.shape[0] - 1)

        session_data = self._create_session_data(
            training_data, label_dict, attribute="response"
        )

        if self.evaluate_on_num_examples:
            session_data, eval_session_data = train_utils.train_val_split(
                session_data, self.evaluate_on_num_examples, self.random_seed
            )
        else:
            eval_session_data = None

        # if self.share_embedding:
        #     if X[0].shape[-1] != Y[0].shape[-1]:
        #         raise ValueError("If embeddings are shared "
        #                          "text features and intent features "
        #                          "must coincide")

        self.graph = tf.Graph()
        with self.graph.as_default():

            # set random seed
            tf.set_random_seed(self.random_seed)

            # allows increasing batch size
            batch_size_in = tf.placeholder(tf.int64)

            (
                self._iterator,
                train_init_op,
                eval_init_op,
            ) = train_utils.create_iterator_init_datasets(
                session_data, eval_session_data, batch_size_in, self.batch_strategy
            )

            self._is_training = tf.placeholder_with_default(False, shape=())

            loss, acc = self._build_tf_train_graph()

            # define which optimizer to use
            self._train_op = tf.train.AdamOptimizer().minimize(loss)

            # train tensorflow graph
            self.session = tf.Session()
            train_utils.train_tf_dataset(
                train_init_op,
                eval_init_op,
                batch_size_in,
                loss,
                acc,
                self._train_op,
                self.session,
                self._is_training,
                self.epochs,
                self.batch_size,
                self.evaluate_on_num_examples,
                self.evaluate_every_num_epochs,
            )

            # rebuild the graph for prediction
            self.pred_confidence = self._build_tf_pred_graph(session_data)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.Return the metadata necessary to load the model again."""

        if self.session is None:
            return {"file": None}

        checkpoint = os.path.join(model_dir, file_name + ".ckpt")

        try:
            os.makedirs(os.path.dirname(checkpoint))
        except OSError as e:
            # be happy if someone already created the path
            import errno

            if e.errno != errno.EEXIST:
                raise
        with self.graph.as_default():
            train_utils.persist_tensor("message_placeholder", self.a_in, self.graph)
            train_utils.persist_tensor("label_placeholder", self.b_in, self.graph)

            train_utils.persist_tensor("similarity_all", self.sim_all, self.graph)
            train_utils.persist_tensor(
                "pred_confidence", self.pred_confidence, self.graph
            )
            train_utils.persist_tensor("similarity", self.sim, self.graph)

            train_utils.persist_tensor("message_embed", self.message_embed, self.graph)
            train_utils.persist_tensor("label_embed", self.label_embed, self.graph)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with io.open(
            os.path.join(model_dir, file_name + "_inv_label_dict.pkl"), "wb"
        ) as f:
            pickle.dump(self.inverted_label_dict, f)

        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: "Metadata" = None,
        cached_component: Optional["ResponseSelector"] = None,
        **kwargs: Any
    ) -> "ResponseSelector":

        if model_dir and meta.get("file"):
            file_name = meta.get("file")
            checkpoint = os.path.join(model_dir, file_name + ".ckpt")
            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session()
                saver = tf.train.import_meta_graph(checkpoint + ".meta")

                saver.restore(sess, checkpoint)

                a_in = train_utils.load_tensor("message_placeholder")
                b_in = train_utils.load_tensor("label_placeholder")

                sim_all = train_utils.load_tensor("similarity_all")
                pred_confidence = train_utils.load_tensor("pred_confidence")
                sim = train_utils.load_tensor("similarity")

                message_embed = train_utils.load_tensor("message_embed")
                label_embed = train_utils.load_tensor("label_embed")
                all_labels_embed = train_utils.load_tensor("all_labels_embed")

            with io.open(
                os.path.join(model_dir, file_name + "_inv_label_dict.pkl"), "rb"
            ) as f:
                inv_label_dict = pickle.load(f)

            return cls(
                component_config=meta,
                inv_label_dict=inv_label_dict,
                session=sess,
                graph=graph,
                message_placeholder=a_in,
                label_placeholder=b_in,
                sim_all=sim_all,
                pred_confidence=pred_confidence,
                similarity_op=sim,
                message_embed=message_embed,
                label_embed=label_embed,
                all_labels_embed=all_labels_embed,
            )

        else:
            logger.warning(
                "Failed to load model. Maybe path {} "
                "doesn't exist"
                "".format(os.path.abspath(model_dir))
            )
            return cls(component_config=meta)
