import logging
import numpy as np
import os
import pickle
import typing
from typing import Any, Dict, List, Optional, Text, Tuple
import warnings

from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.components import Component
from rasa.utils import train_utils
from rasa.nlu.constants import (
    MESSAGE_INTENT_ATTRIBUTE,
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_VECTOR_FEATURE_NAMES,
)

import tensorflow as tf

# avoid warning println on contrib import - remove for tf 2
tf.contrib._warning = None

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.nlu.config import RasaNLUModelConfig
    from rasa.nlu.training_data import TrainingData
    from rasa.nlu.model import Metadata
    from rasa.nlu.training_data import Message


class EmbeddingIntentClassifier(Component):
    """Intent classifier using supervised embeddings.

    The embedding intent classifier embeds user inputs
    and intent labels into the same space.
    Supervised embeddings are trained by maximizing similarity between them.
    It also provides rankings of the labels that did not "win".

    The embedding intent classifier needs to be preceded by
    a featurizer in the pipeline.
    This featurizer creates the features used for the embeddings.
    It is recommended to use ``CountVectorsFeaturizer`` that
    can be optionally preceded by ``SpacyNLP`` and ``SpacyTokenizer``.

    Based on the starspace idea from: https://arxiv.org/abs/1709.03856.
    However, in this implementation the `mu` parameter is treated differently
    and additional hidden layers are added together with dropout.
    """

    provides = ["intent", "intent_ranking"]

    requires = [MESSAGE_VECTOR_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE]]

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer for input words
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_a": [256, 128],
        # sizes of hidden layers before the embedding layer for intent labels
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_b": [],
        # Whether to share the hidden layer weights between input words and labels
        "share_hidden_layers": False,
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
        # to make embedding vectors for correct labels
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect labels
        "mu_neg": -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
        # flag: if true, only minimize the maximum similarity for incorrect labels
        "use_max_sim_neg": True,
        # scale loss inverse proportionally to confidence of correct prediction
        "scale_loss": True,
        # regularization parameters
        # the scale of L2 regularization
        "C2": 0.002,
        # the scale of how critical the algorithm should be of minimizing the
        # maximum similarity between embeddings of different labels
        "C_emb": 0.8,
        # dropout rate for rnn
        "droprate": 0.2,
        # visualization of accuracy
        # how often to calculate training accuracy
        "evaluate_every_num_epochs": 20,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        "evaluate_on_num_examples": 0,  # large values may hurt performance
    }
    # end default properties (DOC MARKER - don't remove)

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        inverted_label_dict: Optional[Dict[int, Text]] = None,
        session: Optional["tf.Session"] = None,
        graph: Optional["tf.Graph"] = None,
        message_placeholder: Optional["tf.Tensor"] = None,
        label_placeholder: Optional["tf.Tensor"] = None,
        similarity_all: Optional["tf.Tensor"] = None,
        pred_confidence: Optional["tf.Tensor"] = None,
        similarity: Optional["tf.Tensor"] = None,
        message_embed: Optional["tf.Tensor"] = None,
        label_embed: Optional["tf.Tensor"] = None,
        all_labels_embed: Optional["tf.Tensor"] = None,
    ) -> None:
        """Declare instant variables with default values"""

        super().__init__(component_config)

        self._load_params()

        # transform numbers to labels
        self.inverted_label_dict = inverted_label_dict
        # encode all label_ids with numbers
        self._encoded_all_label_ids = None

        # tf related instances
        self.session = session
        self.graph = graph
        self.a_in = message_placeholder
        self.b_in = label_placeholder
        self.sim_all = similarity_all
        self.pred_confidence = pred_confidence
        self.sim = similarity

        # persisted embeddings
        self.message_embed = message_embed
        self.label_embed = label_embed
        self.all_labels_embed = all_labels_embed

        # internal tf instances
        self._iterator = None
        self._train_op = None
        self._is_training = None

    # config migration warning
    def _check_old_config_variables(self, config: Dict[Text, Any]) -> None:

        removed_tokenization_params = [
            "intent_tokenization_flag",
            "intent_split_symbol",
        ]
        for removed_param in removed_tokenization_params:
            if removed_param in config:
                warnings.warn(
                    f"Intent tokenization has been moved to Tokenizer components. "
                    f"Your config still mentions '{removed_param}'. Tokenization may fail if you specify the parameter here. "
                    f"Please specify the parameter 'intent_tokenization_flag' and 'intent_split_symbol' in the "
                    f"tokenizer of your NLU pipeline",
                    DeprecationWarning,
                )

    # init helpers
    def _load_nn_architecture_params(self, config: Dict[Text, Any]) -> None:
        self.hidden_layer_sizes = {
            "a": config["hidden_layers_sizes_a"],
            "b": config["hidden_layers_sizes_b"],
        }
        self.share_hidden_layers = config["share_hidden_layers"]
        if (
            self.share_hidden_layers
            and self.hidden_layer_sizes["a"] != self.hidden_layer_sizes["b"]
        ):
            raise ValueError(
                "If hidden layer weights are shared,"
                "hidden_layer_sizes for a and b must coincide"
            )

        self.batch_size = config["batch_size"]
        self.batch_strategy = config["batch_strategy"]

        self.epochs = config["epochs"]

        self.random_seed = self.component_config["random_seed"]

    def _load_embedding_params(self, config: Dict[Text, Any]) -> None:
        self.embed_dim = config["embed_dim"]
        self.num_neg = config["num_neg"]

        self.similarity_type = config["similarity_type"]
        self.loss_type = config["loss_type"]
        if self.similarity_type == "auto":
            if self.loss_type == "softmax":
                self.similarity_type = "inner"
            elif self.loss_type == "margin":
                self.similarity_type = "cosine"

        self.mu_pos = config["mu_pos"]
        self.mu_neg = config["mu_neg"]
        self.use_max_sim_neg = config["use_max_sim_neg"]

        self.scale_loss = config["scale_loss"]

    def _load_regularization_params(self, config: Dict[Text, Any]) -> None:
        self.C2 = config["C2"]
        self.C_emb = config["C_emb"]
        self.droprate = config["droprate"]

    def _load_visual_params(self, config: Dict[Text, Any]) -> None:
        self.evaluate_every_num_epochs = config["evaluate_every_num_epochs"]
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs
        self.evaluate_on_num_examples = config["evaluate_on_num_examples"]

    def _load_params(self) -> None:

        self._check_old_config_variables(self.component_config)
        self._tf_config = train_utils.load_tf_config(self.component_config)
        self._load_nn_architecture_params(self.component_config)
        self._load_embedding_params(self.component_config)
        self._load_regularization_params(self.component_config)
        self._load_visual_params(self.component_config)

    # package safety checks
    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow"]

    # training data helpers:
    @staticmethod
    def _create_label_id_dict(
        training_data: "TrainingData", attribute: Text
    ) -> Dict[Text, int]:
        """Create label_id dictionary"""

        distinct_label_ids = {
            example.get(attribute) for example in training_data.intent_examples
        } - {None}
        return {
            label_id: idx for idx, label_id in enumerate(sorted(distinct_label_ids))
        }

    @staticmethod
    def _find_example_for_label(label, examples, attribute):
        for ex in examples:
            if ex.get(attribute) == label:
                return ex
        return None

    @staticmethod
    def _check_labels_features_exist(
        labels_example: List[Tuple[int, "Message"]], attribute_feature_name: Text
    ) -> bool:
        """Check if all labels have features set"""
        for (label_idx, label_example) in labels_example:
            if label_example.get(attribute_feature_name) is None:
                return False
        return True

    @staticmethod
    def _extract_labels_precomputed_features(
        label_examples: List[Tuple[int, "Message"]], attribute_feature_name: Text
    ) -> np.ndarray:

        # Collect precomputed encodings
        encoded_id_labels = [
            (label_idx, label_example.get(attribute_feature_name))
            for (label_idx, label_example) in label_examples
        ]

        # Sort the list of tuples based on label_idx
        encoded_id_labels = sorted(encoded_id_labels, key=lambda x: x[0])

        encoded_all_labels = [encoding for (index, encoding) in encoded_id_labels]

        return np.array(encoded_all_labels)

    def _compute_default_label_features(
        self, labels_example: List[Tuple[int, "Message"]]
    ) -> np.ndarray:
        """Compute one-hot representation for the labels"""

        return np.eye(len(labels_example))

    def _create_encoded_label_ids(
        self,
        training_data: "TrainingData",
        label_id_dict: Dict[Text, int],
        attribute: Text,
        attribute_feature_name: Text,
    ) -> np.ndarray:
        """Create matrix with label_ids encoded in rows as bag of words. If the features are already computed, fetch
        them from the message object else compute a one hot encoding for the label as the feature vector
        Find a training example for each label and get the encoded features from the corresponding Message object"""

        labels_example = []

        # Collect one example for each label
        for label_name, idx in label_id_dict.items():
            label_example = self._find_example_for_label(
                label_name, training_data.intent_examples, attribute
            )
            labels_example.append((idx, label_example))

        # Collect features, precomputed if they exist, else compute on the fly
        if self._check_labels_features_exist(labels_example, attribute_feature_name):
            encoded_id_labels = self._extract_labels_precomputed_features(
                labels_example, attribute_feature_name
            )
        else:
            encoded_id_labels = self._compute_default_label_features(labels_example)

        return encoded_id_labels

    # noinspection PyPep8Naming
    def _create_session_data(
        self,
        training_data: "TrainingData",
        label_id_dict: Dict[Text, int],
        attribute: Text,
    ) -> "train_utils.SessionData":
        """Prepare data for training and create a SessionData object"""

        X = []
        label_ids = []
        Y = []

        for e in training_data.intent_examples:
            if e.get(attribute):
                X.append(e.get(MESSAGE_VECTOR_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE]))
                label_ids.append(label_id_dict[e.get(attribute)])

        X = np.array(X)
        label_ids = np.array(label_ids)

        for label_id_idx in label_ids:
            Y.append(self._encoded_all_label_ids[label_id_idx])

        Y = np.array(Y)

        return train_utils.SessionData(X=X, Y=Y, label_ids=label_ids)

    # tf helpers:
    def _create_tf_embed_fnn(
        self,
        x_in: "tf.Tensor",
        layer_sizes: List[int],
        fnn_name: Text,
        embed_name: Text,
    ) -> "tf.Tensor":
        """Create nn with hidden layers and name"""

        x = train_utils.create_tf_fnn(
            x_in,
            layer_sizes,
            self.droprate,
            self.C2,
            self._is_training,
            layer_name_suffix=fnn_name,
        )
        return train_utils.create_tf_embed(
            x,
            self.embed_dim,
            self.C2,
            self.similarity_type,
            layer_name_suffix=embed_name,
        )

    def _build_tf_train_graph(self) -> Tuple["tf.Tensor", "tf.Tensor"]:
        self.a_in, self.b_in = self._iterator.get_next()

        all_label_ids = tf.constant(
            self._encoded_all_label_ids, dtype=tf.float32, name="all_label_ids"
        )

        self.message_embed = self._create_tf_embed_fnn(
            self.a_in,
            self.hidden_layer_sizes["a"],
            fnn_name="a_b" if self.share_hidden_layers else "a",
            embed_name="a",
        )

        self.label_embed = self._create_tf_embed_fnn(
            self.b_in,
            self.hidden_layer_sizes["b"],
            fnn_name="a_b" if self.share_hidden_layers else "b",
            embed_name="b",
        )
        self.all_labels_embed = self._create_tf_embed_fnn(
            all_label_ids,
            self.hidden_layer_sizes["b"],
            fnn_name="a_b" if self.share_hidden_layers else "b",
            embed_name="b",
        )

        return train_utils.calculate_loss_acc(
            self.message_embed,
            self.label_embed,
            self.b_in,
            self.all_labels_embed,
            all_label_ids,
            self.num_neg,
            None,
            self.loss_type,
            self.mu_pos,
            self.mu_neg,
            self.use_max_sim_neg,
            self.C_emb,
            self.scale_loss,
        )

    def _build_tf_pred_graph(
        self, session_data: "train_utils.SessionData"
    ) -> "tf.Tensor":
        self.a_in = tf.placeholder(
            tf.float32, (None, session_data.X.shape[-1]), name="a"
        )
        self.b_in = tf.placeholder(
            tf.float32, (None, None, session_data.Y.shape[-1]), name="b"
        )

        self.message_embed = self._create_tf_embed_fnn(
            self.a_in,
            self.hidden_layer_sizes["a"],
            fnn_name="a_b" if self.share_hidden_layers else "a",
            embed_name="a",
        )

        self.sim_all = train_utils.tf_raw_sim(
            self.message_embed[:, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, :, :],
            None,
        )

        self.label_embed = self._create_tf_embed_fnn(
            self.b_in,
            self.hidden_layer_sizes["b"],
            fnn_name="a_b" if self.share_hidden_layers else "b",
            embed_name="b",
        )

        self.sim = train_utils.tf_raw_sim(
            self.message_embed[:, tf.newaxis, :], self.label_embed, None
        )

        return train_utils.confidence_from_sim(self.sim_all, self.similarity_type)

    def check_input_dimension_consistency(self, session_data):

        if self.share_hidden_layers:
            if session_data.X[0].shape[-1] != session_data.Y[0].shape[-1]:
                raise ValueError(
                    "If embeddings are shared "
                    "text features and label features "
                    "must coincide. Check the output dimensions of previous components."
                )

    def preprocess_train_data(self, training_data):
        """Performs sanity checks on training data, extracts encodings for labels and prepares data for training"""

        label_id_dict = self._create_label_id_dict(
            training_data, attribute=MESSAGE_INTENT_ATTRIBUTE
        )

        self.inverted_label_dict = {v: k for k, v in label_id_dict.items()}
        self._encoded_all_label_ids = self._create_encoded_label_ids(
            training_data,
            label_id_dict,
            attribute=MESSAGE_INTENT_ATTRIBUTE,
            attribute_feature_name=MESSAGE_VECTOR_FEATURE_NAMES[
                MESSAGE_INTENT_ATTRIBUTE
            ],
        )

        # check if number of negatives is less than number of label_ids
        logger.debug(
            "Check if num_neg {} is smaller than "
            "number of label_ids {}, "
            "else set num_neg to the number of label_ids - 1"
            "".format(self.num_neg, self._encoded_all_label_ids.shape[0])
        )
        # noinspection PyAttributeOutsideInit
        self.num_neg = min(self.num_neg, self._encoded_all_label_ids.shape[0] - 1)

        session_data = self._create_session_data(
            training_data, label_id_dict, attribute=MESSAGE_INTENT_ATTRIBUTE
        )

        self.check_input_dimension_consistency(session_data)

        return session_data

    def _check_enough_labels(self, session_data) -> bool:

        return len(np.unique(session_data.label_ids)) >= 2

    def train(
        self,
        training_data: "TrainingData",
        cfg: Optional["RasaNLUModelConfig"] = None,
        **kwargs: Any,
    ) -> None:
        """Train the embedding label classifier on a data set."""

        logger.debug("Started training embedding classifier.")

        # set numpy random seed
        np.random.seed(self.random_seed)

        session_data = self.preprocess_train_data(training_data)

        possible_to_train = self._check_enough_labels(session_data)

        if not possible_to_train:
            logger.error(
                "Can not train a classifier. "
                "Need at least 2 different classes. "
                "Skipping training of classifier."
            )
            return

        if self.evaluate_on_num_examples:
            session_data, eval_session_data = train_utils.train_val_split(
                session_data, self.evaluate_on_num_examples, self.random_seed
            )
        else:
            eval_session_data = None

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
            self.session = tf.Session(config=self._tf_config)
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

    # process helpers
    # noinspection PyPep8Naming
    def _calculate_message_sim(self, X: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Calculate message similarities"""

        message_sim = self.session.run(self.pred_confidence, feed_dict={self.a_in: X})

        message_sim = message_sim.flatten()  # sim is a matrix

        label_ids = message_sim.argsort()[::-1]
        message_sim[::-1].sort()

        # transform sim to python list for JSON serializing
        return label_ids, message_sim.tolist()

    def predict_label(self, message):

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
            X = message.get(
                MESSAGE_VECTOR_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE]
            ).reshape(1, -1)

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
        return label, label_ranking

    def process(self, message: "Message", **kwargs: Any) -> None:
        """Return the most likely label and its similarity to the input."""

        label, label_ranking = self.predict_label(message)

        message.set("intent", label, add_to_output=True)
        message.set("intent_ranking", label_ranking, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

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
            train_utils.persist_tensor(
                "all_labels_embed", self.all_labels_embed, self.graph
            )

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with open(
            os.path.join(model_dir, file_name + ".inv_label_dict.pkl"), "wb"
        ) as f:
            pickle.dump(self.inverted_label_dict, f)

        with open(os.path.join(model_dir, file_name + ".tf_config.pkl"), "wb") as f:
            pickle.dump(self._tf_config, f)

        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: "Metadata" = None,
        cached_component: Optional["EmbeddingIntentClassifier"] = None,
        **kwargs: Any,
    ) -> "EmbeddingIntentClassifier":

        if model_dir and meta.get("file"):
            file_name = meta.get("file")
            checkpoint = os.path.join(model_dir, file_name + ".ckpt")

            with open(os.path.join(model_dir, file_name + ".tf_config.pkl"), "rb") as f:
                _tf_config = pickle.load(f)

            graph = tf.Graph()
            with graph.as_default():
                session = tf.compat.v1.Session(config=_tf_config)
                saver = tf.compat.v1.train.import_meta_graph(checkpoint + ".meta")

                saver.restore(session, checkpoint)

                a_in = train_utils.load_tensor("message_placeholder")
                b_in = train_utils.load_tensor("label_placeholder")

                sim_all = train_utils.load_tensor("similarity_all")
                pred_confidence = train_utils.load_tensor("pred_confidence")
                sim = train_utils.load_tensor("similarity")

                message_embed = train_utils.load_tensor("message_embed")
                label_embed = train_utils.load_tensor("label_embed")
                all_labels_embed = train_utils.load_tensor("all_labels_embed")

            with open(
                os.path.join(model_dir, file_name + ".inv_label_dict.pkl"), "rb"
            ) as f:
                inv_label_dict = pickle.load(f)

            return cls(
                component_config=meta,
                inverted_label_dict=inv_label_dict,
                session=session,
                graph=graph,
                message_placeholder=a_in,
                label_placeholder=b_in,
                similarity_all=sim_all,
                pred_confidence=pred_confidence,
                similarity=sim,
                message_embed=message_embed,
                label_embed=label_embed,
                all_labels_embed=all_labels_embed,
            )

        else:
            warnings.warn(
                f"Failed to load nlu model. Maybe path '{os.path.abspath(model_dir)}' "
                "doesn't exist."
            )
            return cls(component_config=meta)
