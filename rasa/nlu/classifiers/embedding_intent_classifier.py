import logging

import numpy as np
import os
import pickle
import scipy.sparse
import typing
from typing import Any, Dict, List, Optional, Text, Tuple, Union

from rasa.nlu.featurizers.featurizer import sequence_to_sentence_features
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.components import Component, any_of
from rasa.utils import train_utils
from rasa.utils.train_utils import SessionDataType
from rasa.nlu.constants import (
    INTENT_ATTRIBUTE,
    TEXT_ATTRIBUTE,
    SPARSE_FEATURE_NAMES,
    DENSE_FEATURE_NAMES,
)

import tensorflow as tf

# avoid warning println on contrib import - remove for tf 2
from rasa.utils.common import raise_warning

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

    requires = [
        any_of(
            DENSE_FEATURE_NAMES[TEXT_ATTRIBUTE], SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE]
        )
    ]

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
        # default dense dimension used if no dense features are present
        "dense_dim": {"text": 512, "label": 20},
        # dimension size of embedding vectors
        "embed_dim": 20,
        # the type of the similarity
        "num_neg": 20,
        # flag if minimize only maximum similarity over incorrect actions
        "similarity_type": "auto",  # string 'auto' or 'cosine' or 'inner'
        # the type of the loss function
        "loss_type": "softmax",  # string 'softmax' or 'margin'
        # number of top intents to normalize scores for softmax loss_type
        # set to 0 to turn off normalization
        "ranking_length": 10,
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

    @staticmethod
    def _check_old_config_variables(config: Dict[Text, Any]) -> None:
        """Config migration warning"""

        removed_tokenization_params = [
            "intent_tokenization_flag",
            "intent_split_symbol",
        ]
        for removed_param in removed_tokenization_params:
            if removed_param in config:
                raise_warning(
                    f"Intent tokenization has been moved to Tokenizer components. "
                    f"Your config still mentions '{removed_param}'. "
                    f"Tokenization may fail if you specify the parameter here. "
                    f"Please specify the parameter 'intent_tokenization_flag' "
                    f"and 'intent_split_symbol' in the "
                    f"tokenizer of your NLU pipeline",
                    FutureWarning,
                )

    # init helpers
    def _load_nn_architecture_params(self, config: Dict[Text, Any]) -> None:
        self.hidden_layer_sizes = {
            "text": config["hidden_layers_sizes_a"],
            "label": config["hidden_layers_sizes_b"],
        }
        self.share_hidden_layers = config["share_hidden_layers"]
        if (
            self.share_hidden_layers
            and self.hidden_layer_sizes["text"] != self.hidden_layer_sizes["label"]
        ):
            raise ValueError(
                "If hidden layer weights are shared,"
                "hidden_layer_sizes for a and b must coincide."
            )

        self.batch_in_size = config["batch_size"]
        self.batch_in_strategy = config["batch_strategy"]

        self.epochs = config["epochs"]

        self.random_seed = self.component_config["random_seed"]

    def _load_embedding_params(self, config: Dict[Text, Any]) -> None:
        self.embed_dim = config["embed_dim"]
        self.num_neg = config["num_neg"]
        self.dense_dim = config["dense_dim"]

        self.similarity_type = config["similarity_type"]
        self.loss_type = config["loss_type"]
        if self.similarity_type == "auto":
            if self.loss_type == "softmax":
                self.similarity_type = "inner"
            elif self.loss_type == "margin":
                self.similarity_type = "cosine"

        self.ranking_length = config["ranking_length"]
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

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        inverted_label_dict: Optional[Dict[int, Text]] = None,
        session: Optional["tf.Session"] = None,
        graph: Optional["tf.Graph"] = None,
        batch_placeholder: Optional["tf.Tensor"] = None,
        similarity_all: Optional["tf.Tensor"] = None,
        pred_confidence: Optional["tf.Tensor"] = None,
        similarity: Optional["tf.Tensor"] = None,
        message_embed: Optional["tf.Tensor"] = None,
        label_embed: Optional["tf.Tensor"] = None,
        all_labels_embed: Optional["tf.Tensor"] = None,
        batch_tuple_sizes: Optional[Dict] = None,
    ) -> None:
        """Declare instance variables with default values"""

        super().__init__(component_config)

        self._load_params()

        # transform numbers to labels
        self.inverted_label_dict = inverted_label_dict
        # encode all label_ids with numbers
        self._label_data = None

        # tf related instances
        self.session = session
        self.graph = graph
        self.batch_in = batch_placeholder
        self.sim_all = similarity_all
        self.pred_confidence = pred_confidence
        self.sim = similarity

        # persisted embeddings
        self.message_embed = message_embed
        self.label_embed = label_embed
        self.all_labels_embed = all_labels_embed

        # keep the input tuple sizes in self.batch_in
        self.batch_tuple_sizes = batch_tuple_sizes

        # internal tf instances
        self._iterator = None
        self._train_op = None
        self._is_training = None

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
    def _find_example_for_label(
        label: Text, examples: List["Message"], attribute: Text
    ) -> Optional["Message"]:
        for ex in examples:
            if ex.get(attribute) == label:
                return ex
        return None

    @staticmethod
    def _check_labels_features_exist(
        labels_example: List["Message"], attribute: Text
    ) -> bool:
        """Check if all labels have features set"""

        for label_example in labels_example:
            if (
                label_example.get(SPARSE_FEATURE_NAMES[attribute]) is None
                and label_example.get(DENSE_FEATURE_NAMES[attribute]) is None
            ):
                return False
        return True

    @staticmethod
    def _extract_and_add_features(
        message: "Message", attribute: Text
    ) -> Tuple[Optional[scipy.sparse.spmatrix], Optional[np.ndarray]]:
        sparse_features = None
        dense_features = None

        if message.get(SPARSE_FEATURE_NAMES[attribute]) is not None:
            sparse_features = message.get(SPARSE_FEATURE_NAMES[attribute])

        if message.get(DENSE_FEATURE_NAMES[attribute]) is not None:
            dense_features = message.get(DENSE_FEATURE_NAMES[attribute])

        if sparse_features is not None and dense_features is not None:
            if sparse_features.shape[0] != dense_features.shape[0]:
                raise ValueError(
                    f"Sequence dimensions for sparse and dense features "
                    f"don't coincide in '{message.text}' for attribute '{attribute}'."
                )

        if attribute != INTENT_ATTRIBUTE:
            # Use only the CLS token vector as features
            sparse_features = sequence_to_sentence_features(sparse_features)
            dense_features = sequence_to_sentence_features(dense_features)

        return sparse_features, dense_features

    def _extract_labels_precomputed_features(
        self, label_examples: List["Message"], attribute: Text = INTENT_ATTRIBUTE
    ) -> List[np.ndarray]:
        """Collect precomputed encodings"""

        sparse_features = []
        dense_features = []

        for e in label_examples:
            _sparse, _dense = self._extract_and_add_features(e, attribute)
            if _sparse is not None:
                sparse_features.append(_sparse)
            if _dense is not None:
                dense_features.append(_dense)

        sparse_features = np.array(sparse_features)
        dense_features = np.array(dense_features)

        return [sparse_features, dense_features]

    @staticmethod
    def _compute_default_label_features(
        labels_example: List["Message"],
    ) -> List[np.ndarray]:
        """Compute one-hot representation for the labels"""

        return [
            np.array(
                [
                    np.expand_dims(a, 0)
                    for a in np.eye(len(labels_example), dtype=np.float32)
                ]
            )
        ]

    def _create_label_data(
        self,
        training_data: "TrainingData",
        label_id_dict: Dict[Text, int],
        attribute: Text,
    ) -> "SessionDataType":
        """Create matrix with label_ids encoded in rows as bag of words.

        Find a training example for each label and get the encoded features
        from the corresponding Message object.
        If the features are already computed, fetch them from the message object
        else compute a one hot encoding for the label as the feature vector.
        """

        # Collect one example for each label
        labels_idx_example = []
        for label_name, idx in label_id_dict.items():
            label_example = self._find_example_for_label(
                label_name, training_data.intent_examples, attribute
            )
            labels_idx_example.append((idx, label_example))

        # Sort the list of tuples based on label_idx
        labels_idx_example = sorted(labels_idx_example, key=lambda x: x[0])
        labels_example = [example for (_, example) in labels_idx_example]

        # Collect features, precomputed if they exist, else compute on the fly
        if self._check_labels_features_exist(labels_example, attribute):
            features = self._extract_labels_precomputed_features(
                labels_example, attribute
            )
        else:
            features = self._compute_default_label_features(labels_example)

        label_data = {}
        self._add_to_session_data(label_data, "label_features", features)
        self._add_mask_to_session_data(label_data, "label_mask", "label_features")

        return label_data

    def _use_default_label_features(self, label_ids: np.ndarray) -> List[np.ndarray]:
        return [
            np.array(
                [
                    self._label_data["label_features"][0][label_id]
                    for label_id in label_ids
                ]
            )
        ]

    # noinspection PyPep8Naming
    def _create_session_data(
        self,
        training_data: List["Message"],
        label_id_dict: Optional[Dict[Text, int]] = None,
        label_attribute: Optional[Text] = INTENT_ATTRIBUTE,
    ) -> "SessionDataType":
        """Prepare data for training and create a SessionDataType object"""

        X_sparse = []
        X_dense = []
        Y_sparse = []
        Y_dense = []
        label_ids = []

        for e in training_data:
            if e.get(label_attribute):
                _sparse, _dense = self._extract_and_add_features(e, TEXT_ATTRIBUTE)
                if _sparse is not None:
                    X_sparse.append(_sparse)
                if _dense is not None:
                    X_dense.append(_dense)

                _sparse, _dense = self._extract_and_add_features(e, label_attribute)
                if _sparse is not None:
                    Y_sparse.append(_sparse)
                if _dense is not None:
                    Y_dense.append(_dense)

                if label_id_dict:
                    label_ids.append(label_id_dict[e.get(label_attribute)])

        X_sparse = np.array(X_sparse)
        X_dense = np.array(X_dense)
        Y_sparse = np.array(Y_sparse)
        Y_dense = np.array(Y_dense)
        label_ids = np.array(label_ids)

        session_data = {}
        self._add_to_session_data(session_data, "text_features", [X_sparse, X_dense])
        self._add_to_session_data(session_data, "label_features", [Y_sparse, Y_dense])
        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        self._add_to_session_data(
            session_data, "label_ids", [np.expand_dims(label_ids, -1)]
        )

        if label_id_dict and (
            "label_features" not in session_data or not session_data["label_features"]
        ):
            # no label features are present, get default features from _label_data
            session_data["label_features"] = self._use_default_label_features(label_ids)

        self._add_mask_to_session_data(session_data, "text_mask", "text_features")
        self._add_mask_to_session_data(session_data, "label_mask", "label_features")

        return session_data

    @staticmethod
    def _add_to_session_data(
        session_data: SessionDataType, key: Text, features: List[np.ndarray]
    ):
        if not features:
            return

        session_data[key] = []

        for data in features:
            if data.size > 0:
                session_data[key].append(data)

    @staticmethod
    def _add_mask_to_session_data(
        session_data: SessionDataType, key: Text, from_key: Text
    ):

        session_data[key] = []

        for data in session_data[from_key]:
            if data.size > 0:
                # explicitly add last dimension to mask
                # to track correctly dynamic sequences
                mask = np.array([np.ones((x.shape[0], 1)) for x in data])
                session_data[key].append(mask)
                break

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

    def _combine_sparse_dense_features(
        self,
        features: List[Union[tf.Tensor, tf.SparseTensor]],
        mask: tf.Tensor,
        name: Text,
    ) -> tf.Tensor:
        dense_features = []

        dense_dim = self.dense_dim[name]
        # if dense features are present use the feature dimension of the dense features
        for f in features:
            if not isinstance(f, tf.SparseTensor):
                dense_dim = f.shape[-1]
                break

        for f in features:
            if isinstance(f, tf.SparseTensor):
                dense_features.append(
                    train_utils.tf_dense_layer_for_sparse(f, dense_dim, name, self.C2)
                )
            else:
                dense_features.append(f)

        output = tf.concat(dense_features, axis=-1) * mask
        output = tf.squeeze(output, axis=1)

        return output

    def _build_tf_train_graph(
        self, session_data: SessionDataType
    ) -> Tuple["tf.Tensor", "tf.Tensor"]:

        # get in tensors from generator
        self.batch_in = self._iterator.get_next()
        # convert encoded all labels into the batch format
        label_batch = train_utils.prepare_batch(self._label_data)

        # convert batch format into sparse and dense tensors
        batch_data, _ = train_utils.batch_to_session_data(self.batch_in, session_data)
        label_data, _ = train_utils.batch_to_session_data(label_batch, self._label_data)

        a = self._combine_sparse_dense_features(
            batch_data["text_features"], batch_data["text_mask"][0], "text"
        )
        b = self._combine_sparse_dense_features(
            batch_data["label_features"], batch_data["label_mask"][0], "label"
        )
        all_bs = self._combine_sparse_dense_features(
            label_data["label_features"], label_data["label_mask"][0], "label"
        )

        self.message_embed = self._create_tf_embed_fnn(
            a,
            self.hidden_layer_sizes["text"],
            fnn_name="text_label" if self.share_hidden_layers else "text",
            embed_name="text",
        )
        self.label_embed = self._create_tf_embed_fnn(
            b,
            self.hidden_layer_sizes["label"],
            fnn_name="text_label" if self.share_hidden_layers else "label",
            embed_name="label",
        )
        self.all_labels_embed = self._create_tf_embed_fnn(
            all_bs,
            self.hidden_layer_sizes["label"],
            fnn_name="text_label" if self.share_hidden_layers else "label",
            embed_name="label",
        )

        return train_utils.calculate_loss_acc(
            self.message_embed,
            self.label_embed,
            b,
            self.all_labels_embed,
            all_bs,
            self.num_neg,
            None,
            self.loss_type,
            self.mu_pos,
            self.mu_neg,
            self.use_max_sim_neg,
            self.C_emb,
            self.scale_loss,
        )

    def _build_tf_pred_graph(self, session_data: "SessionDataType") -> "tf.Tensor":

        shapes, types = train_utils.get_shapes_types(session_data)

        batch_placeholder = []
        for s, t in zip(shapes, types):
            batch_placeholder.append(tf.placeholder(t, s))

        self.batch_in = tf.tuple(batch_placeholder)

        batch_data, self.batch_tuple_sizes = train_utils.batch_to_session_data(
            self.batch_in, session_data
        )

        a = self._combine_sparse_dense_features(
            batch_data["text_features"], batch_data["text_mask"][0], "text"
        )
        b = self._combine_sparse_dense_features(
            batch_data["label_features"], batch_data["label_mask"][0], "label"
        )

        self.all_labels_embed = tf.constant(self.session.run(self.all_labels_embed))

        self.message_embed = self._create_tf_embed_fnn(
            a,
            self.hidden_layer_sizes["text"],
            fnn_name="text_label" if self.share_hidden_layers else "text",
            embed_name="text",
        )

        self.sim_all = train_utils.tf_raw_sim(
            self.message_embed[:, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, :, :],
            None,
        )

        self.label_embed = self._create_tf_embed_fnn(
            b,
            self.hidden_layer_sizes["label"],
            fnn_name="text_label" if self.share_hidden_layers else "label",
            embed_name="label",
        )

        self.sim = train_utils.tf_raw_sim(
            self.message_embed[:, tf.newaxis, :], self.label_embed, None
        )

        return train_utils.confidence_from_sim(self.sim_all, self.similarity_type)

    @staticmethod
    def _get_num_of_features(session_data: "SessionDataType", key: Text) -> int:
        num_features = 0
        for data in session_data[key]:
            if data.size > 0:
                num_features += data[0].shape[-1]
        return num_features

    def check_input_dimension_consistency(self, session_data: "SessionDataType"):
        """Check if text features and intent features have the same dimension."""

        if self.share_hidden_layers:
            num_text_features = self._get_num_of_features(session_data, "text_features")
            num_intent_features = self._get_num_of_features(
                session_data, "label_features"
            )

            if num_text_features != num_intent_features:
                raise ValueError(
                    "If embeddings are shared, "
                    "text features and label features "
                    "must coincide. Check the output dimensions of previous components."
                )

    def preprocess_train_data(self, training_data: "TrainingData"):
        """Prepares data for training.

        Performs sanity checks on training data, extracts encodings for labels.
        """

        label_id_dict = self._create_label_id_dict(
            training_data, attribute=INTENT_ATTRIBUTE
        )

        self.inverted_label_dict = {v: k for k, v in label_id_dict.items()}

        self._label_data = self._create_label_data(
            training_data, label_id_dict, attribute=INTENT_ATTRIBUTE
        )

        session_data = self._create_session_data(
            training_data.intent_examples,
            label_id_dict,
            label_attribute=INTENT_ATTRIBUTE,
        )

        self.check_input_dimension_consistency(session_data)

        return session_data

    @staticmethod
    def _check_enough_labels(session_data: "SessionDataType") -> bool:
        return len(np.unique(session_data["label_ids"])) >= 2

    def train(
        self,
        training_data: "TrainingData",
        cfg: Optional["RasaNLUModelConfig"] = None,
        **kwargs: Any,
    ) -> None:
        """Train the embedding intent classifier on a data set."""

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
                session_data,
                self.evaluate_on_num_examples,
                self.random_seed,
                label_key="label_ids",
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
                session_data,
                eval_session_data,
                batch_size_in,
                self.batch_in_strategy,
                label_key="label_ids",
            )

            self._is_training = tf.placeholder_with_default(False, shape=())

            loss, acc = self._build_tf_train_graph(session_data)

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
                self.batch_in_size,
                self.evaluate_on_num_examples,
                self.evaluate_every_num_epochs,
            )

            # rebuild the graph for prediction
            self.pred_confidence = self._build_tf_pred_graph(session_data)

    # process helpers
    # noinspection PyPep8Naming
    def _calculate_message_sim(
        self, batch: Tuple[np.ndarray]
    ) -> Tuple[np.ndarray, List[float]]:
        """Calculate message similarities"""

        message_sim = self.session.run(
            self.pred_confidence,
            feed_dict={
                _x_in: _x for _x_in, _x in zip(self.batch_in, batch) if _x is not None
            },
        )

        message_sim = message_sim.flatten()  # sim is a matrix

        label_ids = message_sim.argsort()[::-1]

        if self.loss_type == "softmax" and self.ranking_length > 0:
            message_sim = train_utils.normalize(message_sim, self.ranking_length)

        message_sim[::-1].sort()

        # transform sim to python list for JSON serializing
        return label_ids, message_sim.tolist()

    def predict_label(
        self, message: "Message"
    ) -> Tuple[Dict[Text, Any], List[Dict[Text, Any]]]:
        """Predicts the intent of the provided message."""

        label = {"name": None, "confidence": 0.0}
        label_ranking = []

        if self.session is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data."
            )
            return label, label_ranking

        # create session data from message and convert it into a batch of 1
        session_data = self._create_session_data([message])
        batch = train_utils.prepare_batch(
            session_data, tuple_sizes=self.batch_tuple_sizes
        )

        # load tf graph and session
        label_ids, message_sim = self._calculate_message_sim(batch)

        # if X contains all zeros do not predict some label
        if label_ids.size > 0:
            label = {
                "name": self.inverted_label_dict[label_ids[0]],
                "confidence": message_sim[0],
            }

            if self.ranking_length and 0 < self.ranking_length < LABEL_RANKING_LENGTH:
                output_length = self.ranking_length
            else:
                output_length = LABEL_RANKING_LENGTH

            ranking = list(zip(list(label_ids), message_sim))
            ranking = ranking[:output_length]
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
            train_utils.persist_tensor("batch_placeholder", self.batch_in, self.graph)

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

        with open(
            os.path.join(model_dir, file_name + ".batch_tuple_sizes.pkl"), "wb"
        ) as f:
            pickle.dump(self.batch_tuple_sizes, f)

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
        """Loads the trained model from the provided directory."""

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

                batch_in = train_utils.load_tensor("batch_placeholder")

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

            with open(
                os.path.join(model_dir, file_name + ".batch_tuple_sizes.pkl"), "rb"
            ) as f:
                batch_tuple_sizes = pickle.load(f)

            return cls(
                component_config=meta,
                inverted_label_dict=inv_label_dict,
                session=session,
                graph=graph,
                batch_placeholder=batch_in,
                similarity_all=sim_all,
                pred_confidence=pred_confidence,
                similarity=sim,
                message_embed=message_embed,
                label_embed=label_embed,
                all_labels_embed=all_labels_embed,
                batch_tuple_sizes=batch_tuple_sizes,
            )

        else:
            raise_warning(
                f"Failed to load nlu model. "
                f"Maybe the path '{os.path.abspath(model_dir)}' doesn't exist?",
            )
            return cls(component_config=meta)
