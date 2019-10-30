import logging
import numpy as np
import os
import pickle
import scipy.sparse
import typing
from typing import Any, Dict, List, Optional, Text, Tuple, Union
import warnings

from tf_metrics import f1

from nlu.extractors import EntityExtractor
from nlu.test import determine_token_labels
from nlu.tokenizers.tokenizer import Token
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.utils import train_utils
from rasa.utils.train_utils import SessionData
from rasa.nlu.constants import (
    MESSAGE_INTENT_ATTRIBUTE,
    MESSAGE_TEXT_ATTRIBUTE,
    MESSAGE_VECTOR_SPARSE_FEATURE_NAMES,
    MESSAGE_VECTOR_DENSE_FEATURE_NAMES,
    MESSAGE_TOKENS_NAMES,
    MESSAGE_ENTITIES_ATTRIBUTE,
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


class EmbeddingIntentClassifier(EntityExtractor):
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

    provides = ["intent", "intent_ranking", "entities"]

    requires = [MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE]]

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer for input words
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_a": [256, 128],
        # sizes of hidden layers before the embedding layer for intent labels
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_b": [],
        # sizes of hidden layers before the embedding layer for tag labels
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_c": [],
        # Whether to share the hidden layer weights between input words and labels
        "share_hidden_layers": False,
        # number of units in transformer
        "transformer_size": 128,
        # number of transformer layers
        "num_transformer_layers": 1,
        # number of attention heads in transformer
        "num_heads": 4,
        # type of positional encoding in transformer
        "pos_encoding": "timing",  # string 'timing' or 'emb'
        # max sequence length if pos_encoding='emb'
        "max_seq_length": 256,
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
        # use a unidirectional or bidirectional encoder
        "unidirectional_encoder": True,
        # visualization of accuracy
        # how often to calculate training accuracy
        "evaluate_every_num_epochs": 20,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        "evaluate_on_num_examples": 0,  # large values may hurt performance
        # model config
        # if true intent classification is trained and intent predicted
        "intent_classification": True,
        # if true named entity recognition is trained and entities predicted
        "named_entity_recognition": True,
    }
    # end default properties (DOC MARKER - don't remove)

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        inverted_label_dict: Optional[Dict[int, Text]] = None,
        inverted_tag_dict: Optional[Dict[int, Text]] = None,
        session: Optional["tf.Session"] = None,
        graph: Optional["tf.Graph"] = None,
        message_placeholder: Optional["tf.Tensor"] = None,
        label_placeholder: Optional["tf.Tensor"] = None,
        tag_placeholder: Optional["tf.Tensor"] = None,
        similarity_all: Optional["tf.Tensor"] = None,
        intent_prediction: Optional["tf.Tensor"] = None,
        tag_prediction: Optional["tf.Tensor"] = None,
        similarity: Optional["tf.Tensor"] = None,
        label_embed: Optional["tf.Tensor"] = None,
        all_labels_embed: Optional["tf.Tensor"] = None,
        attention_weights: Optional["tf.Tensor"] = None,
    ) -> None:
        """Declare instant variables with default values"""
        super(EmbeddingIntentClassifier, self).__init__(component_config)

        self._load_params()

        # transform numbers to labels
        self.inverted_label_dict = inverted_label_dict
        # transform numbers to tags
        self.inverted_tag_dict = inverted_tag_dict
        # encode all label_ids with numbers
        self._encoded_all_label_ids = None
        # encode all tag_ids with numbers
        self._encoded_all_tag_ids = None

        # tf related instances
        self.session = session
        self.graph = graph
        self.a_in = message_placeholder
        self.b_in = label_placeholder
        self.c_in = tag_placeholder
        self.sim_all = similarity_all
        self.intent_prediction = intent_prediction
        self.entity_prediction = tag_prediction
        self.sim = similarity
        self.attention_weights = attention_weights

        # persisted embeddings
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
                    "Intent tokenization has been moved to Tokenizer components. "
                    "Your config still mentions '{}'. Tokenization may fail if you specify the parameter here."
                    "Please specify the parameter 'intent_tokenization_flag' and 'intent_split_symbol' in the "
                    "tokenizer of your NLU pipeline".format(removed_param)
                )

    # init helpers
    def _load_nn_architecture_params(self, config: Dict[Text, Any]) -> None:
        self.hidden_layer_sizes = {
            "a": config["hidden_layers_sizes_a"],
            "b": config["hidden_layers_sizes_b"],
            "c": config["hidden_layers_sizes_c"],
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

        self.transformer_size = self.component_config["transformer_size"]
        self.num_transformer_layers = self.component_config["num_transformer_layers"]
        self.num_heads = self.component_config["num_heads"]
        self.pos_encoding = self.component_config["pos_encoding"]
        self.max_seq_length = self.component_config["max_seq_length"]
        self.unidirectional_encoder = self.component_config["unidirectional_encoder"]

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

        self.intent_classification = self.component_config["intent_classification"]
        self.named_entity_recognition = self.component_config[
            "named_entity_recognition"
        ]

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

        distinct_label_ids = set(
            [example.get(attribute) for example in training_data.intent_examples]
        ) - {None}
        return {
            label_id: idx for idx, label_id in enumerate(sorted(distinct_label_ids))
        }

    @staticmethod
    def _create_tag_id_dict(
        training_data: "TrainingData", attribute: Text
    ) -> Dict[Text, int]:
        """Create label_id dictionary"""

        distinct_tag_ids = set(
            [
                e["entity"]
                for example in training_data.entity_examples
                for e in example.get(attribute)
            ]
        ) - {None}
        tag_id_dict = {
            tag_id: idx for idx, tag_id in enumerate(sorted(distinct_tag_ids), 1)
        }
        tag_id_dict["O"] = 0
        return tag_id_dict

    @staticmethod
    def _find_example_for_label(
        label: Text, examples: List["Message"], attribute: Text
    ) -> Optional["Message"]:
        for ex in examples:
            if ex.get(attribute) == label:
                return ex
        return None

    @staticmethod
    def _find_example_for_tag(tag, examples, attribute):
        for ex in examples:
            for e in ex.get(attribute):
                if e["entity"] == tag:
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
        """Create matrix with label_ids encoded in rows as bag of words. If the
        features are already computed, fetch them from the message object else compute
        a one hot encoding for the label as the feature vector. Find a training example
        for each label and get the encoded features from the corresponding Message
        object."""

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
            features = self._compute_default_label_features(labels_example)
            encoded_id_labels = [scipy.sparse.csr_matrix(f) for f in features]
            encoded_id_labels = np.array(encoded_id_labels)

        return encoded_id_labels

    def _create_encoded_tag_ids(
        self,
        training_data: "TrainingData",
        tag_id_dict: Dict[Text, int],
        attribute: Text,
    ) -> np.ndarray:
        """Create matrix with tag_ids encoded in rows as bag of words. If the features
        are already computed, fetch them from the message object else compute a one
        hot encoding for the label as the feature vector.
        Find a training example for each tag and get the encoded features from the
        corresponding Message object."""

        tags_example = []

        # Collect one example for each label
        for tag_name, idx in tag_id_dict.items():
            tag_example = self._find_example_for_tag(
                tag_name, training_data.entity_examples, attribute
            )
            tags_example.append((idx, tag_example))

        # Collect features, precomputed if they exist, else compute on the fly
        features = self._compute_default_label_features(tags_example)
        encoded_id_tags = [scipy.sparse.csr_matrix(f) for f in features]
        encoded_id_tags = np.array(encoded_id_tags)

        return encoded_id_tags

    # noinspection PyPep8Naming
    def _create_session_data(
        self,
        training_data: "TrainingData",
        label_id_dict: Dict[Text, int],
        tag_id_dict: Dict[Text, int],
        attribute: Text,
    ) -> "SessionData":
        """Prepare data for training and create a SessionData object"""
        X_sparse = []
        X_dense = []
        Y = []
        intent_ids = []
        tag_ids = []

        for e in training_data.training_examples:
            if e.get(attribute):
                x_sparse, x_dense = self._get_x_features(e)

                if x_sparse is not None:
                    X_sparse.append(x_sparse)
                if x_dense is not None:
                    X_dense.append(x_dense)

                intent_ids.append(label_id_dict[e.get(attribute)])

        for e in training_data.training_examples:
            _tags = []
            for t in e.get("tokens"):
                _tag = determine_token_labels(
                    t, e.get(MESSAGE_ENTITIES_ATTRIBUTE), None
                )
                _tags.append(tag_id_dict[_tag])
            tag_ids.append(scipy.sparse.csr_matrix(np.array([_tags]).T))

        X_sparse = np.array(X_sparse)
        X_dense = np.array(X_dense)
        intent_ids = np.array(intent_ids)
        tag_ids = np.array(tag_ids)

        for label_id_idx in intent_ids:
            Y.append(self._encoded_all_label_ids[label_id_idx])
        Y = np.array(Y)

        X_dict = {}
        if X_sparse.size > 0:
            X_dict["text_features_sparse"] = X_sparse
        if X_dense.size > 0:
            X_dict["text_features_dense"] = X_dense

        return SessionData(
            X_dict,
            {"intent_features": Y},
            {"intent_ids": intent_ids, "tag_ids": tag_ids},
        )

    def _get_x_features(
        self, message: "Message"
    ) -> Tuple[
        Optional[Union[np.ndarray, scipy.sparse.spmatrix]],
        Optional[Union[np.ndarray, scipy.sparse.spmatrix]],
    ]:
        x_sparse = None
        x_dense = None

        if (
            message.get(MESSAGE_VECTOR_SPARSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE])
            is not None
        ):
            x_sparse = message.get(
                MESSAGE_VECTOR_SPARSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE]
            )

        if (
            message.get(MESSAGE_VECTOR_DENSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE])
            is not None
        ):
            x_dense = message.get(
                MESSAGE_VECTOR_DENSE_FEATURE_NAMES[MESSAGE_TEXT_ATTRIBUTE]
            )

        return x_sparse, x_dense

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

    def _build_tf_train_graph(
        self
    ) -> Tuple["tf.Tensor", "tf.Tensor", "tf.Tensor", "tf.Tensor"]:
        intent_loss = tf.constant(0.0)
        intent_metric = tf.constant(0.0)
        entity_loss = tf.constant(0.0)
        entity_metric = tf.constant(0.0)

        # batch = 1 or 2 a_in values, b_in, intent_ids
        batch = self._iterator.get_next()
        self.a_in, self.b_in, self.c_in = self.batch_to_input(batch)

        # transformer
        a, mask = self._create_tf_sequence(self.a_in)

        if self.intent_classification:
            intent_loss, intent_metric = self.build_intent_train_graph(a, mask)

        if self.named_entity_recognition:
            entity_loss, entity_metric = self.build_entity_train_graph(a, mask)

        return intent_loss, intent_metric, entity_loss, entity_metric

    def build_entity_train_graph(self, a, mask):
        # get sequence lengths for NER
        sequence_lengths = tf.cast(tf.reduce_sum(mask, 1), tf.int32)
        if len(sequence_lengths.shape) > 1:
            sequence_lengths = tf.squeeze(sequence_lengths)

        sequence_lengths.set_shape([mask.shape[0]])

        # shape: batch-size, seq-len, dim
        self.c_in = tf.reduce_sum(tf.nn.relu(self.c_in), -1)

        # CRF
        crf_params, logits, pred_ids = self._create_crf(a, sequence_lengths)

        # Loss
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, self.c_in, sequence_lengths, crf_params
        )
        loss = tf.reduce_mean(-log_likelihood)

        pos_tag_indices = [k for k, v in self.inverted_tag_dict.items() if v != "O"]

        # Metrics
        weights = tf.sequence_mask(sequence_lengths)
        num_tags = len(self.inverted_tag_dict)
        metric = f1(self.c_in, pred_ids, num_tags, pos_tag_indices, weights)[1]

        return loss, metric

    def build_intent_train_graph(
        self, a: tf.Tensor, mask: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        last = mask * tf.cumprod(1 - mask, axis=1, exclusive=True, reverse=True)
        last = tf.expand_dims(last, -1)

        # get _cls_ vector for intent classification
        cls_embed = tf.reduce_sum(a * last, 1)
        cls_embed = train_utils.create_tf_embed(
            cls_embed,
            self.embed_dim,
            self.C2,
            self.similarity_type,
            layer_name_suffix="a",
        )

        # all_label_ids is tensor of label-count x 1 x feature-len for labels
        all_label_ids = tf.sparse_tensor_to_dense(
            self._encoded_all_label_ids, name="all_labels_raw"
        )
        all_label_ids = tf.reduce_sum(tf.nn.relu(all_label_ids), 1)

        self.all_labels_embed = self._create_tf_embed_fnn(
            all_label_ids,
            self.hidden_layer_sizes["b"],
            fnn_name="a_b" if self.share_hidden_layers else "b",
            embed_name="b",
        )

        self.b_in = tf.reduce_sum(tf.nn.relu(self.b_in), 1)

        self.label_embed = self._create_tf_embed_fnn(
            self.b_in,
            self.hidden_layer_sizes["b"],
            fnn_name="a_b" if self.share_hidden_layers else "b",
            embed_name="b",
        )

        intent_loss, intent_metric = train_utils.calculate_loss_acc(
            cls_embed,
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
        return intent_loss, intent_metric

    def _create_crf(
        self, input: tf.Tensor, sequence_lengths: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        num_tags = len(self.inverted_tag_dict)

        with tf.variable_scope("model", reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(input, num_tags, name="crf-logits")
            crf_params = tf.get_variable(
                "crf-params", [num_tags, num_tags], dtype=tf.float32
            )
            pred_ids, _ = tf.contrib.crf.crf_decode(
                logits, crf_params, sequence_lengths
            )

            return crf_params, logits, pred_ids

    def _create_tf_sequence(self, a_in: tf.Tensor) -> Tuple["tf.Tensor", "tf.Tensor"]:
        """Create sequence level embedding and mask."""
        # mask different length sequences
        # if there is at least one `-1` it should be masked
        mask = tf.sign(tf.reduce_max(self.a_in, -1) + 1)

        a_in = train_utils.create_tf_fnn(
            a_in,
            self.hidden_layer_sizes["a"],
            self.droprate,
            self.C2,
            self._is_training,
            layer_name_suffix="a",
        )
        self.attention_weights = {}
        hparams = train_utils.create_t2t_hparams(
            self.num_transformer_layers,
            self.transformer_size,
            self.num_heads,
            self.droprate,
            self.pos_encoding,
            self.max_seq_length,
            self._is_training,
            self.unidirectional_encoder,
        )

        a = train_utils.create_t2t_transformer_encoder(
            a_in, mask, self.attention_weights, hparams, self.C2, self._is_training
        )

        return a, mask

    def batch_to_input(self, batch: Tuple) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        """Convert batch input into correct tensors.

        As we do not know what features (sparse and/or dense) were used, we need to
        check what features are provided and parse them accordingly.
        """
        # batch contains 1 or 2 a_in values, b_in, label_ids, tag_ids
        b_in = batch[-3]
        c_in = batch[-1]

        if len(batch) == 4:
            a_in = batch[0]
            return a_in, b_in, c_in

        if len(batch) == 5:
            a_in_1 = batch[0]
            a_in_2 = batch[1]
            # Concatenate a_in features
            # TODO should not be just concatenated
            a_in = tf.concat([a_in_1, a_in_2], axis=1)

            return a_in, b_in, c_in

        raise ValueError("Iterator return unexpected number of tensors.")

    def _build_tf_pred_graph(self, session_data: "SessionData") -> "tf.Tensor":
        num_features_sparse = self._get_num_of_features(
            session_data, "text_features_sparse"
        )
        num_features_dense = self._get_num_of_features(
            session_data, "text_features_dense"
        )

        self.a_in = tf.placeholder(
            tf.float32, (None, num_features_sparse + num_features_dense), name="a"
        )
        self.b_in = tf.placeholder(
            tf.float32,
            (None, None, session_data.Y["intent_features"][0].shape[-1]),
            name="b",
        )
        self.c_in = tf.placeholder(
            tf.int64,
            (None, None, session_data.labels["tag_ids"][0].shape[-1]),
            name="c",
        )

        a, mask = self._create_tf_sequence(self.a_in)

        if self.intent_classification:
            last = mask * tf.cumprod(1 - mask, axis=1, exclusive=True, reverse=True)
            last = tf.expand_dims(last, -1)

            # get _cls_ embedding
            cls_embed = tf.reduce_sum(a * last, 1)
            cls_embed = train_utils.create_tf_embed(
                cls_embed,
                self.embed_dim,
                self.C2,
                self.similarity_type,
                layer_name_suffix="a",
            )

            # reduce dimensionality as input should not be sequence for intent
            # classification
            self.b_in = tf.reduce_sum(self.b_in, 1)

            self.sim_all = train_utils.tf_raw_sim(
                cls_embed[:, tf.newaxis, :],
                self.all_labels_embed[tf.newaxis, :, :],
                None,
            )

            self.label_embed = self._create_tf_embed_fnn(
                self.b_in,
                self.hidden_layer_sizes["b"],
                fnn_name="a_b" if self.share_hidden_layers else "b",
                embed_name="b",
            )

            # predict intents
            self.sim = train_utils.tf_raw_sim(
                cls_embed[:, tf.newaxis, :], self.label_embed, None
            )
            self.intent_prediction = train_utils.confidence_from_sim(
                self.sim_all, self.similarity_type
            )

        if self.named_entity_recognition:
            # get sequence lengths for NER
            sequence_lengths = tf.cast(tf.reduce_sum(mask, 1), tf.int32)

            # predict tags
            _, _, pred_ids = self._create_crf(a, sequence_lengths)
            self.entity_prediction = tf.to_int64(pred_ids)

    def _get_num_of_features(self, session_data: "SessionData", x_key: Text) -> int:
        return session_data.X[x_key][0].shape[-1] if x_key in session_data.X else 0

    def check_input_dimension_consistency(self, session_data: "SessionData"):
        if self.share_hidden_layers:
            num_features_sparse = self._get_num_of_features(
                session_data, "text_features_sparse"
            )
            num_features_dense = self._get_num_of_features(
                session_data, "text_features_dense"
            )

            if (
                num_features_sparse + num_features_dense
                != session_data.Y["intent_features"][0].shape[-1]
            ):
                raise ValueError(
                    "If embeddings are shared "
                    "text features and label features "
                    "must coincide. Check the output dimensions of previous components."
                )

    def preprocess_train_data(self, training_data: "TrainingData"):
        """Performs sanity checks on training data, extracts encodings for labels and
        prepares data for training"""

        label_id_dict = self._create_label_id_dict(
            training_data, attribute=MESSAGE_INTENT_ATTRIBUTE
        )
        self.inverted_label_dict = {v: k for k, v in label_id_dict.items()}

        self._encoded_all_label_ids = self._create_encoded_label_ids(
            training_data,
            label_id_dict,
            attribute=MESSAGE_INTENT_ATTRIBUTE,
            attribute_feature_name=MESSAGE_VECTOR_SPARSE_FEATURE_NAMES[
                MESSAGE_INTENT_ATTRIBUTE
            ],
        )

        tag_id_dict = self._create_tag_id_dict(
            training_data, attribute=MESSAGE_ENTITIES_ATTRIBUTE
        )
        self.inverted_tag_dict = {v: k for k, v in tag_id_dict.items()}

        self._encoded_all_tag_ids = self._create_encoded_tag_ids(
            training_data, tag_id_dict, attribute=MESSAGE_ENTITIES_ATTRIBUTE
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
            training_data,
            label_id_dict,
            tag_id_dict,
            attribute=MESSAGE_INTENT_ATTRIBUTE,
        )

        self.check_input_dimension_consistency(session_data)

        return session_data

    def _check_enough_labels(self, session_data: "SessionData") -> bool:
        return len(np.unique(session_data.labels["intent_ids"])) >= 2

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

        if self.intent_classification:
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
                label_key="intent_ids",
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
                self.batch_strategy,
                label_key="intent_ids",
            )

            self._is_training = tf.placeholder_with_default(False, shape=())

            intent_loss, intent_acc, entity_loss, entity_f1 = (
                self._build_tf_train_graph()
            )

            loss = intent_loss + entity_loss

            # define which optimizer to use
            self._train_op = tf.train.AdamOptimizer().minimize(loss)

            # train tensorflow graph
            self.session = tf.Session(config=self._tf_config)
            # TODO proper loss, acc handling
            train_utils.train_tf_dataset(
                train_init_op,
                eval_init_op,
                batch_size_in,
                loss,
                intent_acc,
                self._train_op,
                self.session,
                self._is_training,
                self.epochs,
                self.batch_size,
                self.evaluate_on_num_examples,
                self.evaluate_every_num_epochs,
            )

            # rebuild the graph for prediction
            self._build_tf_pred_graph(session_data)

            self.attention_weights = train_utils.extract_attention(
                self.attention_weights
            )

    # process helpers
    # noinspection PyPep8Naming
    def _calculate_message_sim(self, X: np.ndarray) -> Tuple[np.ndarray, List[float]]:
        """Calculate message similarities"""

        message_sim = self.session.run(self.intent_prediction, feed_dict={self.a_in: X})

        message_sim = message_sim.flatten()  # sim is a matrix

        label_ids = message_sim.argsort()[::-1]
        message_sim[::-1].sort()

        # transform sim to python list for JSON serializing
        return label_ids, message_sim.tolist()

    def predict_label(
        self, message: "Message"
    ) -> Tuple[Dict[Text, Any], List[Dict[Text, Any]]]:

        label = {"name": None, "confidence": 0.0}
        label_ranking = []
        if self.session is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data"
            )

        else:
            # get features (bag of words/embeddings) for a message
            # noinspection PyPep8Naming
            X = self._extract_features(message)

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

    def _extract_features(self, message: "Message") -> np.ndarray:
        x_sparse, x_dense = self._get_x_features(message)

        if x_sparse is not None:
            x_sparse = x_sparse.toarray().reshape(1, -1)

        if x_dense is not None:
            x_dense = x_dense.reshape(1, -1)

        if x_sparse is not None and x_dense is not None:
            return np.concatenate((x_sparse, x_dense), axis=-1)

        if x_sparse is None and x_dense is not None:
            return x_dense

        if x_sparse is not None and x_dense is None:
            return x_sparse

        raise ValueError("No features found for X.")

    def predict_entities(self, message: "Message") -> List[Dict]:
        if self.session is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
        else:
            # get features (bag of words) for a message
            # noinspection PyPep8Naming
            X = self._extract_features(message)

            # load tf graph and session
            predictions = self.session.run(
                self.entity_prediction, feed_dict={self.a_in: X}
            )

            tags = [self.inverted_tag_dict[p] for p in predictions[0]]

            entities = self._convert_tags_to_entities(
                message.text, message.get("tokens", []), tags
            )

            extracted = self.add_extractor_name(entities)
            entities = message.get("entities", []) + extracted

            return entities

    def _convert_tags_to_entities(
        self, text: str, tokens: List[Token], tags: List[Text]
    ) -> List[Dict[Text, Any]]:
        entities = []
        last_tag = "O"
        for token, tag in zip(tokens, tags):
            if tag == "O":
                last_tag = tag
                continue

            # new tag found
            if last_tag != tag:
                entity = {
                    "entity": tag,
                    "start": token.offset,
                    "end": token.end,
                    "extractor": "flair",
                }
                entities.append(entity)

            # belongs to last entity
            elif last_tag == tag:
                entities[-1]["end"] = token.end

            last_tag = tag

        for entity in entities:
            entity["value"] = text[entity["start"] : entity["end"]]

        return entities

    def process(self, message: "Message", **kwargs: Any) -> None:
        """Return the most likely label and its similarity to the input."""

        if self.intent_classification:
            label, label_ranking = self.predict_label(message)

            message.set("intent", label, add_to_output=True)
            message.set("intent_ranking", label_ranking, add_to_output=True)

        if self.named_entity_recognition:
            entities = self.predict_entities(message)

            message.set("entities", entities, add_to_output=True)

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
            train_utils.persist_tensor("tag_placeholder", self.c_in, self.graph)

            train_utils.persist_tensor("similarity_all", self.sim_all, self.graph)
            train_utils.persist_tensor(
                "intent_prediction", self.intent_prediction, self.graph
            )
            train_utils.persist_tensor(
                "entity_prediction", self.entity_prediction, self.graph
            )
            train_utils.persist_tensor("similarity", self.sim, self.graph)

            train_utils.persist_tensor("label_embed", self.label_embed, self.graph)
            train_utils.persist_tensor(
                "all_labels_embed", self.all_labels_embed, self.graph
            )

            train_utils.persist_tensor(
                "attention_weights", self.attention_weights, self.graph
            )

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with open(
            os.path.join(model_dir, file_name + ".inv_label_dict.pkl"), "wb"
        ) as f:
            pickle.dump(self.inverted_label_dict, f)

        with open(os.path.join(model_dir, file_name + ".inv_tag_dict.pkl"), "wb") as f:
            pickle.dump(self.inverted_tag_dict, f)

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
                c_in = train_utils.load_tensor("tag_placeholder")

                sim_all = train_utils.load_tensor("similarity_all")
                intent_prediction = train_utils.load_tensor("intent_prediction")
                tag_prediction = train_utils.load_tensor("tag_prediction")
                sim = train_utils.load_tensor("similarity")

                label_embed = train_utils.load_tensor("label_embed")
                all_labels_embed = train_utils.load_tensor("all_labels_embed")

                attention_weights = train_utils.load_tensor("attention_weights")

            with open(
                os.path.join(model_dir, file_name + ".inv_label_dict.pkl"), "rb"
            ) as f:
                inv_label_dict = pickle.load(f)

            with open(
                os.path.join(model_dir, file_name + ".inv_tag_dict.pkl"), "rb"
            ) as f:
                inv_tag_dict = pickle.load(f)

            return cls(
                component_config=meta,
                inverted_label_dict=inv_label_dict,
                inverted_tag_dict=inv_tag_dict,
                session=session,
                graph=graph,
                message_placeholder=a_in,
                label_placeholder=b_in,
                tag_placeholder=c_in,
                similarity_all=sim_all,
                intent_prediction=intent_prediction,
                tag_prediction=tag_prediction,
                similarity=sim,
                label_embed=label_embed,
                all_labels_embed=all_labels_embed,
                attention_weights=attention_weights,
            )

        else:
            logger.warning(
                "Failed to load nlu model. Maybe path {} "
                "doesn't exist"
                "".format(os.path.abspath(model_dir))
            )
            return cls(component_config=meta)
