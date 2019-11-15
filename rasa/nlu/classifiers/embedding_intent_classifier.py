import logging
from collections import defaultdict

import numpy as np
import os
import pickle
import scipy.sparse
import typing
from typing import Any, Dict, List, Optional, Text, Tuple, Union
import warnings

from tf_metrics import f1

from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.test import determine_token_labels
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.utils import train_utils
from rasa.utils.train_utils import SessionDataType, TrainingMetrics
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

    requires = []

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
        # default dense dimension used if no dense features are present
        "dense_dim": 512,
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
        # number of entity tags
        "num_tags": None,
        "var_layers": []  # ["pre", "cls", "crf"]
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
                warnings.warn(
                    f"Intent tokenization has been moved to Tokenizer components. "
                    f"Your config still mentions '{removed_param}'. Tokenization may "
                    f"fail if you specify the parameter here. Please specify the "
                    f"parameter 'intent_tokenization_flag' and 'intent_split_symbol' "
                    f"in the tokenizer of your NLU pipeline."
                )

    # init helpers
    def _load_nn_architecture_params(self, config: Dict[Text, Any]) -> None:
        self.hidden_layer_sizes = {
            "text": config["hidden_layers_sizes_a"],
            "intent": config["hidden_layers_sizes_b"],
            "tag": config["hidden_layers_sizes_c"],
        }
        self.share_hidden_layers = config["share_hidden_layers"]
        if (
            self.share_hidden_layers
            and self.hidden_layer_sizes["text"] != self.hidden_layer_sizes["intent"]
        ):
            raise ValueError(
                "If hidden layer weights are shared,"
                "hidden_layer_sizes for a and b must coincide"
            )

        self.batch_in_size = config["batch_size"]
        self.batch_in_strategy = config["batch_strategy"]

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
        self.dense_dim = config["dense_dim"]

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
        self.num_tags = self.component_config["num_tags"]
        self.var_layers = self.component_config["var_layers"]

    # package safety checks
    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow"]

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        inverted_label_dict: Optional[Dict[int, Text]] = None,
        inverted_tag_dict: Optional[Dict[int, Text]] = None,
        session: Optional["tf.Session"] = None,
        graph: Optional["tf.Graph"] = None,
        batch_placeholder: Optional["tf.Tensor"] = None,
        similarity_all: Optional["tf.Tensor"] = None,
        intent_prediction: Optional["tf.Tensor"] = None,
        entity_prediction: Optional["tf.Tensor"] = None,
        similarity: Optional["tf.Tensor"] = None,
        cls_embed: Optional["tf.Tensor"] = None,
        label_embed: Optional["tf.Tensor"] = None,
        all_labels_embed: Optional["tf.Tensor"] = None,
        batch_tuple_sizes: Optional[Dict] = None,
        attention_weights: Optional["tf.Tensor"] = None,
    ) -> None:
        """Declare instant variables with default values"""

        super().__init__(component_config)

        self._load_params()

        # transform numbers to labels
        self.inverted_label_dict = inverted_label_dict
        self.inverted_tag_dict = inverted_tag_dict
        # encode all label_ids with numbers
        self._label_data = None

        # tf related instances
        self.session = session
        self.graph = graph
        self.batch_in = batch_placeholder
        self.sim_all = similarity_all
        self.intent_prediction = intent_prediction
        self.entity_prediction = entity_prediction
        self.sim = similarity

        # persisted embeddings
        self.cls_embed = cls_embed
        self.label_embed = label_embed
        self.all_labels_embed = all_labels_embed

        # keep the input tuple sizes in self.batch_in
        self.batch_tuple_sizes = batch_tuple_sizes

        # internal tf instances
        self._iterator = None
        self._train_op = None
        self._is_training = None

        self.pre_embed_layer = None
        self.cls_embed_layer = None
        self.crf_embed_layer = None

        self.attention_weights = attention_weights

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
        labels_example: List["Message"], attribute: Text
    ) -> bool:
        """Check if all labels have features set"""

        for label_example in labels_example:
            if (
                label_example.get(MESSAGE_VECTOR_SPARSE_FEATURE_NAMES[attribute])
                is None
                and label_example.get(MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute])
                is None
            ):
                return False
        return True

    @staticmethod
    def _extract_and_add_features(
        message: "Message", attribute: Text
    ) -> Tuple[Optional[scipy.sparse.spmatrix], Optional[np.ndarray]]:
        sparse_features = None
        dense_features = None

        if message.get(MESSAGE_VECTOR_SPARSE_FEATURE_NAMES[attribute]) is not None:
            sparse_features = message.get(
                MESSAGE_VECTOR_SPARSE_FEATURE_NAMES[attribute]
            )

        if message.get(MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute]) is not None:
            dense_features = message.get(MESSAGE_VECTOR_DENSE_FEATURE_NAMES[attribute])

        if sparse_features is not None and dense_features is not None:
            if sparse_features.shape[0] != dense_features.shape[0]:
                raise ValueError(
                    f"Sequence dimensions for sparse and dense features "
                    f"don't coincide in '{message.text}'"
                )

        return sparse_features, dense_features

    @staticmethod
    def _compute_default_label_features(
        labels_example: List["Message"],
    ) -> List[np.ndarray]:
        """Compute one-hot representation for the labels"""

        return [
            np.array(
                [
                    scipy.sparse.coo_matrix(
                        ([1], ([0], [idx])), shape=(1, len(labels_example))
                    )
                    for idx in range(len(labels_example))
                ]
            )
        ]

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

    @staticmethod
    def _get_num_of_features(session_data: "SessionDataType", key: Text) -> int:
        num_features = 0
        for data in session_data[key]:
            if data.size > 0:
                num_features += data[0].shape[-1]
        return num_features

    @staticmethod
    def _check_enough_labels(session_data: "SessionDataType") -> bool:
        return len(np.unique(session_data["intent_ids"])) >= 2

    def check_input_dimension_consistency(self, session_data: "SessionDataType"):
        if self.share_hidden_layers:
            num_text_features = self._get_num_of_features(session_data, "text_features")
            num_intent_features = self._get_num_of_features(
                session_data, "intent_features"
            )

            if num_text_features != num_intent_features:
                raise ValueError(
                    "If embeddings are shared "
                    "text features and label features "
                    "must coincide. Check the output dimensions of previous components."
                )

    def _extract_labels_precomputed_features(
        self, label_examples: List["Message"]
    ) -> List[np.ndarray]:
        """Collect precomputed encodings"""

        sparse_features = []
        dense_features = []

        for e in label_examples:
            _sparse, _dense = self._extract_and_add_features(
                e, MESSAGE_INTENT_ATTRIBUTE
            )
            if _sparse is not None:
                sparse_features.append(_sparse)
            if _dense is not None:
                dense_features.append(_dense)

        sparse_features = np.array(sparse_features)
        dense_features = np.array(dense_features)

        return [sparse_features, dense_features]

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
            features = self._extract_labels_precomputed_features(labels_example)
        else:
            features = self._compute_default_label_features(labels_example)

        label_data = {}
        self._add_to_session_data(label_data, "intent_features", features)
        self._add_mask_to_session_data(label_data, "intent_mask", "intent_features")

        return label_data

    def use_default_label_features(self, label_ids: np.ndarray) -> List[np.ndarray]:
        return [
            np.array(
                [
                    self._label_data["intent_features"][0][label_id]
                    for label_id in label_ids
                ]
            )
        ]

    def _create_session_data(
        self,
        training_data: List["Message"],
        label_id_dict: Optional[Dict[Text, int]] = None,
        tag_id_dict: Optional[Dict[Text, int]] = None,
        label_attribute: Optional[Text] = None,
    ) -> "SessionDataType":
        """Prepare data for training and create a SessionDataType object"""

        X_sparse = []
        X_dense = []
        Y_sparse = []
        Y_dense = []
        label_ids = []
        tag_ids = []

        for e in training_data:
            _sparse, _dense = self._extract_and_add_features(e, MESSAGE_TEXT_ATTRIBUTE)
            if _sparse is not None:
                X_sparse.append(_sparse)
            if _dense is not None:
                X_dense.append(_dense)

            _sparse, _dense = self._extract_and_add_features(
                e, MESSAGE_INTENT_ATTRIBUTE
            )
            if _sparse is not None:
                Y_sparse.append(_sparse)
            if _dense is not None:
                Y_dense.append(_dense)

            if label_attribute and e.get(label_attribute):
                label_ids.append(label_id_dict[e.get(label_attribute)])

            if self.named_entity_recognition and tag_id_dict:
                _tags = []
                for t in e.get(MESSAGE_TOKENS_NAMES[MESSAGE_TEXT_ATTRIBUTE]):
                    _tag = determine_token_labels(
                        t, e.get(MESSAGE_ENTITIES_ATTRIBUTE), None
                    )
                    _tags.append(tag_id_dict[_tag])
                # transpose to have seq_len x 1
                tag_ids.append(np.array([_tags]).T)

        X_sparse = np.array(X_sparse)
        X_dense = np.array(X_dense)
        Y_sparse = np.array(Y_sparse)
        Y_dense = np.array(Y_dense)
        label_ids = np.array(label_ids)
        tag_ids = np.array(tag_ids)

        session_data = {}
        self._add_to_session_data(session_data, "text_features", [X_sparse, X_dense])
        self._add_to_session_data(session_data, "intent_features", [Y_sparse, Y_dense])
        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        self._add_to_session_data(
            session_data, "intent_ids", [np.expand_dims(label_ids, -1)]
        )
        self._add_to_session_data(session_data, "tag_ids", [tag_ids])
        self._add_mask_to_session_data(session_data, "text_mask", "text_features")

        if label_attribute and (
            "intent_features" not in session_data or not session_data["intent_features"]
        ):
            # no label features are present, get default features from _label_data
            session_data["intent_features"] = self.use_default_label_features(label_ids)

        return session_data

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
            x, self.embed_dim, self.C2, embed_name, self.similarity_type
        )

    def combine_sparse_dense_features(
        self, features: List[Union["tf.Tensor", "tf.SparseTensor"]], name: Text
    ) -> "tf.Tensor":

        dense_features = []

        dense_dim = self.dense_dim
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

        return tf.concat(dense_features, axis=-1)

    def _create_tf_sequence(self, a_in, mask) -> "tf.Tensor":
        """Create sequence level embedding and mask."""

        a_in = train_utils.create_tf_fnn(
            a_in,
            self.hidden_layer_sizes["text"],
            self.droprate,
            self.C2,
            self._is_training,
            layer_name_suffix="text_intent" if self.share_hidden_layers else "text",
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

        if "pre" in self.var_layers and self.pre_embed_layer is None:
            self.pre_embed_layer = train_utils.create_tfp_embed_layer(self.transformer_size, layer_name_suffix="pre", use_bias=False)

        a = train_utils.create_t2t_transformer_encoder(
            a_in, self.pre_embed_layer, mask, self.attention_weights, hparams, self.C2, self._is_training
        )

        return a

    # build tf graphs:
    def _build_tf_train_graph(self, session_data: SessionDataType) -> TrainingMetrics:

        # get in tensors from generator
        self.batch_in = self._iterator.get_next()
        # convert encoded all labels into the batch format
        label_batch = train_utils.prepare_batch(self._label_data)

        # convert batch format into sparse and dense tensors
        batch_data, _ = train_utils.batch_to_session_data(self.batch_in, session_data)
        label_data, _ = train_utils.batch_to_session_data(label_batch, self._label_data)

        a = self.combine_sparse_dense_features(batch_data["text_features"], "text")
        mask = batch_data["text_mask"][0]

        # transformer
        a = self._create_tf_sequence(a, mask)

        metrics = TrainingMetrics(loss={}, score={})

        if self.intent_classification:
            b = self.combine_sparse_dense_features(
                batch_data["intent_features"], "intent"
            )
            all_bs = self.combine_sparse_dense_features(
                label_data["intent_features"], "intent"
            )

            loss, acc = self._train_intent_graph(a, b, all_bs, mask)
            metrics.loss["i_loss"] = loss
            metrics.score["i_acc"] = acc

        if self.named_entity_recognition:
            c = self.combine_sparse_dense_features(batch_data["tag_ids"], "tag")

            loss, f1_score = self._train_entity_graph(a, c, mask)
            metrics.loss["e_loss"] = loss
            metrics.score["e_f1"] = f1_score

        return metrics

    def _train_entity_graph(
        self, a: "tf.Tensor", c: "tf.Tensor", mask: "tf.Tensor"
    ) -> Tuple["tf.Tensor", "tf.Tensor"]:
        sequence_lengths = tf.cast(tf.reduce_sum(mask[:, :, 0], 1), tf.int32)
        sequence_lengths.set_shape([mask.shape[0]])

        c = tf.reduce_sum(tf.nn.relu(c), -1)
        c = tf.cast(c, tf.int32)

        # tensor shapes
        # a: tensor(batch-size, max-seq-len, dim)
        # sequence_lengths: tensor(batch-size)
        # c: (batch-size, max-seq-len)

        # CRF
        crf_params, logits, pred_ids = self._create_crf(a, sequence_lengths)

        # Loss
        log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
            logits, c, sequence_lengths, crf_params
        )
        loss = tf.reduce_mean(-log_likelihood)

        # calculate f1 score for train predictions
        weights = tf.sequence_mask(sequence_lengths)
        pos_tag_indices = [k for k, v in self.inverted_tag_dict.items() if v != "O"]
        score = f1(c, pred_ids, self.num_tags, pos_tag_indices, weights)

        return loss, score[1]

    def _train_intent_graph(
        self, a: "tf.Tensor", b: "tf.Tensor", all_bs: "tf.Tensor", mask: "tf.Tensor"
    ) -> Tuple["tf.Tensor", "tf.Tensor"]:
        last = mask * tf.cumprod(1 - mask, axis=1, exclusive=True, reverse=True)

        # get _cls_ vector for intent classification
        a_last = tf.reduce_sum(a * last, 1)

        if "cls" in self.var_layers:
            if self.cls_embed_layer is None:
                self.cls_embed_layer = train_utils.create_tfp_embed_layer(self.embed_dim, layer_name_suffix="cls")
            self.cls_embed = self.cls_embed_layer(a_last)
        else:
            self.cls_embed = train_utils.create_tf_embed(
                a_last, self.embed_dim, self.C2, "cls", self.similarity_type
            )

        b = tf.reduce_sum(tf.nn.relu(b), 1)
        all_bs = tf.reduce_sum(tf.nn.relu(all_bs), 1)

        self.label_embed = self._create_tf_embed_fnn(
            b,
            self.hidden_layer_sizes["intent"],
            fnn_name="text_intent" if self.share_hidden_layers else "intent",
            embed_name="intent",
        )
        self.all_labels_embed = self._create_tf_embed_fnn(
            all_bs,
            self.hidden_layer_sizes["intent"],
            fnn_name="text_intent" if self.share_hidden_layers else "intent",
            embed_name="intent",
        )

        return train_utils.calculate_loss_acc(
            self.cls_embed,
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

    def _create_crf(
        self, input: tf.Tensor, sequence_lengths: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.variable_scope("ner", reuse=tf.AUTO_REUSE):

            if "crf-logits" in self.var_layers:
                if self.crf_embed_layer is None:
                    self.crf_embed_layer = train_utils.create_tfp_embed_layer(self.num_tags, layer_name_suffix="crf-logits")
                logits = self.cls_embed_layer(input)
            else:
                logits = train_utils.create_tf_embed(
                    input, self.num_tags, self.C2, "crf-logits"
                )

            crf_params = tf.get_variable(
                "crf-params",
                [self.num_tags, self.num_tags],
                dtype=tf.float32,
                regularizer=tf.contrib.layers.l2_regularizer(self.C2),
            )
            pred_ids, _ = tf.contrib.crf.crf_decode(
                logits, crf_params, sequence_lengths
            )

            return crf_params, logits, pred_ids

    def _build_tf_pred_graph(self, session_data: "SessionDataType"):

        shapes, types = train_utils.get_shapes_types(session_data)

        batch_placeholder = []
        for s, t in zip(shapes, types):
            batch_placeholder.append(tf.placeholder(t, s))

        self.batch_in = tf.tuple(batch_placeholder)

        batch_data, self.batch_tuple_sizes = train_utils.batch_to_session_data(
            self.batch_in, session_data
        )

        a = self.combine_sparse_dense_features(batch_data["text_features"], "text")
        mask = batch_data["text_mask"][0]

        # transformer
        a = self._create_tf_sequence(a, mask)

        if self.intent_classification:
            b = self.combine_sparse_dense_features(
                batch_data["intent_features"], "intent"
            )
            self.all_labels_embed = tf.constant(self.session.run(self.all_labels_embed))

            self._pred_intent_graph(a, b, mask)

        if self.named_entity_recognition:
            self._pred_entity_graph(a, mask)

    def _pred_intent_graph(self, a: "tf.Tensor", b: "tf.Tensor", mask: "tf.Tensor"):
        last = mask * tf.cumprod(1 - mask, axis=1, exclusive=True, reverse=True)

        # get _cls_ embedding
        a_last = tf.reduce_sum(a * last, 1)

        if "cls" in self.var_layers:
            if self.cls_embed_layer is None:
                self.cls_embed_layer = train_utils.create_tfp_embed_layer(self.embed_dim, layer_name_suffix="cls")
            self.cls_embed = self.cls_embed_layer(a_last)
        else:
            self.cls_embed = train_utils.create_tf_embed(
                a_last, self.embed_dim, self.C2, "cls", self.similarity_type
            )

        b = tf.reduce_sum(b, 1)

        self.sim_all = train_utils.tf_raw_sim(
            self.cls_embed[:, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, :, :],
            None,
        )
        self.label_embed = self._create_tf_embed_fnn(
            b,
            self.hidden_layer_sizes["intent"],
            fnn_name="text_intent" if self.share_hidden_layers else "intent",
            embed_name="intent",
        )
        self.sim = train_utils.tf_raw_sim(
            self.cls_embed[:, tf.newaxis, :], self.label_embed, None
        )

        self.intent_prediction = train_utils.confidence_from_sim(
            self.sim_all, self.similarity_type
        )

    def _pred_entity_graph(self, a: "tf.Tensor", mask: "tf.Tensor"):
        sequence_lengths = tf.cast(tf.reduce_sum(mask[:, :, 0], 1), tf.int32)

        # predict tagsx
        _, _, pred_ids = self._create_crf(a, sequence_lengths)
        self.entity_prediction = tf.to_int64(pred_ids)

    # train helpers
    def preprocess_train_data(self, training_data: "TrainingData"):
        """Prepares data for training.

        Performs sanity checks on training data, extracts encodings for labels.
        """

        label_id_dict = self._create_label_id_dict(
            training_data, attribute=MESSAGE_INTENT_ATTRIBUTE
        )
        self.inverted_label_dict = {v: k for k, v in label_id_dict.items()}

        self._label_data = self._create_label_data(
            training_data, label_id_dict, attribute=MESSAGE_INTENT_ATTRIBUTE
        )

        tag_id_dict = self._create_tag_id_dict(
            training_data, attribute=MESSAGE_ENTITIES_ATTRIBUTE
        )
        self.inverted_tag_dict = {v: k for k, v in tag_id_dict.items()}

        session_data = self._create_session_data(
            training_data.training_examples,
            label_id_dict,
            tag_id_dict,
            label_attribute=MESSAGE_INTENT_ATTRIBUTE,
        )

        self.num_tags = len(self.inverted_tag_dict)

        self.check_input_dimension_consistency(session_data)

        return session_data

    # process helpers
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

            ranking = list(zip(list(label_ids), message_sim))
            ranking = ranking[:LABEL_RANKING_LENGTH]
            label_ranking = [
                {"name": self.inverted_label_dict[label_idx], "confidence": score}
                for label_idx, score in ranking
            ]

        return label, label_ranking

    def _calculate_message_sim(
        self, batch: Tuple[np.ndarray]
    ) -> Tuple[np.ndarray, List[float]]:
        """Calculate message similarities"""

        if "pre" in self.var_layers or "cls" in self.var_layers:
            xs = []
            for _ in range(100):
                xs.append(self.session.run(
                    self.intent_prediction,
                    feed_dict={
                        _x_in: _x for _x_in, _x in zip(self.batch_in, batch) if _x is not None
                    },
                ))
            xs = np.array(xs)

            max_mask = (xs.max(axis=-1, keepdims=True) == xs).astype(xs.dtype)

            mask_xs = xs * max_mask
            max_xs = mask_xs.sum(axis=0) / (max_mask.sum(axis=0) + 1e-8)
            freq_xs = max_mask.sum(axis=0) / max_mask.shape[0]

            message_sim = 2 * freq_xs * max_xs / (freq_xs + max_xs + 1e-8)
        else:
            message_sim = self.session.run(
                self.intent_prediction,
                feed_dict={
                    _x_in: _x for _x_in, _x in zip(self.batch_in, batch) if _x is not None
                },
            )

        message_sim = message_sim.flatten()  # sim is a matrix

        label_ids = message_sim.argsort()[::-1]
        message_sim[::-1].sort()

        # transform sim to python list for JSON serializing
        return label_ids, message_sim.tolist()

    def predict_entities(self, message: "Message") -> List[Dict]:
        if self.session is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
            return []

        # create session data from message and convert it into a batch of 1
        self.num_tags = len(self.inverted_tag_dict)
        session_data = self._create_session_data([message])
        batch = train_utils.prepare_batch(
            session_data, tuple_sizes=self.batch_tuple_sizes
        )

        # load tf graph and session
        if "pre" in self.var_layers or "crf" in self.var_layers:
            xs = []
            for _ in range(100):
                xs.append(self.session.run(
                    self.entity_prediction,
                    feed_dict={
                        _x_in: _x for _x_in, _x in zip(self.batch_in, batch) if _x is not None
                    },
                ))
            xs = np.array(xs)

            max_mask = (xs.max(axis=-1, keepdims=True) == xs).astype(xs.dtype)

            mask_xs = xs * max_mask
            max_xs = mask_xs.sum(axis=0) / (max_mask.sum(axis=0) + 1e-8)
            freq_xs = max_mask.sum(axis=0) / max_mask.shape[0]

            predictions = 2 * freq_xs * max_xs / (freq_xs + max_xs + 1e-8)
        else:
            predictions = self.session.run(
                self.entity_prediction,
                feed_dict={
                    _x_in: _x for _x_in, _x in zip(self.batch_in, batch) if _x is not None
                },
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

    # methods to overwrite
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
                self.batch_in_strategy,
                label_key="intent_ids",
            )

            self._is_training = tf.placeholder_with_default(False, shape=())

            metrics = self._build_tf_train_graph(session_data)

            # define which optimizer to use
            loss = tf.add_n(list(metrics.loss.values()))
            self._train_op = tf.train.AdamOptimizer().minimize(loss)

            # train tensorflow graph
            self.session = tf.Session(config=self._tf_config)

            train_utils.train_tf_dataset(
                train_init_op,
                eval_init_op,
                batch_size_in,
                metrics,
                self._train_op,
                self.session,
                self._is_training,
                self.epochs,
                self.batch_in_size,
                self.evaluate_on_num_examples,
                self.evaluate_every_num_epochs,
            )

            # rebuild the graph for prediction
            self._build_tf_pred_graph(session_data)

            self.attention_weights = train_utils.extract_attention(
                self.attention_weights
            )

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
            train_utils.persist_tensor("batch_placeholder", self.batch_in, self.graph)

            train_utils.persist_tensor("similarity_all", self.sim_all, self.graph)
            train_utils.persist_tensor(
                "intent_prediction", self.intent_prediction, self.graph
            )
            train_utils.persist_tensor(
                "entity_prediction", self.entity_prediction, self.graph
            )
            train_utils.persist_tensor("similarity", self.sim, self.graph)

            train_utils.persist_tensor("cls_embed", self.cls_embed, self.graph)
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
                cls_embed = train_utils.load_tensor("cls_embed")
                intent_prediction = train_utils.load_tensor("intent_prediction")
                entity_prediction = train_utils.load_tensor("entity_prediction")
                sim = train_utils.load_tensor("similarity")

                message_embed = train_utils.load_tensor("message_embed")
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

            with open(
                os.path.join(model_dir, file_name + ".batch_tuple_sizes.pkl"), "rb"
            ) as f:
                batch_tuple_sizes = pickle.load(f)

            return cls(
                component_config=meta,
                inverted_label_dict=inv_label_dict,
                inverted_tag_dict=inv_tag_dict,
                session=session,
                graph=graph,
                batch_placeholder=batch_in,
                similarity_all=sim_all,
                intent_prediction=intent_prediction,
                entity_prediction=entity_prediction,
                similarity=sim,
                cls_embed=cls_embed,
                label_embed=label_embed,
                all_labels_embed=all_labels_embed,
                attention_weights=attention_weights,
                batch_tuple_sizes=batch_tuple_sizes,
            )

        else:
            logger.warning(
                f"Failed to load nlu model. Maybe path {os.path.abspath(model_dir)} "
                f"doesn't exist"
            )
            return cls(component_config=meta)
