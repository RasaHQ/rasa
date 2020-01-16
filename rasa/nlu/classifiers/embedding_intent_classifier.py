import logging

import numpy as np
import os
import pickle
import scipy.sparse
import typing
import warnings

from typing import Any, Dict, List, Optional, Text, Tuple, Union
from shutil import copyfile

from rasa.nlu.extractors.crf_entity_extractor import CRFEntityExtractor

import rasa.utils.io as io_utils
from rasa.utils.plotter import Plotter
from rasa.nlu.extractors import EntityExtractor
from rasa.nlu.test import determine_token_labels
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.featurizers.featurizer import sequence_to_sentence_features
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.nlu.components import Component, any_of

from rasa.utils import train_utils
from rasa.utils import tf_layers
from rasa.utils import tf_models
from rasa.utils.train_utils import SessionDataType, TrainingMetrics
from rasa.nlu.constants import (
    INTENT_ATTRIBUTE,
    TEXT_ATTRIBUTE,
    ENTITIES_ATTRIBUTE,
    SPARSE_FEATURE_NAMES,
    DENSE_FEATURE_NAMES,
    TOKENS_NAMES,
)

import tensorflow as tf
import tensorflow_addons as tfa

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.nlu.config import RasaNLUModelConfig
    from rasa.nlu.training_data import TrainingData
    from rasa.nlu.model import Metadata
    from rasa.nlu.training_data import Message


MESSAGE_BILOU_ENTITIES_ATTRIBUTE = "BILOU_entities"
shapes, types = None, None


class EmbeddingIntentClassifier(EntityExtractor):
    """label classifier using supervised embeddings.

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
        "hidden_layers_sizes_a": [],
        # sizes of hidden layers before the embedding layer for intent labels
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_b": [],
        # sizes of hidden layers before the embedding layer for tag labels
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_c": [],
        # Whether to share the hidden layer weights between input words and labels
        "share_hidden_layers": False,
        # number of units in transformer
        "transformer_size": 256,
        # number of transformer layers
        "num_transformer_layers": 2,
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
        # optimizer
        "optimizer": "Adam",  # can be either 'Adam' (default) or 'Nadam'
        "learning_rate": 0.001,
        "normalize_loss": False,
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
        "masked_lm_loss": False,
        "sparse_input_dropout": False,
        "bilou_flag": False,
    }
    # end default properties (DOC MARKER - don't remove)

    @staticmethod
    def _check_old_config_variables(config: Dict[Text, Any]) -> None:
        """Config migration warning"""

        removed_tokenization_params = [
            "label_tokenization_flag",
            "label_split_symbol",
        ]
        for removed_param in removed_tokenization_params:
            if removed_param in config:
                warnings.warn(
                    f"label tokenization has been moved to Tokenizer components. "
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
            "tag": config["hidden_layers_sizes_c"],
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

        self.batch_size = config["batch_size"]
        self.batch_strategy = config["batch_strategy"]

        self.optimizer = config["optimizer"]
        self.normalize_loss = config["normalize_loss"]
        self.learning_rate = config["learning_rate"]
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
        self.masked_lm_loss = self.component_config["masked_lm_loss"]
        self.sparse_input_dropout = self.component_config["sparse_input_dropout"]
        self.bilou_flag = self.component_config["bilou_flag"]

    # package safety checks
    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow"]

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        inverted_label_dict: Optional[Dict[int, Text]] = None,
        inverted_tag_dict: Optional[Dict[int, Text]] = None,
        model=None,
        predict_func=None,
        batch_tuple_sizes: Optional[Dict] = None,
        attention_weights: Optional["tf.Tensor"] = None,
    ) -> None:
        """Declare instance variables with default values"""

        super().__init__(component_config)

        self._load_params()

        # transform numbers to labels
        self.inverted_label_dict = inverted_label_dict
        self.inverted_tag_dict = inverted_tag_dict

        self.model = model
        self.predict_func = predict_func

        # encode all label_ids with numbers
        self._label_data = None

        # keep the input tuple sizes in self.batch_in
        self.batch_tuple_sizes = batch_tuple_sizes

        # internal tf instances
        self._iterator = None
        self._train_op = None
        self._is_training = None

        # number of entity tags
        self.num_tags = 0

        self.attention_weights = attention_weights

        self.training_log_file = io_utils.create_temporary_file("")

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
        training_data: "TrainingData", bilou_flag: bool
    ) -> Dict[Text, int]:
        """Create label_id dictionary"""

        if bilou_flag:
            bilou_prefix = ["B-", "I-", "L-", "U-"]
            distinct_tag_ids = set(
                [
                    e[2:]
                    for example in training_data.training_examples
                    if example.get(MESSAGE_BILOU_ENTITIES_ATTRIBUTE)
                    for e in example.get(MESSAGE_BILOU_ENTITIES_ATTRIBUTE)
                ]
            ) - {""}

            tag_id_dict = {
                f"{prefix}{tag_id}": idx_1 * len(bilou_prefix) + idx_2 + 1
                for idx_1, tag_id in enumerate(sorted(distinct_tag_ids))
                for idx_2, prefix in enumerate(bilou_prefix)
            }
            tag_id_dict["O"] = 0

            return tag_id_dict

        distinct_tag_ids = set(
            [
                e["entity"]
                for example in training_data.entity_examples
                for e in example.get(ENTITIES_ATTRIBUTE)
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

        # if attribute != INTENT_ATTRIBUTE:
        #     # Use only the CLS token vector as features
        #     sparse_features = sequence_to_sentence_features(sparse_features)
        #     dense_features = sequence_to_sentence_features(dense_features)

        return sparse_features, dense_features

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

        if not session_data[key]:
            del session_data[key]

    @staticmethod
    def _add_mask_to_session_data(
        session_data: SessionDataType, key: Text, from_key: Text
    ):

        if not session_data.get(from_key):
            return

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

    def check_input_dimension_consistency(self, session_data: "SessionDataType"):
        if self.share_hidden_layers:
            num_text_features = self._get_num_of_features(session_data, "text_features")
            num_intent_features = self._get_num_of_features(
                session_data, "label_features"
            )

            if num_text_features != num_intent_features:
                raise ValueError(
                    "If embeddings are shared "
                    "text features and label features "
                    "must coincide. Check the output dimensions of previous components."
                )

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

    # @staticmethod
    # def _compute_default_label_features(
    #     labels_example: List["Message"],
    # ) -> List[np.ndarray]:
    #     """Compute one-hot representation for the labels"""
    # 
    #     return [
    #         np.array(
    #             [
    #                 scipy.sparse.coo_matrix(
    #                     ([1], ([0], [idx])), shape=(1, len(labels_example))
    #                 )
    #                 for idx in range(len(labels_example))
    #             ]
    #         )
    #     ]

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
            _sparse, _dense = self._extract_and_add_features(e, TEXT_ATTRIBUTE)
            if _sparse is not None:
                X_sparse.append(_sparse)
            if _dense is not None:
                X_dense.append(_dense)

            if e.get(label_attribute):
                _sparse, _dense = self._extract_and_add_features(e, label_attribute)
                if _sparse is not None:
                    Y_sparse.append(_sparse)
                if _dense is not None:
                    Y_dense.append(_dense)

                if label_id_dict:
                    label_ids.append(label_id_dict[e.get(label_attribute)])

            if self.named_entity_recognition and tag_id_dict:
                if self.bilou_flag:
                    if e.get(MESSAGE_BILOU_ENTITIES_ATTRIBUTE):
                        _tags = [
                            tag_id_dict[_tag]
                            if _tag in tag_id_dict
                            else tag_id_dict["O"]
                            for _tag in e.get(MESSAGE_BILOU_ENTITIES_ATTRIBUTE)
                        ]
                    else:
                        _tags = [
                            tag_id_dict["O"]
                            for _ in e.get(TOKENS_NAMES[TEXT_ATTRIBUTE])
                        ]
                else:
                    _tags = []
                    for t in e.get(TOKENS_NAMES[TEXT_ATTRIBUTE]):
                        _tag = determine_token_labels(
                            t, e.get(ENTITIES_ATTRIBUTE), None
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
        self._add_to_session_data(session_data, "label_features", [Y_sparse, Y_dense])
        if label_attribute and (
            "label_features" not in session_data or not session_data["label_features"]
        ):
            # no label features are present, get default features from _label_data
            session_data["label_features"] = self._use_default_label_features(label_ids)

        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        self._add_to_session_data(
            session_data, "label_ids", [np.expand_dims(label_ids, -1)]
        )
        self._add_to_session_data(session_data, "tag_ids", [tag_ids])

        self._add_mask_to_session_data(session_data, "text_mask", "text_features")
        self._add_mask_to_session_data(session_data, "label_mask", "label_features")

        return session_data

    # train helpers
    def _apply_bilou_schema(self, training_data: "TrainingData"):
        if not self.named_entity_recognition:
            return

        for example in training_data.training_examples:
            entities = example.get(ENTITIES_ATTRIBUTE)

            if not entities:
                continue

            entities = CRFEntityExtractor._convert_example(example)
            output = CRFEntityExtractor._bilou_tags_from_offsets(
                example.get(TOKENS_NAMES[TEXT_ATTRIBUTE]), entities
            )

            example.set(MESSAGE_BILOU_ENTITIES_ATTRIBUTE, output)

    def preprocess_train_data(self, training_data: "TrainingData"):
        """Prepares data for training.

        Performs sanity checks on training data, extracts encodings for labels.
        """
        if self.bilou_flag:
            self._apply_bilou_schema(training_data)

        label_id_dict = self._create_label_id_dict(
            training_data, attribute=INTENT_ATTRIBUTE
        )
        self.inverted_label_dict = {v: k for k, v in label_id_dict.items()}

        self._label_data = self._create_label_data(
            training_data, label_id_dict, attribute=INTENT_ATTRIBUTE
        )

        tag_id_dict = self._create_tag_id_dict(training_data, self.bilou_flag)
        self.inverted_tag_dict = {v: k for k, v in tag_id_dict.items()}

        session_data = self._create_session_data(
            training_data.training_examples,
            label_id_dict,
            tag_id_dict,
            label_attribute=INTENT_ATTRIBUTE,
        )

        self.num_tags = len(self.inverted_tag_dict)

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
                label_key="label_ids",
            )
        else:
            eval_session_data = None

        # TODO set it in the model
        # set random seed
        tf.random.set_seed(self.random_seed)

        self.model = DIET(
            session_data,
            eval_session_data,
            self._label_data,
            self.dense_dim,
            self.embed_dim,
            self.hidden_layer_sizes,
            self.share_hidden_layers,
            self.num_transformer_layers,
            self.transformer_size,
            self.num_heads,
            self.max_seq_length,
            self.unidirectional_encoder,
            self.C2,
            self.droprate,
            self.sparse_input_dropout,
            self.num_neg,
            self.loss_type,
            self.mu_pos,
            self.mu_neg,
            self.use_max_sim_neg,
            self.C_emb,
            self.scale_loss,
            self.similarity_type,
            self.masked_lm_loss,
            self.intent_classification,
            self.named_entity_recognition,
            self.inverted_tag_dict,
            self.learning_rate,
            self.batch_strategy,
        )

        self.model.fit(
            self.epochs,
            self.batch_size,
            self.evaluate_on_num_examples,
            self.evaluate_every_num_epochs,
            output_file=self.training_log_file,
        )

        # rebuild the graph for prediction
        # self.model.build_for_predict()

        # self.attention_weights = train_utils.extract_attention(self.attention_weights)

    # process helpers
    def _predict(self, message: "Message"):
        if self.model is None or self.predict_func is None:
            return

        # create session data from message and convert it into a batch of 1
        session_data = self._create_session_data([message])
        self.model.session_data = session_data
        predict_dataset = self.model.predict_dataset()
        batch_in = next(iter(predict_dataset))

        return self.predict_func(batch_in)

    def _predict_label(
            self, out
    ) -> Tuple[Dict[Text, Any], List[Dict[Text, Any]]]:
        """Predicts the intent of the provided message."""

        label = {"name": None, "confidence": 0.0}
        label_ranking = []

        if self.model is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data."
            )
            return label, label_ranking

        message_sim = out["i_scores"].numpy()

        message_sim = message_sim.flatten()  # sim is a matrix

        label_ids = message_sim.argsort()[::-1]
        message_sim[::-1].sort()
        message_sim = message_sim.tolist()

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

    def _predict_entities(self, out, message: "Message") -> List[Dict]:
        if self.model is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
            return []

        # load tf graph and session
        predictions = out["e_ids"].numpy()

        tags = [self.inverted_tag_dict[p] for p in predictions[0]]

        if self.bilou_flag:
            tags = [t[2:] if t[:2] in ["B-", "I-", "U-", "L-"] else t for t in tags]

        entities = self._convert_tags_to_entities(
            message.text, message.get("tokens", []), tags,predictions
        )

        extracted = self.add_extractor_name(entities)
        entities = message.get("entities", []) + extracted

        return entities

    @staticmethod
    def _convert_tags_to_entities(
        text: str, tokens: List[Token], tags: List[Text],predictions
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
                    "start": token.start,
                    "end": token.end,
                    "extractor": "DIET",
                }
                entities.append(entity)

            # belongs to last entity
            elif last_tag == tag:
                entities[-1]["end"] = token.end

            last_tag = tag

        for entity in entities:
            entity["value"] = text[entity["start"]: entity["end"]]

        return entities

    def process(self, message: "Message", **kwargs: Any) -> None:
        """Return the most likely label and its similarity to the input."""

        out = self._predict(message)

        if self.intent_classification:
            label, label_ranking = self._predict_label(out)

            message.set("label", label, add_to_output=True)
            message.set("label_ranking", label_ranking, add_to_output=True)

        if self.named_entity_recognition:
            entities = self._predict_entities(out, message)

            message.set("entities", entities, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

        if self.model is None:
            return {"file": None}

        tf_model_file = os.path.join(model_dir, file_name + ".tf_model")

        # # plot training curves
        # plotter = Plotter()
        # plotter.plot_training_curves(self.training_log_file, model_dir)
        # # copy trainig log file
        # copyfile(self.training_log_file, os.path.join(model_dir, "training-log.tsv"))

        try:
            os.makedirs(os.path.dirname(tf_model_file))
        except OSError as e:
            # be happy if someone already created the path
            import errno

            if e.errno != errno.EEXIST:
                raise

        self.model.save_weights(tf_model_file, save_format='tf')

        dummy_session_data = {
            k: [v[:1] for v in vs]
            for k, vs in self.model.session_data.items()
        }

        with open(
            os.path.join(model_dir, file_name + ".dummy_session_data.pkl"), "wb"
        ) as f:
            pickle.dump(dummy_session_data, f)

        with open(
            os.path.join(model_dir, file_name + ".label_data.pkl"), "wb"
        ) as f:
            pickle.dump(self._label_data, f)

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
        """Loads the trained model from the provided directory."""

        if model_dir and meta.get("file"):
            file_name = meta.get("file")
            tf_model_file = os.path.join(model_dir, file_name + ".tf_model")

            with open(os.path.join(model_dir, file_name + ".tf_config.pkl"), "rb") as f:
                _tf_config = pickle.load(f)

            with open(
                os.path.join(model_dir, file_name + ".dummy_session_data.pkl"), "rb"
            ) as f:
                dummy_session_data = pickle.load(f)

            with open(
                os.path.join(model_dir, file_name + ".label_data.pkl"), "rb"
            ) as f:
                label_data = pickle.load(f)

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

            hidden_layer_sizes = {
                "text": meta["hidden_layers_sizes_a"],
                "label": meta["hidden_layers_sizes_b"],
                "tag": meta["hidden_layers_sizes_c"],
            }
            similarity_type = meta["similarity_type"]
            if similarity_type == "auto":
                if meta["loss_type"] == "softmax":
                    similarity_type = "inner"
                elif meta["loss_type"] == "margin":
                    similarity_type = "cosine"

            model = DIET(
                dummy_session_data,
                None,
                label_data,
                meta["dense_dim"],
                meta["embed_dim"],
                hidden_layer_sizes,
                meta["share_hidden_layers"],
                meta["num_transformer_layers"],
                meta["transformer_size"],
                meta["num_heads"],
                meta["max_seq_length"],
                meta["unidirectional_encoder"],
                meta["C2"],
                meta["droprate"],
                meta["sparse_input_dropout"],
                meta["num_neg"],
                meta["loss_type"],
                meta["mu_pos"],
                meta["mu_neg"],
                meta["use_max_sim_neg"],
                meta["C_emb"],
                meta["scale_loss"],
                similarity_type,
                meta["masked_lm_loss"],
                meta["intent_classification"],
                meta["named_entity_recognition"],
                inv_tag_dict,
                meta["learning_rate"],
                meta["batch_strategy"],
            )
            logger.debug("Loading the model ...")
            model.fit(
                1,
                1,
                0,
                0,
                silent=True,
                eager=True,
            )
            model.load_weights(tf_model_file)

            # build the graph for prediction
            model.set_training_phase(False)
            model.session_data = {
                k: vs
                for k, vs in model.session_data.items()
                if "text" in k
            }
            model.build_for_predict()
            predict_dataset = model.predict_dataset()
            predict_func = tf.function(
                model.predict, input_signature=[predict_dataset.element_spec]
            )
            batch_in = next(iter(predict_dataset))
            predict_func(batch_in)
            logger.debug("Finished loading the model.")

            return cls(
                component_config=meta,
                inverted_label_dict=inv_label_dict,
                inverted_tag_dict=inv_tag_dict,
                model=model,
                predict_func=predict_func,
                batch_tuple_sizes=batch_tuple_sizes,
            )

        else:
            warnings.warn(
                f"Failed to load nlu model. "
                f"Maybe path '{os.path.abspath(model_dir)}' doesn't exist."
            )
            return cls(component_config=meta)


class DIET(tf_models.RasaModel):
    @staticmethod
    def _create_sparse_dense_layer(values, name, reg_lambda, dense_dim):

        sparse = False
        for v in values:
            if isinstance(v[0], scipy.sparse.spmatrix):
                sparse = True
            else:
                # if dense features are present
                # use the feature dimension of the dense features
                dense_dim = v[0].shape[-1]

        if sparse:
            return tf_layers.DenseForSparse(
                units=dense_dim, reg_lambda=reg_lambda, name=name
            )

    @staticmethod
    def _input_dim(values, dense_dim):

        for v in values:
            if not isinstance(v[0], scipy.sparse.spmatrix):
                # if dense features are present
                # use the feature dimension of the dense features
                dense_dim = v[0].shape[-1]
                break

        return dense_dim * len(values)

    def __init__(
        self,
        session_data,
        eval_session_data,
        label_data,
        dense_dim,
        embed_dim,
        hidden_layer_sizes,
        share_hidden_layers,
        num_transformer_layers,
        transformer_size,
        num_heads,
        max_seq_length,
        unidirectional_encoder,
        reg_lambda,
        droprate,
        sparse_input_dropout,
        num_neg,
        loss_type,
        mu_pos,
        mu_neg,
        use_max_sim_neg,
        C_emb,
        scale_loss,
        similarity_type,
        masked_lm_loss,
        intent_classification,
        named_entity_recognition,
        inverted_tag_dict,
        learning_rate,
        batch_in_strategy,
    ):
        super(DIET, self).__init__(name="DIET")

        # data
        self.session_data = session_data
        self.eval_session_data = eval_session_data
        label_batch = train_utils.prepare_batch(label_data)
        self.tf_label_data = train_utils.batch_to_session_data(
            label_batch, label_data
        )

        # options
        self._sparse_input_dropout = sparse_input_dropout
        self._num_neg = num_neg
        self._loss_type = loss_type
        self._mu_pos = mu_pos
        self._mu_neg = mu_neg
        self._use_max_sim_neg = use_max_sim_neg
        self._C_emb = C_emb
        self._scale_loss = scale_loss
        self._similarity_type = similarity_type
        self._masked_lm_loss = masked_lm_loss
        self._intent_classification = intent_classification
        self._named_entity_recognition = named_entity_recognition
        self._inverted_tag_dict = inverted_tag_dict
        self._num_tags = len(inverted_tag_dict)
        self._batch_in_strategy = batch_in_strategy

        # tf objects
        self._sparse_dropout = tf_layers.SparseDropout(rate=droprate)
        self._sparse_to_dense = {
            "text": self._create_sparse_dense_layer(
                session_data["text_features"], "text", reg_lambda, dense_dim["text"]
            ),
            "label": self._create_sparse_dense_layer(
                session_data["label_features"], "label", reg_lambda, dense_dim["label"]
            ),
        }

        self._ffnn = {
            "text": tf_layers.ReluFfn(
                hidden_layer_sizes["text"],
                droprate,
                reg_lambda,
                "text_intent" if share_hidden_layers else "text",
            ),
            "label": tf_layers.ReluFfn(
                hidden_layer_sizes["label"],
                droprate,
                reg_lambda,
                "text_intent" if share_hidden_layers else "label",
            ),
        }

        if num_transformer_layers > 0:
            self._transformer = tf_layers.TransformerEncoder(
                num_transformer_layers,
                transformer_size,
                num_heads,
                transformer_size * 4,
                max_seq_length,
                reg_lambda,
                droprate,
                unidirectional_encoder,
                name="text_encoder",
            )
        else:
            self._transformer = lambda x, mask, training: x

        self._embed = {}
        self.train_metrics = {"t_loss": tf.keras.metrics.Mean(name="t_loss")}
        self.eval_metrics = {"val_t_loss": tf.keras.metrics.Mean(name="val_t_loss")}
        if self._masked_lm_loss:
            self._input_mask = tf_layers.InputMask()
            self._embed["text_mask"] = tf_layers.Embed(
                embed_dim, reg_lambda, "text_mask", similarity_type
            )
            self._embed["text_token"] = tf_layers.Embed(
                embed_dim, reg_lambda, "text_token", similarity_type
            )
            self._loss_mask = tf_layers.DotProductLoss(
                self._num_neg,
                self._loss_type,
                self._mu_pos,
                self._mu_neg,
                self._use_max_sim_neg,
                self._C_emb,
                self._scale_loss,
            )
            self.train_metrics["m_loss"] = tf.keras.metrics.Mean(name="m_loss")
            self.train_metrics["m_acc"] = tf.keras.metrics.Mean(name="m_acc")
            self.eval_metrics["val_m_loss"] = tf.keras.metrics.Mean(name="val_m_loss")
            self.eval_metrics["val_m_acc"] = tf.keras.metrics.Mean(name="val_m_acc")
        else:
            self._input_mask = None

        if self._intent_classification:
            self._embed["text"] = tf_layers.Embed(
                embed_dim, reg_lambda, "text", similarity_type
            )
            self._embed["label"] = tf_layers.Embed(
                embed_dim, reg_lambda, "label", similarity_type
            )
            self._loss_label = tf_layers.DotProductLoss(
                self._num_neg,
                self._loss_type,
                self._mu_pos,
                self._mu_neg,
                self._use_max_sim_neg,
                self._C_emb,
                self._scale_loss,
            )
            self.train_metrics["i_loss"] = tf.keras.metrics.Mean(name="i_loss")
            self.train_metrics["i_acc"] = tf.keras.metrics.Mean(name="i_acc")
            self.eval_metrics["val_i_loss"] = tf.keras.metrics.Mean(name="val_i_loss")
            self.eval_metrics["val_i_acc"] = tf.keras.metrics.Mean(name="val_i_acc")

        if self._named_entity_recognition:
            self._embed["logits"] = tf_layers.Embed(
                self._num_tags, reg_lambda, "logits"
            )
            self._crf = tf_layers.CRF(self._num_tags, reg_lambda)
            self.train_metrics["e_loss"] = tf.keras.metrics.Mean(name="e_loss")
            self.train_metrics["e_f1"] = tf.keras.metrics.Mean(name="e_f1")
            self.eval_metrics["val_e_loss"] = tf.keras.metrics.Mean(name="val_e_loss")
            self.eval_metrics["val_e_f1"] = tf.keras.metrics.Mean(name="val_e_f1")
        else:
            self._crf = None

        # tf tensors
        self.training = tf.ones((), tf.bool)

        # tf training
        self._optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.entity_f1 = tfa.metrics.F1Score(
            num_classes=self._num_tags - 1,  # `0` prediction is not a prediction
            average="micro",
        )

        # persist
        self.all_labels_embed = None
        self.batch_tuple_sizes = None

    def set_training_phase(self, training: bool):
        if training:
            self.training = tf.ones((), tf.bool)
        else:
            self.training = tf.zeros((), tf.bool)

    def _combine_sparse_dense_features(
        self,
        features: List[Union["tf.Tensor", "tf.SparseTensor"]],
        mask: "tf.Tensor",
        name: Text,
        sparse_dropout: bool = False,
    ) -> "tf.Tensor":

        dense_features = []

        for f in features:
            if isinstance(f, tf.SparseTensor):
                if sparse_dropout:
                    _f = self._sparse_dropout(f, self.training)
                else:
                    _f = f

                dense_features.append(self._sparse_to_dense[name](_f))
            else:
                dense_features.append(f)

        return tf.concat(dense_features, axis=-1) * mask

    def _create_bow(
        self,
        features: List[Union["tf.Tensor", "tf.SparseTensor"]],
        mask: "tf.Tensor",
        name: Text,
    ):

        x = self._combine_sparse_dense_features(features, mask, name)
        return self._ffnn[name](tf.reduce_sum(x, 1), self.training)

    def _create_sequence(
        self,
        features: List[Union["tf.Tensor", "tf.SparseTensor"]],
        mask: "tf.Tensor",
        name: Text,
        masked_lm_loss: bool = False,
    ):
        x = self._combine_sparse_dense_features(
            features, mask, name, sparse_dropout=self._sparse_input_dropout
        )

        if masked_lm_loss:
            pre, lm_mask_bool = self._input_mask(x, mask, self.training)
        else:
            pre, lm_mask_bool = (x, None)

        transformed = self._transformer(pre, 1 - mask, self.training)
        transformed = tf.nn.relu(transformed)

        return transformed, x, lm_mask_bool

    def _mask_loss(self, a_transformed, a, lm_mask_bool, name):
        # make sure there is at least one element in the mask
        lm_mask_bool = tf.cond(
            tf.reduce_any(lm_mask_bool),
            lambda: lm_mask_bool,
            lambda: tf.scatter_nd([[0, 0, 0]], [True], tf.shape(lm_mask_bool)),
        )

        lm_mask_bool = tf.squeeze(lm_mask_bool, -1)
        a_t_masked = tf.boolean_mask(a_transformed, lm_mask_bool)
        a_masked = tf.boolean_mask(a, lm_mask_bool)

        a_t_masked_embed = self._embed[f"{name}_mask"](a_t_masked)
        a_masked_embed = self._embed[f"{name}_token"](a_masked)

        return self._loss_mask(
            a_t_masked_embed,
            a_masked_embed,
            a_masked,
            a_masked_embed,
            a_masked,
        )

    def _build_all_b(self):
        all_labels = self._create_bow(
            self.tf_label_data["label_features"],
            self.tf_label_data["label_mask"][0],
            "label",
        )
        all_labels_embed = self._embed["label"](all_labels)

        return all_labels_embed, all_labels

    def _intent_loss(self, a, b):
        all_labels_embed, all_labels = self._build_all_b()

        a_embed = self._embed["text"](a)
        b_embed = self._embed["label"](b)

        return self._loss_label(
            a_embed,
            b_embed,
            b,
            all_labels_embed,
            all_labels,
        )

    def _entity_loss(
        self, a: "tf.Tensor", c: "tf.Tensor", mask: "tf.Tensor", sequence_lengths
    ) -> Tuple["tf.Tensor", "tf.Tensor"]:

        # remove cls token
        sequence_lengths = sequence_lengths - 1
        c = tf.cast(c[:, :, 0], tf.int32)
        logits = self._embed["logits"](a)

        # tensor shapes
        # a: tensor(batch-size, max-seq-len, dim)
        # sequence_lengths: tensor(batch-size)
        # c: (batch-size, max-seq-len)

        # CRF Loss
        loss = self._crf.loss(logits, c, sequence_lengths)

        # CRF preds
        pred_ids = self._crf(logits, sequence_lengths)

        # TODO check that f1 calculation is correct
        # calculate f1 score for train predictions
        mask_bool = tf.cast(mask[:, :, 0], tf.bool)
        # pick only non padding values and flatten sequences
        c_masked = tf.boolean_mask(c, mask_bool)
        pred_ids_masked = tf.boolean_mask(pred_ids, mask_bool)
        # set `0` prediction to not a prediction
        c_masked_1 = tf.one_hot(c_masked - 1, self._num_tags - 1)
        pred_ids_masked_1 = tf.one_hot(pred_ids_masked - 1, self._num_tags - 1)

        f1 = self.entity_f1(c_masked_1, pred_ids_masked_1)

        return loss, f1

    def _train_losses_scores(self, batch_in):
        tf_batch_data = train_utils.batch_to_session_data(
            batch_in, self.session_data
        )

        mask_text = tf_batch_data["text_mask"][0]
        sequence_lengths = tf.cast(tf.reduce_sum(mask_text[:, :, 0], 1), tf.int32)

        text_transformed, text_in, lm_mask_bool_text = self._create_sequence(
            tf_batch_data["text_features"], mask_text, "text", self._masked_lm_loss
        )

        losses = {}
        scores = {}

        if self._masked_lm_loss:
            loss, acc = self._mask_loss(
                text_transformed, text_in, lm_mask_bool_text, "text"
            )
            losses["m_loss"] = loss
            scores["m_acc"] = acc

        if self._intent_classification:
            # get _cls_ vector for intent classification
            last_index = tf.maximum(
                tf.constant(0, dtype=sequence_lengths.dtype), sequence_lengths - 1
            )
            idxs = tf.stack([tf.range(tf.shape(last_index)[0]), last_index], axis=1)
            cls = tf.gather_nd(text_transformed, idxs)

            label = self._create_bow(
                tf_batch_data["label_features"],
                tf_batch_data["label_mask"][0],
                "label",
            )
            loss, acc = self._intent_loss(cls, label)
            losses["i_loss"] = loss
            scores["i_acc"] = acc

        if self._named_entity_recognition:
            tags = tf_batch_data["tag_ids"][0]

            loss, f1 = self._entity_loss(
                text_transformed, tags, mask_text, sequence_lengths
            )
            losses["e_loss"] = loss
            scores["e_f1"] = f1

        return losses, scores

    def train_on_batch(self, batch_in):
        with tf.GradientTape() as tape:
            losses, scores = self._train_losses_scores(batch_in)
            regularization_loss = tf.math.add_n(self.losses)
            pred_loss = tf.math.add_n(list(losses.values()))
            total_loss = pred_loss + regularization_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self._optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.train_metrics["t_loss"].update_state(total_loss)
        for k, v in losses.items():
            self.train_metrics[k].update_state(v)
        for k, v in scores.items():
            self.train_metrics[k].update_state(v)

    def train_dataset(self, batch_size):
        return train_utils.create_tf_dataset(
            self.session_data,
            batch_size,
            label_key="label_ids",
            batch_strategy=self._batch_in_strategy,
            shuffle=True,
        )

    def eval(self, batch_in):
        losses, scores = self._train_losses_scores(batch_in)
        total_loss = tf.math.add_n(list(losses.values())) + self.losses

        self.eval_metrics["val_t_loss"].update_state(total_loss)
        for k, v in losses.items():
            self.eval_metrics[f"val_{k}"].update_state(v)
        for k, v in scores.items():
            self.eval_metrics[f"val_{k}"].update_state(v)

    def eval_dataset(self, batch_size):
        if self.eval_session_data is not None:
            return train_utils.create_tf_dataset(
                self.eval_session_data, batch_size, label_key="label_ids"
            )

    def build_for_predict(self):
        self.batch_tuple_sizes = train_utils.batch_tuple_sizes(self.session_data)

        all_labels_embed, _ = self._build_all_b()
        self.all_labels_embed = tf.constant(all_labels_embed.numpy())

    def predict(self, batch_in):
        tf_batch_data = train_utils.batch_to_session_data(
            batch_in, self.session_data
        )

        mask_text = tf_batch_data["text_mask"][0]
        sequence_lengths = tf.cast(tf.reduce_sum(mask_text[:, :, 0], 1), tf.int32)

        text_transformed, _, _ = self._create_sequence(
            tf_batch_data["text_features"], mask_text, "text"
        )

        out = {}
        if self._intent_classification:
            # get _cls_ vector for intent classification
            last_index = tf.maximum(
                tf.constant(0, dtype=sequence_lengths.dtype), sequence_lengths - 1
            )
            idxs = tf.stack([tf.range(tf.shape(last_index)[0]), last_index], axis=1)
            cls = tf.gather_nd(text_transformed, idxs)
            cls_embed = self._embed["text"](cls)

            sim_all = self._loss_label.sim(
                cls_embed[:, tf.newaxis, :],
                self.all_labels_embed[tf.newaxis, :, :],
                None,
            )
            # label = self._create_bow(
            #     tf_batch_data["label_features"],
            #     tf_batch_data["label_mask"][0],
            #     "label",
            # )
            # label_embed = self._embed["label"](label)
            # sim = train_utils.tf_raw_sim(
            #     cls_embed[:, tf.newaxis, :], label_embed, None
            # )

            scores = train_utils.confidence_from_sim(
                sim_all, self._similarity_type
            )
            out["i_scores"] = scores

        if self._named_entity_recognition:
            sequence_lengths = sequence_lengths - 1
            logits = self._embed["logits"](text_transformed)
            pred_ids = self._crf(logits, sequence_lengths)
            out["e_ids"] = pred_ids

        return out

    def predict_dataset(self):
        return train_utils.create_tf_dataset(
            self.session_data,
            1,
            label_key="label_ids",
        )
