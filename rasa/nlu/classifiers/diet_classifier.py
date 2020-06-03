import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import os
import scipy.sparse
import tensorflow as tf
import tensorflow_addons as tfa

from typing import Any, Dict, List, Optional, Text, Tuple, Union, Type, NamedTuple

import rasa.utils.common as common_utils
import rasa.utils.io as io_utils
import rasa.nlu.utils.bilou_utils as bilou_utils
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.components import Component
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.extractors.extractor import EntityExtractor
from rasa.nlu.test import determine_token_labels
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.utils import train_utils
from rasa.utils.tensorflow import layers
from rasa.utils.tensorflow.transformer import TransformerEncoder
from rasa.utils.tensorflow.models import RasaModel
from rasa.utils.tensorflow.model_data import RasaModelData, FeatureSignature
from rasa.nlu.constants import (
    INTENT,
    TEXT,
    ENTITIES,
    NO_ENTITY_TAG,
    SPARSE_FEATURE_NAMES,
    DENSE_FEATURE_NAMES,
    TOKENS_NAMES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
)
from rasa.nlu.config import RasaNLUModelConfig, InvalidConfigError
from rasa.nlu.training_data import TrainingData
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import Message
from rasa.utils.tensorflow.constants import (
    LABEL,
    HIDDEN_LAYERS_SIZES,
    SHARE_HIDDEN_LAYERS,
    TRANSFORMER_SIZE,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
    LEARNING_RATE,
    DENSE_DIMENSION,
    RANKING_LENGTH,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    SPARSE_INPUT_DROPOUT,
    DENSE_INPUT_DROPOUT,
    MASKED_LM,
    ENTITY_RECOGNITION,
    TENSORBOARD_LOG_DIR,
    INTENT_CLASSIFICATION,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    UNIDIRECTIONAL_ENCODER,
    DROP_RATE,
    DROP_RATE_ATTENTION,
    WEIGHT_SPARSITY,
    NEGATIVE_MARGIN_SCALE,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    BILOU_FLAG,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    SOFTMAX,
    AUTO,
    BALANCED,
    TENSORBOARD_LOG_LEVEL,
)


logger = logging.getLogger(__name__)

TEXT_FEATURES = f"{TEXT}_features"
LABEL_FEATURES = f"{LABEL}_features"
LABEL_IDS = f"{LABEL}_ids"
TAG_IDS = "tag_ids"
TEXT_SEQ_LENGTH = f"{TEXT}_lengths"
LABEL_SEQ_LENGTH = f"{LABEL}_lengths"

POSSIBLE_TAGS = [ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_ROLE, ENTITY_ATTRIBUTE_GROUP]


class EntityTagSpec(NamedTuple):
    """Specification of an entity tag present in the training data."""

    tag_name: Text
    ids_to_tags: Dict[int, Text]
    tags_to_ids: Dict[Text, int]
    num_tags: int


class DIETClassifier(IntentClassifier, EntityExtractor):
    """DIET (Dual Intent and Entity Transformer) is a multi-task architecture for
    intent classification and entity recognition.

    The architecture is based on a transformer which is shared for both tasks.
    A sequence of entity labels is predicted through a Conditional Random Field (CRF)
    tagging layer on top of the transformer output sequence corresponding to the
    input sequence of tokens. The transformer output for the ``__CLS__`` token and
    intent labels are embedded into a single semantic vector space. We use the
    dot-product loss to maximize the similarity with the target label and minimize
    similarities with negative samples.
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Featurizer]

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # ## Architecture of the used neural network
        # Hidden layer sizes for layers before the embedding layers for user message
        # and labels.
        # The number of hidden layers is equal to the length of the corresponding
        # list.
        HIDDEN_LAYERS_SIZES: {TEXT: [], LABEL: []},
        # Whether to share the hidden layer weights between user message and labels.
        SHARE_HIDDEN_LAYERS: False,
        # Number of units in transformer
        TRANSFORMER_SIZE: 256,
        # Number of transformer layers
        NUM_TRANSFORMER_LAYERS: 2,
        # Number of attention heads in transformer
        NUM_HEADS: 4,
        # If 'True' use key relative embeddings in attention
        KEY_RELATIVE_ATTENTION: False,
        # If 'True' use value relative embeddings in attention
        VALUE_RELATIVE_ATTENTION: False,
        # Max position for relative embeddings
        MAX_RELATIVE_POSITION: None,
        # Use a unidirectional or bidirectional encoder.
        UNIDIRECTIONAL_ENCODER: False,
        # ## Training parameters
        # Initial and final batch sizes:
        # Batch size will be linearly increased for each epoch.
        BATCH_SIZES: [64, 256],
        # Strategy used when creating batches.
        # Can be either 'sequence' or 'balanced'.
        BATCH_STRATEGY: BALANCED,
        # Number of epochs to train
        EPOCHS: 300,
        # Set random seed to any 'int' to get reproducible results
        RANDOM_SEED: None,
        # Initial learning rate for the optimizer
        LEARNING_RATE: 0.001,
        # ## Parameters for embeddings
        # Dimension size of embedding vectors
        EMBEDDING_DIMENSION: 20,
        # Default dense dimension to use if no dense features are present.
        DENSE_DIMENSION: {TEXT: 512, LABEL: 20},
        # The number of incorrect labels. The algorithm will minimize
        # their similarity to the user input during training.
        NUM_NEG: 20,
        # Type of similarity measure to use, either 'auto' or 'cosine' or 'inner'.
        SIMILARITY_TYPE: AUTO,
        # The type of the loss function, either 'softmax' or 'margin'.
        LOSS_TYPE: SOFTMAX,
        # Number of top actions to normalize scores for loss type 'softmax'.
        # Set to 0 to turn off normalization.
        RANKING_LENGTH: 10,
        # Indicates how similar the algorithm should try to make embedding vectors
        # for correct labels.
        # Should be 0.0 < ... < 1.0 for 'cosine' similarity type.
        MAX_POS_SIM: 0.8,
        # Maximum negative similarity for incorrect labels.
        # Should be -1.0 < ... < 1.0 for 'cosine' similarity type.
        MAX_NEG_SIM: -0.4,
        # If 'True' the algorithm only minimizes maximum similarity over
        # incorrect intent labels, used only if 'loss_type' is set to 'margin'.
        USE_MAX_NEG_SIM: True,
        # If 'True' scale loss inverse proportionally to the confidence
        # of the correct prediction
        SCALE_LOSS: False,
        # ## Regularization parameters
        # The scale of regularization
        REGULARIZATION_CONSTANT: 0.002,
        # The scale of how important is to minimize the maximum similarity
        # between embeddings of different labels,
        # used only if 'loss_type' is set to 'margin'.
        NEGATIVE_MARGIN_SCALE: 0.8,
        # Dropout rate for encoder
        DROP_RATE: 0.2,
        # Dropout rate for attention
        DROP_RATE_ATTENTION: 0,
        # Sparsity of the weights in dense layers
        WEIGHT_SPARSITY: 0.8,
        # If 'True' apply dropout to sparse input tensors
        SPARSE_INPUT_DROPOUT: True,
        # If 'True' apply dropout to dense input tensors
        DENSE_INPUT_DROPOUT: True,
        # ## Evaluation parameters
        # How often calculate validation accuracy.
        # Small values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EPOCHS: 20,
        # How many examples to use for hold out validation set
        # Large values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EXAMPLES: 0,
        # ## Model config
        # If 'True' intent classification is trained and intent predicted.
        INTENT_CLASSIFICATION: True,
        # If 'True' named entity recognition is trained and entities predicted.
        ENTITY_RECOGNITION: True,
        # If 'True' random tokens of the input message will be masked and the model
        # should predict those tokens.
        MASKED_LM: False,
        # 'BILOU_flag' determines whether to use BILOU tagging or not.
        # If set to 'True' labelling is more rigorous, however more
        # examples per entity are required.
        # Rule of thumb: you should have more than 100 examples per entity.
        BILOU_FLAG: True,
        # If you want to use tensorboard to visualize training and validation metrics,
        # set this option to a valid output directory.
        TENSORBOARD_LOG_DIR: None,
        # Define when training metrics for tensorboard should be logged.
        # Either after every epoch or for every training step.
        # Valid values: 'epoch' and 'minibatch'
        TENSORBOARD_LOG_LEVEL: "epoch",
    }

    # init helpers
    def _check_masked_lm(self) -> None:
        if (
            self.component_config[MASKED_LM]
            and self.component_config[NUM_TRANSFORMER_LAYERS] == 0
        ):
            raise ValueError(
                f"If number of transformer layers is 0, "
                f"'{MASKED_LM}' option should be 'False'."
            )

    def _check_share_hidden_layers_sizes(self) -> None:
        if self.component_config.get(SHARE_HIDDEN_LAYERS):
            first_hidden_layer_sizes = next(
                iter(self.component_config[HIDDEN_LAYERS_SIZES].values())
            )
            # check that all hidden layer sizes are the same
            identical_hidden_layer_sizes = all(
                current_hidden_layer_sizes == first_hidden_layer_sizes
                for current_hidden_layer_sizes in self.component_config[
                    HIDDEN_LAYERS_SIZES
                ].values()
            )
            if not identical_hidden_layer_sizes:
                raise ValueError(
                    f"If hidden layer weights are shared, "
                    f"{HIDDEN_LAYERS_SIZES} must coincide."
                )

    def _check_config_parameters(self) -> None:
        self.component_config = train_utils.check_deprecated_options(
            self.component_config
        )

        self._check_masked_lm()
        self._check_share_hidden_layers_sizes()

        self.component_config = train_utils.update_similarity_type(
            self.component_config
        )
        self.component_config = train_utils.update_evaluation_parameters(
            self.component_config
        )

    # package safety checks
    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow"]

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        index_label_id_mapping: Optional[Dict[int, Text]] = None,
        entity_tag_specs: Optional[List[EntityTagSpec]] = None,
        model: Optional[RasaModel] = None,
    ) -> None:
        """Declare instance variables with default values."""

        if component_config is not None and EPOCHS not in component_config:
            common_utils.raise_warning(
                f"Please configure the number of '{EPOCHS}' in your configuration file."
                f" We will change the default value of '{EPOCHS}' in the future to 1. "
            )

        super().__init__(component_config)

        self._check_config_parameters()

        # transform numbers to labels
        self.index_label_id_mapping = index_label_id_mapping

        self._entity_tag_specs = entity_tag_specs

        self.model = model

        self._label_data: Optional[RasaModelData] = None
        self._data_example: Optional[Dict[Text, List[np.ndarray]]] = None

    @property
    def label_key(self) -> Optional[Text]:
        return LABEL_IDS if self.component_config[INTENT_CLASSIFICATION] else None

    @staticmethod
    def model_class() -> Type[RasaModel]:
        return DIET

    # training data helpers:
    @staticmethod
    def _label_id_index_mapping(
        training_data: TrainingData, attribute: Text
    ) -> Dict[Text, int]:
        """Create label_id dictionary."""

        distinct_label_ids = {
            example.get(attribute) for example in training_data.intent_examples
        } - {None}
        return {
            label_id: idx for idx, label_id in enumerate(sorted(distinct_label_ids))
        }

    @staticmethod
    def _invert_mapping(mapping: Dict) -> Dict:
        return {value: key for key, value in mapping.items()}

    def _create_entity_tag_specs(
        self, training_data: TrainingData
    ) -> List[EntityTagSpec]:
        """Create entity tag specifications with their respective tag id mappings."""

        _tag_specs = []

        for tag_name in POSSIBLE_TAGS:
            if self.component_config[BILOU_FLAG]:
                tag_id_index_mapping = bilou_utils.build_tag_id_dict(
                    training_data, tag_name
                )
            else:
                tag_id_index_mapping = self._tag_id_index_mapping_for(
                    tag_name, training_data
                )

            if tag_id_index_mapping:
                _tag_specs.append(
                    EntityTagSpec(
                        tag_name=tag_name,
                        tags_to_ids=tag_id_index_mapping,
                        ids_to_tags=self._invert_mapping(tag_id_index_mapping),
                        num_tags=len(tag_id_index_mapping),
                    )
                )

        return _tag_specs

    @staticmethod
    def _tag_id_index_mapping_for(
        tag_name: Text, training_data: TrainingData
    ) -> Optional[Dict[Text, int]]:
        """Create mapping from tag name to id."""
        if tag_name == ENTITY_ATTRIBUTE_ROLE:
            distinct_tags = training_data.entity_roles
        elif tag_name == ENTITY_ATTRIBUTE_GROUP:
            distinct_tags = training_data.entity_groups
        else:
            distinct_tags = training_data.entities

        distinct_tags = distinct_tags - {NO_ENTITY_TAG} - {None}

        if not distinct_tags:
            return None

        tag_id_dict = {
            tag_id: idx for idx, tag_id in enumerate(sorted(distinct_tags), 1)
        }
        # NO_ENTITY_TAG corresponds to non-entity which should correspond to 0 index
        # needed for correct prediction for padding
        tag_id_dict[NO_ENTITY_TAG] = 0

        return tag_id_dict

    @staticmethod
    def _find_example_for_label(
        label: Text, examples: List[Message], attribute: Text
    ) -> Optional[Message]:
        for ex in examples:
            if ex.get(attribute) == label:
                return ex
        return None

    @staticmethod
    def _check_labels_features_exist(
        labels_example: List[Message], attribute: Text
    ) -> bool:
        """Checks if all labels have features set."""

        return all(
            label_example.get(SPARSE_FEATURE_NAMES[attribute]) is not None
            or label_example.get(DENSE_FEATURE_NAMES[attribute]) is not None
            for label_example in labels_example
        )

    def _extract_features(
        self, message: Message, attribute: Text
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

        # If we don't use the transformer and we don't want to do entity recognition,
        # to speed up training take only the sentence features as feature vector.
        # It corresponds to the feature vector for the last token - CLS token.
        # We would not make use of the sequence anyway in this setup. Carrying over
        # those features to the actual training process takes quite some time.
        if (
            self.component_config[NUM_TRANSFORMER_LAYERS] == 0
            and not self.component_config[ENTITY_RECOGNITION]
            and attribute != INTENT
        ):
            sparse_features = train_utils.sequence_to_sentence_features(sparse_features)
            dense_features = train_utils.sequence_to_sentence_features(dense_features)

        return sparse_features, dense_features

    def _check_input_dimension_consistency(self, model_data: RasaModelData) -> None:
        """Checks if features have same dimensionality if hidden layers are shared."""

        if self.component_config.get(SHARE_HIDDEN_LAYERS):
            num_text_features = model_data.feature_dimension(TEXT_FEATURES)
            num_label_features = model_data.feature_dimension(LABEL_FEATURES)

            if num_text_features != num_label_features:
                raise ValueError(
                    "If embeddings are shared text features and label features "
                    "must coincide. Check the output dimensions of previous components."
                )

    def _extract_labels_precomputed_features(
        self, label_examples: List[Message], attribute: Text = INTENT
    ) -> List[np.ndarray]:
        """Collects precomputed encodings."""

        sparse_features = []
        dense_features = []

        for e in label_examples:
            _sparse, _dense = self._extract_features(e, attribute)
            if _sparse is not None:
                sparse_features.append(_sparse)
            if _dense is not None:
                dense_features.append(_dense)

        sparse_features = np.array(sparse_features)
        dense_features = np.array(dense_features)

        return [sparse_features, dense_features]

    @staticmethod
    def _compute_default_label_features(
        labels_example: List[Message],
    ) -> List[np.ndarray]:
        """Computes one-hot representation for the labels."""

        eye_matrix = np.eye(len(labels_example), dtype=np.float32)
        # add sequence dimension to one-hot labels
        return [np.array([np.expand_dims(a, 0) for a in eye_matrix])]

    def _create_label_data(
        self,
        training_data: TrainingData,
        label_id_dict: Dict[Text, int],
        attribute: Text,
    ) -> RasaModelData:
        """Create matrix with label_ids encoded in rows as bag of words.

        Find a training example for each label and get the encoded features
        from the corresponding Message object.
        If the features are already computed, fetch them from the message object
        else compute a one hot encoding for the label as the feature vector.
        """

        # Collect one example for each label
        labels_idx_examples = []
        for label_name, idx in label_id_dict.items():
            label_example = self._find_example_for_label(
                label_name, training_data.intent_examples, attribute
            )
            labels_idx_examples.append((idx, label_example))

        # Sort the list of tuples based on label_idx
        labels_idx_examples = sorted(labels_idx_examples, key=lambda x: x[0])
        labels_example = [example for (_, example) in labels_idx_examples]

        # Collect features, precomputed if they exist, else compute on the fly
        if self._check_labels_features_exist(labels_example, attribute):
            features = self._extract_labels_precomputed_features(
                labels_example, attribute
            )
        else:
            features = self._compute_default_label_features(labels_example)

        label_data = RasaModelData()
        label_data.add_features(LABEL_FEATURES, features)

        label_ids = np.array([idx for (idx, _) in labels_idx_examples])
        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        label_data.add_features(LABEL_IDS, [np.expand_dims(label_ids, -1)])

        label_data.add_lengths(LABEL_SEQ_LENGTH, LABEL_FEATURES)

        return label_data

    def _use_default_label_features(self, label_ids: np.ndarray) -> List[np.ndarray]:
        all_label_features = self._label_data.get(LABEL_FEATURES)[0]
        return [np.array([all_label_features[label_id] for label_id in label_ids])]

    def _create_model_data(
        self,
        training_data: List[Message],
        label_id_dict: Optional[Dict[Text, int]] = None,
        label_attribute: Optional[Text] = None,
        training: bool = True,
    ) -> RasaModelData:
        """Prepare data for training and create a RasaModelData object"""

        X_sparse = []
        X_dense = []
        Y_sparse = []
        Y_dense = []
        label_ids = []
        tag_name_to_tag_ids = defaultdict(list)

        for example in training_data:
            if label_attribute is None or example.get(label_attribute):
                _sparse, _dense = self._extract_features(example, TEXT)
                if _sparse is not None:
                    X_sparse.append(_sparse)
                if _dense is not None:
                    X_dense.append(_dense)

            # only add features for intent labels during training
            if training and example.get(label_attribute):
                _sparse, _dense = self._extract_features(example, label_attribute)
                if _sparse is not None:
                    Y_sparse.append(_sparse)
                if _dense is not None:
                    Y_dense.append(_dense)

                if label_id_dict:
                    label_ids.append(label_id_dict[example.get(label_attribute)])

            # only add tag_ids during training
            if training and self.component_config.get(ENTITY_RECOGNITION):
                for tag_spec in self._entity_tag_specs:
                    tag_name_to_tag_ids[tag_spec.tag_name].append(
                        self._tag_ids_for_crf(example, tag_spec)
                    )

        X_sparse = np.array(X_sparse)
        X_dense = np.array(X_dense)
        Y_sparse = np.array(Y_sparse)
        Y_dense = np.array(Y_dense)
        label_ids = np.array(label_ids)
        tag_name_to_tag_ids = {
            tag_name: np.array(tag_ids)
            for tag_name, tag_ids in tag_name_to_tag_ids.items()
        }

        model_data = RasaModelData(label_key=self.label_key)
        model_data.add_features(TEXT_FEATURES, [X_sparse, X_dense])
        model_data.add_features(LABEL_FEATURES, [Y_sparse, Y_dense])
        if label_attribute and model_data.feature_not_exist(LABEL_FEATURES):
            # no label features are present, get default features from _label_data
            model_data.add_features(
                LABEL_FEATURES, self._use_default_label_features(label_ids)
            )

        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        model_data.add_features(LABEL_IDS, [np.expand_dims(label_ids, -1)])

        for tag_name, tag_ids in tag_name_to_tag_ids.items():
            model_data.add_features(f"{tag_name}_{TAG_IDS}", [tag_ids])

        model_data.add_lengths(TEXT_SEQ_LENGTH, TEXT_FEATURES)
        model_data.add_lengths(LABEL_SEQ_LENGTH, LABEL_FEATURES)

        return model_data

    def _tag_ids_for_crf(self, example: Message, tag_spec: EntityTagSpec) -> np.ndarray:
        """Create a np.array containing the tag ids of the given message."""
        if self.component_config[BILOU_FLAG]:
            _tags = bilou_utils.bilou_tags_to_ids(
                example, tag_spec.tags_to_ids, tag_spec.tag_name
            )
        else:
            _tags = []
            for token in example.get(TOKENS_NAMES[TEXT]):
                _tag = determine_token_labels(
                    token, example.get(ENTITIES), attribute_key=tag_spec.tag_name
                )
                _tags.append(tag_spec.tags_to_ids[_tag])

        # transpose to have seq_len x 1
        return np.array([_tags]).T

    # train helpers
    def preprocess_train_data(self, training_data: TrainingData) -> RasaModelData:
        """Prepares data for training.

        Performs sanity checks on training data, extracts encodings for labels.
        """

        if self.component_config[BILOU_FLAG]:
            bilou_utils.apply_bilou_schema(training_data)

        label_id_index_mapping = self._label_id_index_mapping(
            training_data, attribute=INTENT
        )

        if not label_id_index_mapping:
            # no labels are present to train
            return RasaModelData()

        self.index_label_id_mapping = self._invert_mapping(label_id_index_mapping)

        self._label_data = self._create_label_data(
            training_data, label_id_index_mapping, attribute=INTENT
        )

        self._entity_tag_specs = self._create_entity_tag_specs(training_data)

        label_attribute = (
            INTENT if self.component_config[INTENT_CLASSIFICATION] else None
        )

        model_data = self._create_model_data(
            training_data.training_examples,
            label_id_index_mapping,
            label_attribute=label_attribute,
        )

        self._check_input_dimension_consistency(model_data)

        return model_data

    @staticmethod
    def _check_enough_labels(model_data: RasaModelData) -> bool:
        return len(np.unique(model_data.get(LABEL_IDS))) >= 2

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Train the embedding intent classifier on a data set."""

        model_data = self.preprocess_train_data(training_data)
        if model_data.is_empty():
            logger.debug(
                f"Cannot train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the classifier."
            )
            return

        if self.component_config.get(INTENT_CLASSIFICATION):
            if not self._check_enough_labels(model_data):
                logger.error(
                    f"Cannot train '{self.__class__.__name__}'. "
                    f"Need at least 2 different intent classes. "
                    f"Skipping training of classifier."
                )
                return
        if self.component_config.get(ENTITY_RECOGNITION):
            self.check_correct_entity_annotations(training_data)

        # keep one example for persisting and loading
        self._data_example = model_data.first_data_example()

        self.model = self.model_class()(
            data_signature=model_data.get_signature(),
            label_data=self._label_data,
            entity_tag_specs=self._entity_tag_specs,
            config=self.component_config,
        )

        self.model.fit(
            model_data,
            self.component_config[EPOCHS],
            self.component_config[BATCH_SIZES],
            self.component_config[EVAL_NUM_EXAMPLES],
            self.component_config[EVAL_NUM_EPOCHS],
            self.component_config[BATCH_STRATEGY],
        )

    # process helpers
    def _predict(self, message: Message) -> Optional[Dict[Text, tf.Tensor]]:
        if self.model is None:
            logger.debug(
                f"There is no trained model for '{self.__class__.__name__}': The "
                f"component is either not trained or didn't receive enough training "
                f"data."
            )
            return None

        # create session data from message and convert it into a batch of 1
        model_data = self._create_model_data([message], training=False)

        return self.model.predict(model_data)

    def _predict_label(
        self, predict_out: Optional[Dict[Text, tf.Tensor]]
    ) -> Tuple[Dict[Text, Any], List[Dict[Text, Any]]]:
        """Predicts the intent of the provided message."""

        label = {"name": None, "confidence": 0.0}
        label_ranking = []

        if predict_out is None:
            return label, label_ranking

        message_sim = predict_out["i_scores"].numpy()

        message_sim = message_sim.flatten()  # sim is a matrix

        label_ids = message_sim.argsort()[::-1]

        if (
            self.component_config[LOSS_TYPE] == SOFTMAX
            and self.component_config[RANKING_LENGTH] > 0
        ):
            message_sim = train_utils.normalize(
                message_sim, self.component_config[RANKING_LENGTH]
            )

        message_sim[::-1].sort()
        message_sim = message_sim.tolist()

        # if X contains all zeros do not predict some label
        if label_ids.size > 0:
            label = {
                "name": self.index_label_id_mapping[label_ids[0]],
                "confidence": message_sim[0],
            }

            if (
                self.component_config[RANKING_LENGTH]
                and 0 < self.component_config[RANKING_LENGTH] < LABEL_RANKING_LENGTH
            ):
                output_length = self.component_config[RANKING_LENGTH]
            else:
                output_length = LABEL_RANKING_LENGTH

            ranking = list(zip(list(label_ids), message_sim))
            ranking = ranking[:output_length]
            label_ranking = [
                {"name": self.index_label_id_mapping[label_idx], "confidence": score}
                for label_idx, score in ranking
            ]

        return label, label_ranking

    def _predict_entities(
        self, predict_out: Optional[Dict[Text, tf.Tensor]], message: Message
    ) -> List[Dict]:
        if predict_out is None:
            return []

        predicted_tags = self._entity_label_to_tags(predict_out)

        entities = self.convert_predictions_into_entities(
            message.text, message.get(TOKENS_NAMES[TEXT], []), predicted_tags
        )

        entities = self.add_extractor_name(entities)
        entities = message.get(ENTITIES, []) + entities

        return entities

    def _entity_label_to_tags(
        self, predict_out: Dict[Text, Any]
    ) -> Dict[Text, List[Text]]:
        predicted_tags = {}

        for tag_spec in self._entity_tag_specs:
            predictions = predict_out[f"e_{tag_spec.tag_name}_ids"].numpy()
            tags = [tag_spec.ids_to_tags[p] for p in predictions[0]]

            if self.component_config[BILOU_FLAG]:
                tags = bilou_utils.ensure_consistent_bilou_tagging(tags)
                tags = bilou_utils.remove_bilou_prefixes(tags)

            predicted_tags[tag_spec.tag_name] = tags

        return predicted_tags

    def process(self, message: Message, **kwargs: Any) -> None:
        """Return the most likely label and its similarity to the input."""

        out = self._predict(message)

        if self.component_config[INTENT_CLASSIFICATION]:
            label, label_ranking = self._predict_label(out)

            message.set(INTENT, label, add_to_output=True)
            message.set("intent_ranking", label_ranking, add_to_output=True)

        if self.component_config[ENTITY_RECOGNITION]:
            entities = self._predict_entities(out, message)

            message.set(ENTITIES, entities, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

        if self.model is None:
            return {"file": None}

        model_dir = Path(model_dir)
        tf_model_file = model_dir / f"{file_name}.tf_model"

        io_utils.create_directory_for_file(tf_model_file)

        self.model.save(str(tf_model_file))

        io_utils.pickle_dump(
            model_dir / f"{file_name}.data_example.pkl", self._data_example
        )
        io_utils.pickle_dump(
            model_dir / f"{file_name}.label_data.pkl", self._label_data
        )
        io_utils.json_pickle(
            model_dir / f"{file_name}.index_label_id_mapping.pkl",
            self.index_label_id_mapping,
        )

        entity_tag_specs = (
            [tag_spec._asdict() for tag_spec in self._entity_tag_specs]
            if self._entity_tag_specs
            else []
        )
        io_utils.dump_obj_as_json_to_file(
            model_dir / f"{file_name}.entity_tag_specs.json", entity_tag_specs
        )

        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component: Optional["DIETClassifier"] = None,
        **kwargs: Any,
    ) -> "DIETClassifier":
        """Loads the trained model from the provided directory."""

        if not model_dir or not meta.get("file"):
            logger.debug(
                f"Failed to load model for '{cls.__name__}'. "
                f"Maybe you did not provide enough training data and no model was "
                f"trained or the path '{os.path.abspath(model_dir)}' doesn't exist?"
            )
            return cls(component_config=meta)

        (
            index_label_id_mapping,
            entity_tag_specs,
            label_data,
            meta,
            data_example,
        ) = cls._load_from_files(meta, model_dir)

        meta = train_utils.update_similarity_type(meta)

        model = cls._load_model(
            entity_tag_specs, label_data, meta, data_example, model_dir
        )

        return cls(
            component_config=meta,
            index_label_id_mapping=index_label_id_mapping,
            entity_tag_specs=entity_tag_specs,
            model=model,
        )

    @classmethod
    def _load_from_files(cls, meta: Dict[Text, Any], model_dir: Text):
        file_name = meta.get("file")

        model_dir = Path(model_dir)

        data_example = io_utils.pickle_load(model_dir / f"{file_name}.data_example.pkl")
        label_data = io_utils.pickle_load(model_dir / f"{file_name}.label_data.pkl")
        index_label_id_mapping = io_utils.json_unpickle(
            model_dir / f"{file_name}.index_label_id_mapping.pkl"
        )
        entity_tag_specs = io_utils.read_json_file(
            model_dir / f"{file_name}.entity_tag_specs.json"
        )
        entity_tag_specs = [
            EntityTagSpec(
                tag_name=tag_spec["tag_name"],
                ids_to_tags={
                    int(key): value for key, value in tag_spec["ids_to_tags"].items()
                },
                tags_to_ids={
                    key: int(value) for key, value in tag_spec["tags_to_ids"].items()
                },
                num_tags=tag_spec["num_tags"],
            )
            for tag_spec in entity_tag_specs
        ]

        # jsonpickle converts dictionary keys to strings
        index_label_id_mapping = {
            int(key): value for key, value in index_label_id_mapping.items()
        }

        return (
            index_label_id_mapping,
            entity_tag_specs,
            label_data,
            meta,
            data_example,
        )

    @classmethod
    def _load_model(
        cls,
        entity_tag_specs: List[EntityTagSpec],
        label_data: RasaModelData,
        meta: Dict[Text, Any],
        data_example: Dict[Text, List[np.ndarray]],
        model_dir: Text,
    ):
        file_name = meta.get("file")
        tf_model_file = os.path.join(model_dir, file_name + ".tf_model")

        label_key = LABEL_IDS if meta[INTENT_CLASSIFICATION] else None
        model_data_example = RasaModelData(label_key=label_key, data=data_example)

        model = cls.model_class().load(
            tf_model_file,
            model_data_example,
            data_signature=model_data_example.get_signature(),
            label_data=label_data,
            entity_tag_specs=entity_tag_specs,
            config=meta,
        )

        # build the graph for prediction
        predict_data_example = RasaModelData(
            label_key=label_key,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                if TEXT in feature_name
            },
        )

        model.build_for_predict(predict_data_example)

        return model


# accessing _tf_layers with any key results in key-error, disable it
# pytype: disable=key-error


class DIET(RasaModel):
    def __init__(
        self,
        data_signature: Dict[Text, List[FeatureSignature]],
        label_data: RasaModelData,
        entity_tag_specs: Optional[List[EntityTagSpec]],
        config: Dict[Text, Any],
    ) -> None:
        super().__init__(
            name="DIET",
            random_seed=config[RANDOM_SEED],
            tensorboard_log_dir=config[TENSORBOARD_LOG_DIR],
            tensorboard_log_level=config[TENSORBOARD_LOG_LEVEL],
        )

        self.config = config

        self.data_signature = data_signature
        self._check_data()

        self.predict_data_signature = {
            feature_name: features
            for feature_name, features in data_signature.items()
            if TEXT in feature_name
        }

        label_batch = label_data.prepare_batch()
        self.tf_label_data = self.batch_to_model_data_format(
            label_batch, label_data.get_signature()
        )

        self._entity_tag_specs = self._ordered_tag_specs(entity_tag_specs)

        # tf objects
        self._tf_layers: Dict[Text : tf.keras.layers.Layer] = {}
        self._prepare_layers()

        # tf training
        self._set_optimizer(tf.keras.optimizers.Adam(config[LEARNING_RATE]))
        self._create_metrics()
        self._update_metrics_to_log()

        self.all_labels_embed = None  # needed for efficient prediction

    @staticmethod
    def _ordered_tag_specs(
        entity_tag_specs: Optional[List[EntityTagSpec]],
    ) -> List[EntityTagSpec]:
        """Ensure that order of entity tag specs matches CRF layer order."""
        if entity_tag_specs is None:
            return []

        crf_order = [
            ENTITY_ATTRIBUTE_TYPE,
            ENTITY_ATTRIBUTE_ROLE,
            ENTITY_ATTRIBUTE_GROUP,
        ]

        ordered_tag_spec = []

        for tag_name in crf_order:
            for tag_spec in entity_tag_specs:
                if tag_name == tag_spec.tag_name:
                    ordered_tag_spec.append(tag_spec)

        return ordered_tag_spec

    def _check_data(self) -> None:
        if TEXT_FEATURES not in self.data_signature:
            raise InvalidConfigError(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if self.config[INTENT_CLASSIFICATION]:
            if LABEL_FEATURES not in self.data_signature:
                raise InvalidConfigError(
                    f"No label features specified. "
                    f"Cannot train '{self.__class__.__name__}' model."
                )
            if (
                self.config[SHARE_HIDDEN_LAYERS]
                and self.data_signature[TEXT_FEATURES]
                != self.data_signature[LABEL_FEATURES]
            ):
                raise ValueError(
                    "If hidden layer weights are shared, data signatures "
                    "for text_features and label_features must coincide."
                )

        if (
            self.config[ENTITY_RECOGNITION]
            and f"{ENTITY_ATTRIBUTE_TYPE}_{TAG_IDS}" not in self.data_signature
        ):
            common_utils.raise_warning(
                f"You specified '{self.__class__.__name__}' to train entities, but "
                f"no entities are present in the training data. Skip training of "
                f"entities."
            )
            self.config[ENTITY_RECOGNITION] = False

    def _create_metrics(self) -> None:
        # self.metrics will have the same order as they are created
        # so create loss metrics first to output losses first
        self.mask_loss = tf.keras.metrics.Mean(name="m_loss")
        self.intent_loss = tf.keras.metrics.Mean(name="i_loss")
        self.entity_loss = tf.keras.metrics.Mean(name="entity_loss")
        self.entity_group_loss = tf.keras.metrics.Mean(name="group_loss")
        self.entity_role_loss = tf.keras.metrics.Mean(name="role_loss")
        # create accuracy metrics second to output accuracies second
        self.mask_acc = tf.keras.metrics.Mean(name="m_acc")
        self.response_acc = tf.keras.metrics.Mean(name="i_acc")
        self.entity_f1 = tf.keras.metrics.Mean(name="entity_f1")
        self.entity_group_f1 = tf.keras.metrics.Mean(name="group_f1")
        self.entity_role_f1 = tf.keras.metrics.Mean(name="role_f1")

    def _update_metrics_to_log(self) -> None:
        if self.config[MASKED_LM]:
            self.metrics_to_log += ["m_loss", "m_acc"]
        if self.config[INTENT_CLASSIFICATION]:
            self.metrics_to_log += ["i_loss", "i_acc"]
        if self.config[ENTITY_RECOGNITION]:
            for tag_spec in self._entity_tag_specs:
                if tag_spec.num_tags != 0:
                    name = tag_spec.tag_name
                    self.metrics_to_log += [f"{name}_loss", f"{name}_f1"]

    def _prepare_layers(self) -> None:
        self.text_name = TEXT
        self._prepare_sequence_layers(self.text_name)
        if self.config[MASKED_LM]:
            self._prepare_mask_lm_layers(self.text_name)
        if self.config[INTENT_CLASSIFICATION]:
            self.label_name = TEXT if self.config[SHARE_HIDDEN_LAYERS] else LABEL
            self._prepare_input_layers(self.label_name)
            self._prepare_label_classification_layers()
        if self.config[ENTITY_RECOGNITION]:
            self._prepare_entity_recognition_layers()

    def _prepare_sparse_dense_layers(
        self,
        feature_signatures: List[FeatureSignature],
        name: Text,
        reg_lambda: float,
        dense_dim: int,
    ) -> None:
        sparse = False
        dense = False
        for is_sparse, shape in feature_signatures:
            if is_sparse:
                sparse = True
            else:
                dense = True
                # if dense features are present
                # use the feature dimension of the dense features
                dense_dim = shape[-1]

        if sparse:
            self._tf_layers[f"sparse_to_dense.{name}"] = layers.DenseForSparse(
                units=dense_dim, reg_lambda=reg_lambda, name=name
            )
            if not dense:
                # create dense labels for the input to use in negative sampling
                self._tf_layers[f"sparse_to_dense_ids.{name}"] = layers.DenseForSparse(
                    units=2, trainable=False, name=f"sparse_to_dense_ids.{name}"
                )

    def _prepare_input_layers(self, name: Text) -> None:
        self._tf_layers[f"sparse_input_dropout.{name}"] = layers.SparseDropout(
            rate=self.config[DROP_RATE]
        )
        self._tf_layers[f"dense_input_dropout.{name}"] = tf.keras.layers.Dropout(
            rate=self.config[DROP_RATE]
        )
        self._prepare_sparse_dense_layers(
            self.data_signature[f"{name}_features"],
            name,
            self.config[REGULARIZATION_CONSTANT],
            self.config[DENSE_DIMENSION][name],
        )
        self._tf_layers[f"ffnn.{name}"] = layers.Ffnn(
            self.config[HIDDEN_LAYERS_SIZES][name],
            self.config[DROP_RATE],
            self.config[REGULARIZATION_CONSTANT],
            self.config[WEIGHT_SPARSITY],
            name,
        )

    def _prepare_embed_layers(self, name: Text) -> None:
        self._tf_layers[f"embed.{name}"] = layers.Embed(
            self.config[EMBEDDING_DIMENSION],
            self.config[REGULARIZATION_CONSTANT],
            name,
            self.config[SIMILARITY_TYPE],
        )

    def _prepare_dot_product_loss(self, name: Text, scale_loss: bool) -> None:
        self._tf_layers[f"loss.{name}"] = layers.DotProductLoss(
            self.config[NUM_NEG],
            self.config[LOSS_TYPE],
            self.config[MAX_POS_SIM],
            self.config[MAX_NEG_SIM],
            self.config[USE_MAX_NEG_SIM],
            self.config[NEGATIVE_MARGIN_SCALE],
            scale_loss,
            # set to 1 to get deterministic behaviour
            parallel_iterations=1 if self.random_seed is not None else 1000,
        )

    def _prepare_sequence_layers(self, name: Text) -> None:
        self._prepare_input_layers(name)

        if self.config[NUM_TRANSFORMER_LAYERS] > 0:
            self._tf_layers[f"{name}_transformer"] = TransformerEncoder(
                self.config[NUM_TRANSFORMER_LAYERS],
                self.config[TRANSFORMER_SIZE],
                self.config[NUM_HEADS],
                self.config[TRANSFORMER_SIZE] * 4,
                self.config[REGULARIZATION_CONSTANT],
                dropout_rate=self.config[DROP_RATE],
                attention_dropout_rate=self.config[DROP_RATE_ATTENTION],
                sparsity=self.config[WEIGHT_SPARSITY],
                unidirectional=self.config[UNIDIRECTIONAL_ENCODER],
                use_key_relative_position=self.config[KEY_RELATIVE_ATTENTION],
                use_value_relative_position=self.config[VALUE_RELATIVE_ATTENTION],
                max_relative_position=self.config[MAX_RELATIVE_POSITION],
                name=f"{name}_encoder",
            )
        else:
            # create lambda so that it can be used later without the check
            self._tf_layers[f"{name}_transformer"] = lambda x, mask, training: x

    def _prepare_mask_lm_layers(self, name: Text) -> None:
        self._tf_layers[f"{name}_input_mask"] = layers.InputMask()

        self._prepare_embed_layers(f"{name}_lm_mask")
        self._prepare_embed_layers(f"{name}_golden_token")

        # mask loss is additional loss
        # set scaling to False, so that it doesn't overpower other losses
        self._prepare_dot_product_loss(f"{name}_mask", scale_loss=False)

    def _prepare_label_classification_layers(self) -> None:
        self._prepare_embed_layers(TEXT)
        self._prepare_embed_layers(LABEL)

        self._prepare_dot_product_loss(LABEL, self.config[SCALE_LOSS])

    def _prepare_entity_recognition_layers(self) -> None:
        for tag_spec in self._entity_tag_specs:
            name = tag_spec.tag_name
            num_tags = tag_spec.num_tags
            self._tf_layers[f"embed.{name}.logits"] = layers.Embed(
                num_tags, self.config[REGULARIZATION_CONSTANT], f"logits.{name}"
            )
            self._tf_layers[f"crf.{name}"] = layers.CRF(
                num_tags, self.config[REGULARIZATION_CONSTANT], self.config[SCALE_LOSS]
            )
            self._tf_layers[f"embed.{name}.tags"] = layers.Embed(
                self.config[EMBEDDING_DIMENSION],
                self.config[REGULARIZATION_CONSTANT],
                f"tags.{name}",
            )

    def _combine_sparse_dense_features(
        self,
        features: List[Union[np.ndarray, tf.Tensor, tf.SparseTensor]],
        mask: tf.Tensor,
        name: Text,
        sparse_dropout: bool = False,
        dense_dropout: bool = False,
    ) -> tf.Tensor:

        dense_features = []

        for f in features:
            if isinstance(f, tf.SparseTensor):
                if sparse_dropout:
                    _f = self._tf_layers[f"sparse_input_dropout.{name}"](
                        f, self._training
                    )
                else:
                    _f = f

                dense_f = self._tf_layers[f"sparse_to_dense.{name}"](_f)

                if dense_dropout:
                    dense_f = self._tf_layers[f"dense_input_dropout.{name}"](
                        dense_f, self._training
                    )

                dense_features.append(dense_f)
            else:
                dense_features.append(f)

        return tf.concat(dense_features, axis=-1) * mask

    def _features_as_seq_ids(
        self, features: List[Union[np.ndarray, tf.Tensor, tf.SparseTensor]], name: Text
    ) -> Optional[tf.Tensor]:
        """Creates dense labels for negative sampling."""

        # if there are dense features - we can use them
        for f in features:
            if not isinstance(f, tf.SparseTensor):
                return tf.stop_gradient(f)

        # use additional sparse to dense layer
        for f in features:
            if isinstance(f, tf.SparseTensor):
                return tf.stop_gradient(
                    self._tf_layers[f"sparse_to_dense_ids.{name}"](f)
                )

        return None

    def _create_bow(
        self,
        features: List[Union[tf.Tensor, tf.SparseTensor]],
        mask: tf.Tensor,
        name: Text,
        sparse_dropout: bool = False,
        dense_dropout: bool = False,
    ) -> tf.Tensor:

        x = self._combine_sparse_dense_features(
            features, mask, name, sparse_dropout, dense_dropout
        )
        x = tf.reduce_sum(x, axis=1)  # convert to bag-of-words
        return self._tf_layers[f"ffnn.{name}"](x, self._training)

    def _create_sequence(
        self,
        features: List[Union[tf.Tensor, tf.SparseTensor]],
        mask: tf.Tensor,
        name: Text,
        sparse_dropout: bool = False,
        dense_dropout: bool = False,
        masked_lm_loss: bool = False,
        sequence_ids: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor, Optional[tf.Tensor], Optional[tf.Tensor]]:
        if sequence_ids:
            seq_ids = self._features_as_seq_ids(features, name)
        else:
            seq_ids = None

        inputs = self._combine_sparse_dense_features(
            features, mask, name, sparse_dropout, dense_dropout
        )

        inputs = self._tf_layers[f"ffnn.{name}"](inputs, self._training)

        if masked_lm_loss:
            transformer_inputs, lm_mask_bool = self._tf_layers[f"{name}_input_mask"](
                inputs, mask, self._training
            )
        else:
            transformer_inputs = inputs
            lm_mask_bool = None

        outputs = self._tf_layers[f"{name}_transformer"](
            transformer_inputs, 1 - mask, self._training
        )

        if self.config[NUM_TRANSFORMER_LAYERS] > 0:
            # apply activation
            outputs = tfa.activations.gelu(outputs)

        return outputs, inputs, seq_ids, lm_mask_bool

    def _create_all_labels(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_label_ids = self.tf_label_data[LABEL_IDS][0]

        label_lengths = self._get_sequence_lengths(
            self.tf_label_data[LABEL_SEQ_LENGTH][0]
        )
        mask_label = self._compute_mask(label_lengths)

        x = self._create_bow(
            self.tf_label_data[LABEL_FEATURES], mask_label, self.label_name
        )
        all_labels_embed = self._tf_layers[f"embed.{LABEL}"](x)

        return all_label_ids, all_labels_embed

    @staticmethod
    def _last_token(x: tf.Tensor, sequence_lengths: tf.Tensor) -> tf.Tensor:
        last_sequence_index = tf.maximum(0, sequence_lengths - 1)
        batch_index = tf.range(tf.shape(last_sequence_index)[0])

        indices = tf.stack([batch_index, last_sequence_index], axis=1)
        return tf.gather_nd(x, indices)

    def _mask_loss(
        self,
        outputs: tf.Tensor,
        inputs: tf.Tensor,
        seq_ids: tf.Tensor,
        lm_mask_bool: tf.Tensor,
        name: Text,
    ) -> tf.Tensor:
        # make sure there is at least one element in the mask
        lm_mask_bool = tf.cond(
            tf.reduce_any(lm_mask_bool),
            lambda: lm_mask_bool,
            lambda: tf.scatter_nd([[0, 0, 0]], [True], tf.shape(lm_mask_bool)),
        )

        lm_mask_bool = tf.squeeze(lm_mask_bool, -1)
        # pick elements that were masked
        outputs = tf.boolean_mask(outputs, lm_mask_bool)
        inputs = tf.boolean_mask(inputs, lm_mask_bool)
        ids = tf.boolean_mask(seq_ids, lm_mask_bool)

        outputs_embed = self._tf_layers[f"embed.{name}_lm_mask"](outputs)
        inputs_embed = self._tf_layers[f"embed.{name}_golden_token"](inputs)

        return self._tf_layers[f"loss.{name}_mask"](
            outputs_embed, inputs_embed, ids, inputs_embed, ids
        )

    def _calculate_label_loss(
        self, a: tf.Tensor, b: tf.Tensor, label_ids: tf.Tensor
    ) -> tf.Tensor:
        all_label_ids, all_labels_embed = self._create_all_labels()

        a_embed = self._tf_layers[f"embed.{TEXT}"](a)
        b_embed = self._tf_layers[f"embed.{LABEL}"](b)

        return self._tf_layers[f"loss.{LABEL}"](
            a_embed, b_embed, label_ids, all_labels_embed, all_label_ids
        )

    def _calculate_entity_loss(
        self,
        inputs: tf.Tensor,
        tag_ids: tf.Tensor,
        mask: tf.Tensor,
        sequence_lengths: tf.Tensor,
        tag_name: Text,
        entity_tags: Optional[tf.Tensor] = None,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        tag_ids = tf.cast(tag_ids[:, :, 0], tf.int32)

        if entity_tags is not None:
            _tags = self._tf_layers[f"embed.{tag_name}.tags"](entity_tags)
            inputs = tf.concat([inputs, _tags], axis=-1)

        logits = self._tf_layers[f"embed.{tag_name}.logits"](inputs)

        # should call first to build weights
        pred_ids = self._tf_layers[f"crf.{tag_name}"](logits, sequence_lengths)
        # pytype cannot infer that 'self._tf_layers["crf"]' has the method '.loss'
        # pytype: disable=attribute-error
        loss = self._tf_layers[f"crf.{tag_name}"].loss(
            logits, tag_ids, sequence_lengths
        )
        f1 = self._tf_layers[f"crf.{tag_name}"].f1_score(tag_ids, pred_ids, mask)
        # pytype: enable=attribute-error

        return loss, f1, logits

    @staticmethod
    def _compute_mask(sequence_lengths: tf.Tensor) -> tf.Tensor:
        mask = tf.sequence_mask(sequence_lengths, dtype=tf.float32)
        # explicitly add last dimension to mask
        # to track correctly dynamic sequences
        return tf.expand_dims(mask, -1)

    @staticmethod
    def _get_sequence_lengths(sequence_lengths: tf.Tensor) -> tf.Tensor:
        return tf.cast(sequence_lengths, dtype=tf.int32)

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)

        sequence_lengths = self._get_sequence_lengths(tf_batch_data[TEXT_SEQ_LENGTH][0])
        mask_text = self._compute_mask(sequence_lengths)

        (
            text_transformed,
            text_in,
            text_seq_ids,
            lm_mask_bool_text,
        ) = self._create_sequence(
            tf_batch_data[TEXT_FEATURES],
            mask_text,
            self.text_name,
            sparse_dropout=self.config[SPARSE_INPUT_DROPOUT],
            dense_dropout=self.config[DENSE_INPUT_DROPOUT],
            masked_lm_loss=self.config[MASKED_LM],
            sequence_ids=True,
        )

        losses = []

        if self.config[MASKED_LM]:
            loss, acc = self._mask_loss(
                text_transformed, text_in, text_seq_ids, lm_mask_bool_text, TEXT
            )
            self.mask_loss.update_state(loss)
            self.mask_acc.update_state(acc)
            losses.append(loss)

        if self.config[INTENT_CLASSIFICATION]:
            loss = self._batch_loss_intent(
                sequence_lengths, text_transformed, tf_batch_data
            )
            losses.append(loss)

        if self.config[ENTITY_RECOGNITION]:
            losses += self._batch_loss_entities(
                mask_text, sequence_lengths, text_transformed, tf_batch_data
            )

        return tf.math.add_n(losses)

    def _batch_loss_intent(
        self,
        sequence_lengths: tf.Tensor,
        text_transformed: tf.Tensor,
        tf_batch_data: Dict[Text, List[tf.Tensor]],
    ) -> tf.Tensor:
        # get _cls_ vector for intent classification
        cls = self._last_token(text_transformed, sequence_lengths)

        label_lengths = self._get_sequence_lengths(tf_batch_data[LABEL_SEQ_LENGTH][0])
        mask_label = self._compute_mask(label_lengths)

        label_ids = tf_batch_data[LABEL_IDS][0]
        label = self._create_bow(
            tf_batch_data[LABEL_FEATURES], mask_label, self.label_name
        )

        loss, acc = self._calculate_label_loss(cls, label, label_ids)

        self.intent_loss.update_state(loss)
        self.response_acc.update_state(acc)

        return loss

    def _batch_loss_entities(
        self,
        mask_text: tf.Tensor,
        sequence_lengths: tf.Tensor,
        text_transformed: tf.Tensor,
        tf_batch_data: Dict[Text, List[tf.Tensor]],
    ) -> List[tf.Tensor]:
        losses = []

        sequence_lengths -= 1  # remove cls token

        entity_tags = None

        for tag_spec in self._entity_tag_specs:
            if tag_spec.num_tags == 0:
                continue

            tag_ids = tf_batch_data[f"{tag_spec.tag_name}_{TAG_IDS}"][0]

            loss, f1, _logits = self._calculate_entity_loss(
                text_transformed,
                tag_ids,
                mask_text,
                sequence_lengths,
                tag_spec.tag_name,
                entity_tags,
            )

            if tag_spec.tag_name == ENTITY_ATTRIBUTE_TYPE:
                # use the entity tags as additional input for the role
                # and group CRF
                entity_tags = tf.one_hot(
                    tf.cast(tag_ids[:, :, 0], tf.int32), depth=tag_spec.num_tags
                )

            self._update_entity_metrics(loss, f1, tag_spec.tag_name)

            losses.append(loss)

        return losses

    def _update_entity_metrics(self, loss: tf.Tensor, f1: tf.Tensor, tag_name: Text):
        if tag_name == ENTITY_ATTRIBUTE_TYPE:
            self.entity_loss.update_state(loss)
            self.entity_f1.update_state(f1)
        elif tag_name == ENTITY_ATTRIBUTE_GROUP:
            self.entity_group_loss.update_state(loss)
            self.entity_group_f1.update_state(f1)
        elif tag_name == ENTITY_ATTRIBUTE_ROLE:
            self.entity_role_loss.update_state(loss)
            self.entity_role_f1.update_state(f1)

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )

        sequence_lengths = self._get_sequence_lengths(tf_batch_data[TEXT_SEQ_LENGTH][0])
        mask_text = self._compute_mask(sequence_lengths)

        text_transformed, _, _, _ = self._create_sequence(
            tf_batch_data[TEXT_FEATURES], mask_text, self.text_name
        )

        predictions: Dict[Text, tf.Tensor] = {}

        if self.config[INTENT_CLASSIFICATION]:
            predictions.update(
                self._batch_predict_intents(sequence_lengths, text_transformed)
            )

        if self.config[ENTITY_RECOGNITION]:
            predictions.update(
                self._batch_predict_entities(sequence_lengths, text_transformed)
            )

        return predictions

    def _batch_predict_entities(
        self, sequence_lengths: tf.Tensor, text_transformed: tf.Tensor
    ) -> Dict[Text, tf.Tensor]:
        predictions: Dict[Text, tf.Tensor] = {}

        entity_tags = None

        for tag_spec in self._entity_tag_specs:
            # skip crf layer if it was not trained
            if tag_spec.num_tags == 0:
                continue

            name = tag_spec.tag_name
            _input = text_transformed

            if entity_tags is not None:
                _tags = self._tf_layers[f"embed.{name}.tags"](entity_tags)
                _input = tf.concat([_input, _tags], axis=-1)

            _logits = self._tf_layers[f"embed.{name}.logits"](_input)
            pred_ids = self._tf_layers[f"crf.{name}"](_logits, sequence_lengths - 1)

            predictions[f"e_{name}_ids"] = pred_ids

            if name == ENTITY_ATTRIBUTE_TYPE:
                # use the entity tags as additional input for the role
                # and group CRF
                entity_tags = tf.one_hot(
                    tf.cast(pred_ids, tf.int32), depth=tag_spec.num_tags
                )

        return predictions

    def _batch_predict_intents(
        self, sequence_lengths: tf.Tensor, text_transformed: tf.Tensor
    ) -> Dict[Text, tf.Tensor]:

        if self.all_labels_embed is None:
            _, self.all_labels_embed = self._create_all_labels()

        # get _cls_ vector for intent classification
        cls = self._last_token(text_transformed, sequence_lengths)
        cls_embed = self._tf_layers[f"embed.{TEXT}"](cls)

        # pytype cannot infer that 'self._tf_layers[f"loss.{LABEL}"]' has methods
        # like '.sim' or '.confidence_from_sim'
        # pytype: disable=attribute-error
        sim_all = self._tf_layers[f"loss.{LABEL}"].sim(
            cls_embed[:, tf.newaxis, :], self.all_labels_embed[tf.newaxis, :, :]
        )
        scores = self._tf_layers[f"loss.{LABEL}"].confidence_from_sim(
            sim_all, self.config[SIMILARITY_TYPE]
        )
        # pytype: enable=attribute-error

        return {"i_scores": scores}


# pytype: enable=key-error
