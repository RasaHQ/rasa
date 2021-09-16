import copy
import logging
from pathlib import Path

import numpy as np
import os
import tensorflow as tf

from typing import Any, Dict, List, Optional, Text, Tuple, Union, Type

import rasa.shared.utils.io
import rasa.utils.io as io_utils
import rasa.nlu.utils.bilou_utils as bilou_utils
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.components import Component
from rasa.nlu.classifiers.classifier import IntentClassifier
from rasa.nlu.extractors.extractor import EntityExtractor, EntityTagSpec
from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.utils import train_utils
from rasa.utils.tensorflow import rasa_layers
from rasa.utils.tensorflow.models import RasaModel, TransformerRasaModel
from rasa.utils.tensorflow import model_data_utils
from rasa.utils.tensorflow.model_data import (
    RasaModelData,
    FeatureSignature,
    FeatureArray,
)
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.nlu.constants import (
    FEATURE_TYPE_SEQUENCE,
    TEXT,
    INTENT,
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_ATTRIBUTE_GROUP,
    ENTITY_ATTRIBUTE_ROLE,
    NO_ENTITY_TAG,
    SPLIT_ENTITIES_BY_COMMA,
    FEATURE_TYPE_SENTENCE,
)
from rasa.nlu.config import RasaNLUModelConfig
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.model import Metadata
from rasa.utils.tensorflow.exceptions import RasaModelConfigException
from rasa.utils.tensorflow.constants import (
    LABEL,
    IDS,
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
    CONNECTION_DENSITY,
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
    AUTO,
    BALANCED,
    CROSS_ENTROPY,
    TENSORBOARD_LOG_LEVEL,
    CONCAT_DIMENSION,
    FEATURIZERS,
    CHECKPOINT_MODEL,
    SEQUENCE,
    SENTENCE,
    SEQUENCE_LENGTH,
    DENSE_DIMENSION,
    MASK,
    CONSTRAIN_SIMILARITIES,
    MODEL_CONFIDENCE,
    SOFTMAX,
)

logger = logging.getLogger(__name__)


LABEL_KEY = LABEL
LABEL_SUB_KEY = IDS

POSSIBLE_TAGS = [ENTITY_ATTRIBUTE_TYPE, ENTITY_ATTRIBUTE_ROLE, ENTITY_ATTRIBUTE_GROUP]


class DIETClassifier(IntentClassifier, EntityExtractor):
    """A multi-task model for intent classification and entity extraction.

    DIET is Dual Intent and Entity Transformer.
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
        # The number of hidden layers is equal to the length of the corresponding list.
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
        # Dense dimension to use for sparse features.
        DENSE_DIMENSION: {TEXT: 128, LABEL: 20},
        # Default dimension to use for concatenating sequence and sentence features.
        CONCAT_DIMENSION: {TEXT: 128, LABEL: 20},
        # The number of incorrect labels. The algorithm will minimize
        # their similarity to the user input during training.
        NUM_NEG: 20,
        # Type of similarity measure to use, either 'auto' or 'cosine' or 'inner'.
        SIMILARITY_TYPE: AUTO,
        # The type of the loss function, either 'cross_entropy' or 'margin'.
        LOSS_TYPE: CROSS_ENTROPY,
        # Number of top intents to normalize scores for. Applicable with
        # loss type 'cross_entropy' and 'softmax' confidences. Set to 0
        # to turn off normalization.
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
        # Fraction of trainable weights in internal layers.
        CONNECTION_DENSITY: 0.2,
        # If 'True' apply dropout to sparse input tensors
        SPARSE_INPUT_DROPOUT: True,
        # If 'True' apply dropout to dense input tensors
        DENSE_INPUT_DROPOUT: True,
        # ## Evaluation parameters
        # How often calculate validation accuracy.
        # Small values may hurt performance.
        EVAL_NUM_EPOCHS: 20,
        # How many examples to use for hold out validation set
        # Large values may hurt performance, e.g. model accuracy.
        # Set to 0 for no validation.
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
        # Valid values: 'epoch' and 'batch'
        TENSORBOARD_LOG_LEVEL: "epoch",
        # Perform model checkpointing
        CHECKPOINT_MODEL: False,
        # Specify what features to use as sequence and sentence features
        # By default all features in the pipeline are used.
        FEATURIZERS: [],
        # Split entities by comma, this makes sense e.g. for a list of ingredients
        # in a recipie, but it doesn't make sense for the parts of an address
        SPLIT_ENTITIES_BY_COMMA: True,
        # If 'True' applies sigmoid on all similarity terms and adds
        # it to the loss function to ensure that similarity values are
        # approximately bounded. Used inside softmax loss only.
        CONSTRAIN_SIMILARITIES: False,
        # Model confidence to be returned during inference. Possible values -
        # 'softmax' and 'linear_norm'.
        MODEL_CONFIDENCE: SOFTMAX,
    }

    def _check_and_autocorrect_component_config(self) -> None:

        # auto correct general settings in the component config via train_utils
        self.component_config = train_utils.check_deprecated_options(
            self.component_config
        )
        self.component_config = train_utils.update_confidence_type(
            self.component_config
        )
        self.component_config = train_utils.update_deprecated_loss_type(
            self.component_config
        )
        self.component_config = train_utils.update_deprecated_sparsity_to_density(
            self.component_config
        )
        self.component_config = train_utils.update_similarity_type(
            self.component_config
        )

        # validation of general settings
        # Note: This needs to happen before next check because the
        # next check would set CHECKPOINT_MODEL to False and hence we'd end up
        # with a silent fail of checkpointing.
        train_utils.validate_configuration_settings(self.component_config)

        # convert `EVAL_NUM_EPOCHS == -1` to actual number
        self.component_config = train_utils.update_evaluation_parameters(
            self.component_config
        )

        # sanity checks for architecture
        self._check_has_a_goal()
        self._check_masked_lm()
        self._check_share_hidden_layers_sizes()

    def _check_has_a_goal(self) -> None:
        if (
            not self.component_config[INTENT_CLASSIFICATION]
            and not self.component_config[ENTITY_RECOGNITION]
        ):
            raise RasaModelConfigException(
                f"Model is neither asked to perform entity recognition nor "
                f"to predict any label. Switch on {ENTITY_RECOGNITION} or "
                f"{INTENT_CLASSIFICATION} in your component config."
            )

    def _check_masked_lm(self) -> None:
        if (
            self.component_config[MASKED_LM]
            and self.component_config[NUM_TRANSFORMER_LAYERS] == 0
        ):
            raise RasaModelConfigException(
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
                raise RasaModelConfigException(
                    f"If hidden layer weights are shared, "
                    f"{HIDDEN_LAYERS_SIZES} must coincide."
                )

    # package safety checks
    @classmethod
    def required_packages(cls) -> List[Text]:
        """Returns the required python packages."""
        return ["tensorflow"]

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        index_label_id_mapping: Optional[Dict[int, Text]] = None,
        entity_tag_specs: Optional[List[EntityTagSpec]] = None,
        model: Optional[RasaModel] = None,
        finetune_mode: bool = False,
        sparse_feature_sizes: Optional[Dict[Text, Dict[Text, List[int]]]] = None,
    ) -> None:
        """Declare instance variables with default values."""
        if component_config is not None and EPOCHS not in component_config:
            rasa.shared.utils.io.raise_warning(
                f"Please configure the number of '{EPOCHS}' in your configuration file."
                f" We will change the default value of '{EPOCHS}' in the future to 1. "
            )

        super().__init__(component_config)
        self._check_and_autocorrect_component_config()

        self.index_label_id_mapping = index_label_id_mapping or {}
        self._entity_tag_specs = entity_tag_specs
        self.model = model
        self.finetune_mode = finetune_mode
        self._sparse_feature_sizes = sparse_feature_sizes

        # derive more settings from component config and defeaults:
        self.split_entities_config = self.init_split_entities()
        self.label_attribute = (
            INTENT if self.component_config[INTENT_CLASSIFICATION] else None
        )
        self.tmp_checkpoint_dir = None
        if self.component_config[CHECKPOINT_MODEL]:
            self.tmp_checkpoint_dir = Path(rasa.utils.io.create_temporary_directory())

        # internal helper
        self._label_data: Optional[RasaModelData] = None
        self._data_example: Optional[Dict[Text, Dict[Text, List[FeatureArray]]]] = None

        if not self.model and self.finetune_mode:
            raise rasa.shared.exceptions.InvalidParameterException(
                f"{self.__class__.__name__} was instantiated "
                f"with `model=None` and `finetune_mode=True`. "
                f"This is not a valid combination as the component "
                f"needs an already instantiated and trained model "
                f"to continue training in finetune mode."
            )

    @property
    def label_key(self) -> Optional[Text]:
        """Return key if intent classification is activated."""
        return LABEL_KEY if self.component_config[INTENT_CLASSIFICATION] else None

    @property
    def label_sub_key(self) -> Optional[Text]:
        """Return sub key if intent classification is activated."""
        return LABEL_SUB_KEY if self.component_config[INTENT_CLASSIFICATION] else None

    @staticmethod
    def model_class() -> Type[RasaModel]:
        """Returns the `DIET` model class."""
        return DIET

    # training data helpers:
    def _create_label_id_to_index_mapping(
        self, messages: List[Message],
    ) -> Dict[Text, int]:
        """Create a label id dictionary from the intent examples in the given data."""
        distinct_label_ids = {
            example.get(self.label_attribute) for example in messages
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

    def _uses_sequence_features_for_input_text(self) -> bool:
        """Whether we need sequence features for the `TEXT` attribute.

        Note that, for prediction of some label attribute, the `DIETClassifier`
        can make use of sequence features even if the number of
        transformer layers is 0 because for that it creates a BOW representation
        by simply summing over sequence+sentence features.
        """
        return True

    def _needs_sentence_features_for_input_text(self) -> bool:
        """Whether we need sentence features for the `TEXT` attribute.

        For the `DIETClassifier`, we will need a sentence feature if and only if we
        are predicting a label because this sentence feature is used as the
        embedding of the "`__CLS__`" token.
        """
        return self.label_attribute is not None

    def _needs_sentence_features_for_labels(self) -> bool:
        """Whether we expect/require sentence level features for the label attribute.

        For the `DIETClassifier`, we don't because `DIET` uses a BOW representation for
        it's final prediction which is obtained by simply summing over
        sequence+sentence features of the labels.
        """
        return False

    def _check_input_dimension_consistency(self, model_data: RasaModelData) -> None:
        """Checks if features have same dimensionality if hidden layers are shared."""
        if self.component_config.get(SHARE_HIDDEN_LAYERS):
            num_text_sentence_features = model_data.number_of_units(TEXT, SENTENCE)
            num_label_sentence_features = model_data.number_of_units(LABEL, SENTENCE)
            num_text_sequence_features = model_data.number_of_units(TEXT, SEQUENCE)
            num_label_sequence_features = model_data.number_of_units(LABEL, SEQUENCE)

            if (0 < num_text_sentence_features != num_label_sentence_features > 0) or (
                0 < num_text_sequence_features != num_label_sequence_features > 0
            ):
                raise RasaModelConfigException(
                    "If embeddings are shared text features and label features "
                    "must coincide. Check the output dimensions of previous components."
                )

    @staticmethod
    def _compute_default_label_features(num_labels: int) -> List[FeatureArray]:
        """Computes one-hot representation for the labels."""
        logger.debug("No label features found. Computing default label features.")

        eye_matrix = np.eye(num_labels, dtype=np.float32)
        # add sequence dimension to one-hot labels
        return [
            FeatureArray(
                np.array([np.expand_dims(a, 0) for a in eye_matrix]),
                number_of_dimensions=3,
            )
        ]

    def _create_label_data(
        self,
        messages: List[Message],
        label_id_dict: Dict[Text, int],
        label_attribute: Text,
    ) -> RasaModelData:
        """Creates a rasa data model that represents the specified labels.

        First, this method extracts **one** training example for each label.
        If all of these training examples contain at least one feature type
        (`SENTENCE` / `SEQUENCE`), then this method just extracts these. Otherwise,
        it computes one-hot encodings which serve as sentence level features.

        Note that, if and only if sequence level features have been extracted, a
        `SEQUENCE_LENGTH` key will be added under the `LABEL` key.

        Moreover, `FeatureArrays` containing the corresponding label ids will be added
        with subkey `ID`.

        Args:
            messages: messages used for training
            label_id_dict: a dictionary mapping all relevant labels to some index
            label_attribute: the message attribute which corresponds to the label

        Returns:
            a RasaDataModel that contains, for each kind of feature (see above),
            list with one `FeatureArray` per label

        Raises:
            a `ValueError` if some label is not supported by any training example
        """
        label_indices, label_examples = self._collect_one_example_per_label(
            messages=messages,
            label_id_dict=label_id_dict,
            label_attribute=label_attribute,
        )

        sequence_features, sentence_features = None, None

        # Collect the features from the collected examples, if there are any features
        all_label_examples_contain_some_features = all(
            example.features_present(
                attribute=label_attribute,
                featurizers=self.component_config[FEATURIZERS],
            )
            for example in label_examples
        )
        if all_label_examples_contain_some_features:
            (
                sequence_features,
                sentence_features,
            ) = model_data_utils.combine_attribute_features_from_all_messages(
                messages=label_examples,
                attribute=label_attribute,
                featurizers=self.component_config[FEATURIZERS],
            )

        # If there are no features or if the sentence features are missing but
        # required, then load the default label (sentence) features.
        needs_sentence_features = self._needs_sentence_features_for_labels()
        if not sentence_features and (not sequence_features or needs_sentence_features):

            if sequence_features and needs_sentence_features:
                rasa.shared.utils.io.raise_warning(
                    f"Expected sentence level features but only received "
                    f"sequence level features for `{label_attribute}` attribute. "
                    f"Falling back to using default one-hot embedding vectors. "
                )
            sequence_features = None
            sentence_features = self._compute_default_label_features(
                num_labels=len(label_id_dict)
            )

        # Save them as a RasaModelData
        label_data = RasaModelData()
        label_data.add_features(LABEL, SEQUENCE, sequence_features)
        label_data.add_features(LABEL, SENTENCE, sentence_features)

        # Add label id features - and sequence length information (if sequence
        # features are present)
        label_data.add_features(
            LABEL_KEY,
            LABEL_SUB_KEY,
            [FeatureArray(np.expand_dims(label_indices, -1), number_of_dimensions=2)],
        )
        label_data.add_lengths(LABEL, SEQUENCE_LENGTH, LABEL, SEQUENCE)

        return label_data

    @staticmethod
    def _collect_one_example_per_label(
        messages: List[Message], label_id_dict: Dict[Text, int], label_attribute: Text,
    ) -> Tuple[np.ndarray, List[Message]]:
        """Collect one example message for each label.

        Args:
          messages: the messages from which examples need to be extracted.
          label_attribute: the attribute used as label.

        Returns:
          a sorted array containing the indices from the given `label_id_dict`
          exactly and a list of messages where the i-th message is an example for
          the label whose index is given by the i-th entry in the array

        Raises:
          a `InvalidConfigException` if the given messages do not contain at least
          one example per label
        """
        labelidx_message_tuples = []
        debug_unsupported_labels = set()
        for label_name, label_idx in label_id_dict.items():
            message = next(
                (
                    message
                    for message in messages
                    if message.get(label_attribute) == label_name
                ),
                None,
            )
            if message is None:
                debug_unsupported_labels.update([label_name])
            labelidx_message_tuples.append((label_idx, message))

        if len(debug_unsupported_labels):
            raise InvalidConfigException(
                "Expected at least one example for each label "
                "in the given training_data but could not find "
                f"any example(s) for {sorted(debug_unsupported_labels)}."
            )

        # Sort the list of tuples based on the label_idx
        labelidx_message_tuples = sorted(labelidx_message_tuples, key=lambda x: x[0])

        # Extract the (aligned sequences of) label indices and messages
        sorted_labelidx = np.array([idx for (idx, _) in labelidx_message_tuples])
        sorted_messages = [example for (_, example) in labelidx_message_tuples]
        return sorted_labelidx, sorted_messages

    def _use_default_label_features(self, label_ids: np.ndarray) -> List[FeatureArray]:
        """Grabs the pre-computed default features for the `LABEL` key.

        In case that no features had been present during training for
        the label key, these sentence level default features are the one-hot
        features that have been computed via `_compute_default_label_features`.

        Otherwise, this function will return some precomputed features
        that have been present for the `LABEL` key in training data during
        training.

        Args:
          label_ids: an int array representing label ids

        Returns:
          The `SENTENCE`-level features.
        """
        feature_arrays: List[FeatureArray] = self._label_data.get(LABEL, SENTENCE)
        all_label_features = feature_arrays[0]
        return [
            FeatureArray(
                np.array([all_label_features[label_id] for label_id in label_ids]),
                number_of_dimensions=all_label_features.number_of_dimensions,
            )
        ]

    def _create_model_data(
        self,
        messages: List[Message],
        training: bool = True,
        label_id_dict: Optional[Dict[Text, int]] = None,
    ) -> RasaModelData:
        """Creates training data to be fed to the model from the given messages.

        Args:
          messages: The messages from which we want to extract the training data.
          training: Whether we're in training mode and should add label data
          label_id_dict: A mapping from labels to label ids that is
            only required for label data creation (i.e. if `training` is set to `True`)

        Returns:
           training data that contains features for labels if and only if
           `training` had been set to `True`
        """
        # Create RasaModelData
        model_data = RasaModelData(
            label_key=self.label_key, label_sub_key=self.label_sub_key,
        )

        # Add features for text and labels/entities
        # (Note that label/entities will only be included if training is True)
        features, sparse_feature_sizes = self._collect_features(
            messages=messages, training=training
        )
        attribute_data, _ = model_data_utils.convert_to_data_format(
            features, consider_dialogue_dimension=False
        )
        model_data.add_data(attribute_data)

        # Add sequence length information (only if SEQUENCE sub_key exists here)
        model_data.add_lengths(TEXT, SEQUENCE_LENGTH, TEXT, SEQUENCE)

        # Add information on size of sparse features
        # NOTE: Current implementation doesn't yet account for updating sparse
        # feature sizes of label attributes. That's why we remove them.
        sparse_feature_sizes.pop(self.label_attribute, None)
        model_data.add_sparse_feature_sizes(sparse_feature_sizes)

        # Add special ID features for the labels and catch edge cases
        if training and self.label_attribute is not None:
            label_ids = self._add_label_id_features(model_data, messages, label_id_dict)
            # in case there are no features for labels at this point, use the
            # default label features (cf. `_create_label_data`)
            self._fallback_to_default_label_features_if_necessary(
                label_ids=label_ids, model_data=model_data
            )

        # Use generic LABEL_KEY instead of specific label_attribute key
        if training:
            # As label_attribute can have different values, e.g. INTENT or RESPONSE,
            # copy over the features to the LABEL key to make
            # it easier to access the label features inside the model itself.
            # Note that update_key doesn't update any keys that aren't there.
            for subkey in [SENTENCE, SEQUENCE_LENGTH, MASK, SEQUENCE]:
                model_data.update_key(self.label_attribute, subkey, LABEL_KEY, subkey)

            model_data.add_lengths(LABEL_KEY, SEQUENCE_LENGTH, LABEL_KEY, SEQUENCE)

        # Make sure all keys are in the same order during training and prediction
        # as we rely on the order of key and sub-key when constructing the actual
        # tensors from the model data
        model_data.sort()

        return model_data

    def _collect_features(
        self, messages: List[Message], training: bool = True,
    ) -> Tuple[List[Dict[Text, List[Features]]], Dict[Text, Dict[Text, List[int]]]]:
        """Collects all features from the given messages for `model_data` creation.

        Args:
          messages: The messages from which we want to extract features.
          training: Whether we're in training mode and hence should add label data.

        Returns:
          a tuple containing a list of feature dictionaries (i.e. mappings from
          attributes to the respective features) and a dictionary with the
          sparse feature sizes per key/sub-key pair
        """
        # (1) collect all features for the text input first
        (
            features,
            sparse_feature_sizes,
        ) = model_data_utils.featurize_training_examples(
            messages,
            [TEXT],
            entity_tag_specs=self._entity_tag_specs,
            featurizers=self.component_config[FEATURIZERS],
            bilou_tagging=self.component_config[BILOU_FLAG],
            type=(
                None
                if self._uses_sequence_features_for_input_text()
                else FEATURE_TYPE_SENTENCE
            ),
        )

        # (2) add feature for the labels and entities if in training mode
        if training:

            labels_to_consider = []
            if self.label_attribute is not None:
                labels_to_consider.append(self.label_attribute)
            if self.component_config[ENTITY_RECOGNITION] and self._entity_tag_specs:
                # Add entities as labels only during training and only if there was
                # training data added for entities with DIET configured to predict
                # entities.
                labels_to_consider.append(ENTITIES)

            (
                features_for_labels,
                sparse_feature_sizes_for_labels,
            ) = model_data_utils.featurize_training_examples(
                messages,
                attributes=labels_to_consider,
                entity_tag_specs=self._entity_tag_specs,
                featurizers=self.component_config[FEATURIZERS],
                bilou_tagging=self.component_config[BILOU_FLAG],
            )

            # add the label features to the text features
            for feats_txt, feats_lbl in zip(features, features_for_labels):
                feats_txt.update(feats_lbl)
            sparse_feature_sizes.update(sparse_feature_sizes_for_labels)

        return features, sparse_feature_sizes

    def _fallback_to_default_label_features_if_necessary(
        self, label_ids: List[int], model_data: RasaModelData,
    ) -> None:
        """Fills in default label features for the label attribute if necessary.

        Here, "required" means that no features are present for the label attribute
        in the given `model_data`. If there is no `label_attribute` set, then nothing
        will be changed.

        Observe that the logic here matches the logic used in `_create_label_data`
        but instead of preparing the default label features (as in
        `_create_label_data`), here we *use* these default label features.

        Args:
          label_ids: list of label ids (indices) that contains entries for just as
            many records as the `model_data`
          model_data: data which possibly does not contain sequence
            or sentence features for the label attribute
        """
        if self.label_attribute is None:
            return

        # If *no* label features have been found before, then we load the default
        # label features that should've been computed in `_create_label_data`.
        sentence_missing = model_data.does_feature_not_exist(
            self.label_attribute, SENTENCE
        )
        sequence_missing = model_data.does_feature_not_exist(
            self.label_attribute, SEQUENCE
        )
        needs_sentence = self._needs_sentence_features_for_labels()
        if sentence_missing and (needs_sentence or sequence_missing):
            if (not sequence_missing) and sentence_missing and needs_sentence:
                rasa.shared.utils.io.raise_warning(
                    f"Expected sentence level features but only received "
                    f"sequence level features for {self.label_attribute}. "
                    f"Falling back to using default one-hot embedding vectors. "
                )
            # no label features are present, get default features from _label_data
            model_data.add_features(
                LABEL_KEY,
                SENTENCE,
                self._use_default_label_features(np.array(label_ids)),
            )

    def _add_label_id_features(
        self,
        model_data: RasaModelData,
        messages: List[Message],
        label_id_dict: Dict[Text, int],
    ) -> List[int]:
        """Adds label id features to the given `model_data`.

        Only add those if there is a label attribute. Otherwise, nothing changes.

        Returns:
          list of the ids (indices) used to create the features
        """
        if self.label_attribute is None:
            return []

        if label_id_dict is None:
            raise ValueError("Expected some label id mapping.")
        if self._label_data is None:
            raise ValueError(
                "Expected label data of type `RasaModelData` but `self._label_data` "
                " is `None`."
            )

        label_ids = []
        for example in messages:
            if example.get(self.label_attribute):
                label_ids.append(label_id_dict[example.get(self.label_attribute, "")])

        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        model_data.add_features(
            LABEL_KEY,
            LABEL_SUB_KEY,
            [FeatureArray(np.expand_dims(label_ids, -1), number_of_dimensions=2)],
        )
        return label_ids

    # train helpers
    def preprocess_train_data(self, training_data: TrainingData) -> RasaModelData:
        """Prepares data for training.

        Performs sanity checks on training data, extracts encodings for labels.

        Args:
          training_data: The complete training data.

        Returns:
          the training data required to train this component
        """
        if not training_data.nlu_examples:
            return RasaModelData()

        # apply bilou tagging to all messages
        if self.component_config[BILOU_FLAG]:
            bilou_utils.apply_bilou_schema(training_data)
        self._entity_tag_specs = self._create_entity_tag_specs(training_data)

        # filter messages and extract labels
        training_messages = training_data.nlu_examples
        if self.label_attribute is not None:
            # keep only the messages that have a label
            training_messages = [
                example
                for example in training_messages
                if self.label_attribute in example.data
            ]
            # construct a label to index mapping
            label_id_index_mapping = self._create_label_id_to_index_mapping(
                training_messages,
            )
            # give up if no messages/labels are there:
            if not label_id_index_mapping:
                # no labels are present to train
                return RasaModelData()

            self.index_label_id_mapping = self._invert_mapping(label_id_index_mapping)
            self._label_data = self._create_label_data(
                training_messages, label_id_index_mapping, self.label_attribute
            )
        else:
            # TODO: check that at least some entities are contained... ?

            # Create this dummy label data, because the `RasaModel` cannot deal
            # with empty label data at the moment:
            label_id_index_mapping = {"0": 0}
            self.index_label_id_mapping = self._invert_mapping(label_id_index_mapping)
            self._label_data = self._create_label_data(
                messages=[Message(data={"dummy": "0"})],
                label_id_dict=label_id_index_mapping,
                label_attribute="dummy",
            )

        # sanity checks
        self._check_training_messages(messages=training_messages)

        # create model data from filtered list of messages
        model_data = self._create_model_data(
            training_messages, training=True, label_id_dict=label_id_index_mapping,
        )

        self._check_input_dimension_consistency(model_data)

        return model_data

    def _check_training_messages(self, messages: List[Message]) -> None:
        """Runs sanity checks for the training messages.

        Note that this check assumes that all messages where featurized by the same
        featurizers and hence contain the same features.
        """
        first_message = messages[0]  # because we assume they all look alike
        featurizers = self.component_config[FEATURIZERS]

        # we don't use ".get_dense_features" etc. from message because that
        # method concatenates features and we loose the origin information
        relevant_features = [
            feature
            for feature in first_message.features
            if (not featurizers) or feature.origin in featurizers
        ]
        relevant_features_by_type = {
            type: [] for type in [FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE]
        }
        for feature in relevant_features:
            relevant_features_by_type.setdefault(feature.type, []).append(feature)

        if self._needs_sentence_features_for_input_text():
            if not any(
                feature.attribute == TEXT
                for feature in relevant_features_by_type[FEATURE_TYPE_SENTENCE]
            ):
                raise RasaModelConfigException(
                    "Expected all featurizers to produce sentence features."
                )

        if self._uses_sequence_features_for_input_text():
            # there should be a sentence feature for every sequence feature
            # because DIET will try to append the sentence feature to the sequence
            # ... but we don't want to break things running with LexicalSyntactic
            # so just warn.
            if not set(
                feature.origin
                for feature in relevant_features_by_type[FEATURE_TYPE_SENTENCE]
            ) == set(
                feature.origin
                for feature in relevant_features_by_type[FEATURE_TYPE_SEQUENCE]
            ):
                rasa.shared.utils.io.raise_warning(
                    "Expected all featurizers to produce sequence "
                    "and sentence features. Continuing nonetheless."
                )

    @staticmethod
    def _check_enough_labels(model_data: RasaModelData) -> bool:
        return len(np.unique(model_data.get(LABEL_KEY, LABEL_SUB_KEY))) >= 2

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

        if not self.finetune_mode:
            # No pre-trained model to load from. Create a new instance of the model.
            self.model = self._instantiate_model_class(model_data)
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(self.component_config[LEARNING_RATE])
            )
        else:
            self.model.adjust_for_incremental_training(
                data_example=self._data_example,
                new_sparse_feature_sizes=model_data.get_sparse_feature_sizes(),
                old_sparse_feature_sizes=self._sparse_feature_sizes,
            )
        self._sparse_feature_sizes = model_data.get_sparse_feature_sizes()

        data_generator, validation_data_generator = train_utils.create_data_generators(
            model_data,
            self.component_config[BATCH_SIZES],
            self.component_config[EPOCHS],
            self.component_config[BATCH_STRATEGY],
            self.component_config[EVAL_NUM_EXAMPLES],
            self.component_config[RANDOM_SEED],
        )
        callbacks = train_utils.create_common_callbacks(
            self.component_config[EPOCHS],
            self.component_config[TENSORBOARD_LOG_DIR],
            self.component_config[TENSORBOARD_LOG_LEVEL],
            self.tmp_checkpoint_dir,
        )

        self.model.fit(
            data_generator,
            epochs=self.component_config[EPOCHS],
            validation_data=validation_data_generator,
            validation_freq=self.component_config[EVAL_NUM_EPOCHS],
            callbacks=callbacks,
            verbose=False,
            shuffle=False,  # we use custom shuffle inside data generator
        )

    # process helpers
    def _predict(
        self, message: Message
    ) -> Optional[Dict[Text, Union[tf.Tensor, Dict[Text, tf.Tensor]]]]:
        if self.model is None:
            logger.debug(
                f"There is no trained model for '{self.__class__.__name__}': The "
                f"component is either not trained or didn't receive enough training "
                f"data."
            )
            return None

        # create session data from message and convert it into a batch of 1
        model_data = self._create_model_data([message], training=False)
        return self.model.run_inference(model_data)

    def _predict_label(
        self, predict_out: Optional[Dict[Text, tf.Tensor]]
    ) -> Tuple[Dict[Text, Any], List[Dict[Text, Any]]]:
        """Predicts the intent of the provided message."""
        label: Dict[Text, Any] = {"name": None, "id": None, "confidence": 0.0}
        label_ranking = []

        if predict_out is None:
            return label, label_ranking

        message_sim = predict_out["i_scores"]

        message_sim = message_sim.flatten()  # sim is a matrix

        label_ids = message_sim.argsort()[::-1]

        if (
            self.component_config[RANKING_LENGTH] > 0
            and self.component_config[MODEL_CONFIDENCE] == SOFTMAX
        ):
            # TODO: This should be removed in 3.0 when softmax as
            #  model confidence and normalization is completely deprecated.
            message_sim = train_utils.normalize(
                message_sim, self.component_config[RANKING_LENGTH]
            )
        message_sim[::-1].sort()
        message_sim = message_sim.tolist()

        # if X contains all zeros do not predict some label
        if label_ids.size > 0:
            label = {
                "id": hash(self.index_label_id_mapping[label_ids[0]]),
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
                {
                    "id": hash(self.index_label_id_mapping[label_idx]),
                    "name": self.index_label_id_mapping[label_idx],
                    "confidence": score,
                }
                for label_idx, score in ranking
            ]

        return label, label_ranking

    def _predict_entities(
        self, predict_out: Optional[Dict[Text, tf.Tensor]], message: Message
    ) -> List[Dict]:
        if predict_out is None:
            return []

        predicted_tags, confidence_values = train_utils.entity_label_to_tags(
            predict_out, self._entity_tag_specs, self.component_config[BILOU_FLAG]
        )

        entities = self.convert_predictions_into_entities(
            message.get(TEXT),
            message.get(TOKENS_NAMES[TEXT], []),
            predicted_tags,
            self.split_entities_config,
            confidence_values,
        )

        entities = self.add_extractor_name(entities)
        entities = message.get(ENTITIES, []) + entities

        return entities

    def process(self, message: Message, **kwargs: Any) -> None:
        """Augments the message with intents, entities, and diagnostic data."""
        out = self._predict(message)

        if self.component_config[INTENT_CLASSIFICATION]:
            label, label_ranking = self._predict_label(out)

            message.set(INTENT, label, add_to_output=True)
            message.set("intent_ranking", label_ranking, add_to_output=True)

        if self.component_config[ENTITY_RECOGNITION]:
            entities = self._predict_entities(out, message)

            message.set(ENTITIES, entities, add_to_output=True)

        if out and DIAGNOSTIC_DATA in out:
            message.add_diagnostic_data(self.unique_name, out.get(DIAGNOSTIC_DATA))

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """
        import shutil

        if self.model is None:
            return {"file": None}

        model_dir = Path(model_dir)
        tf_model_file = model_dir / f"{file_name}.tf_model"

        rasa.shared.utils.io.create_directory_for_file(tf_model_file)

        if self.component_config[CHECKPOINT_MODEL]:
            shutil.move(self.tmp_checkpoint_dir, model_dir / "checkpoints")
        self.model.save(str(tf_model_file))

        io_utils.pickle_dump(
            model_dir / f"{file_name}.data_example.pkl", self._data_example
        )
        io_utils.pickle_dump(
            model_dir / f"{file_name}.sparse_feature_sizes.pkl",
            self._sparse_feature_sizes,
        )
        io_utils.pickle_dump(
            model_dir / f"{file_name}.label_data.pkl", dict(self._label_data.data)
        )
        io_utils.json_pickle(
            model_dir / f"{file_name}.index_label_id_mapping.json",
            self.index_label_id_mapping,
        )

        entity_tag_specs = (
            [tag_spec._asdict() for tag_spec in self._entity_tag_specs]
            if self._entity_tag_specs
            else []
        )
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            model_dir / f"{file_name}.entity_tag_specs.json", entity_tag_specs
        )

        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text,
        model_metadata: Metadata = None,
        cached_component: Optional["DIETClassifier"] = None,
        should_finetune: bool = False,
        **kwargs: Any,
    ) -> "DIETClassifier":
        """Loads the trained model from the provided directory."""
        if not meta.get("file"):
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
            sparse_feature_sizes,
        ) = cls._load_from_files(meta, model_dir)

        meta = train_utils.override_defaults(cls.defaults, meta)
        meta = train_utils.update_confidence_type(meta)
        meta = train_utils.update_similarity_type(meta)
        meta = train_utils.update_deprecated_loss_type(meta)

        model = cls._load_model(
            entity_tag_specs,
            label_data,
            meta,
            data_example,
            model_dir,
            finetune_mode=should_finetune,
        )

        return cls(
            component_config=meta,
            index_label_id_mapping=index_label_id_mapping,
            entity_tag_specs=entity_tag_specs,
            model=model,
            finetune_mode=should_finetune,
            sparse_feature_sizes=sparse_feature_sizes,
        )

    @classmethod
    def _load_from_files(
        cls, meta: Dict[Text, Any], model_dir: Text
    ) -> Tuple[
        Dict[int, Text],
        List[EntityTagSpec],
        RasaModelData,
        Dict[Text, Any],
        Dict[Text, Dict[Text, List[FeatureArray]]],
        Dict[Text, Dict[Text, List[int]]],
    ]:
        file_name = meta.get("file")

        model_dir = Path(model_dir)

        data_example = io_utils.pickle_load(model_dir / f"{file_name}.data_example.pkl")
        label_data = io_utils.pickle_load(model_dir / f"{file_name}.label_data.pkl")
        label_data = RasaModelData(data=label_data)
        sparse_feature_sizes = io_utils.pickle_load(
            model_dir / f"{file_name}.sparse_feature_sizes.pkl"
        )
        index_label_id_mapping = io_utils.json_unpickle(
            model_dir / f"{file_name}.index_label_id_mapping.json"
        )
        entity_tag_specs = rasa.shared.utils.io.read_json_file(
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
            sparse_feature_sizes,
        )

    @classmethod
    def _load_model(
        cls,
        entity_tag_specs: List[EntityTagSpec],
        label_data: RasaModelData,
        meta: Dict[Text, Any],
        data_example: Dict[Text, Dict[Text, List[FeatureArray]]],
        model_dir: Text,
        finetune_mode: bool = False,
    ) -> "RasaModel":
        file_name = meta.get("file")
        tf_model_file = os.path.join(model_dir, file_name + ".tf_model")

        label_key = LABEL_KEY if meta[INTENT_CLASSIFICATION] else None
        label_sub_key = LABEL_SUB_KEY if meta[INTENT_CLASSIFICATION] else None

        model_data_example = RasaModelData(
            label_key=label_key, label_sub_key=label_sub_key, data=data_example
        )

        model = cls._load_model_class(
            tf_model_file,
            model_data_example,
            label_data,
            entity_tag_specs,
            meta,
            finetune_mode=finetune_mode,
        )

        return model

    @classmethod
    def _load_model_class(
        cls,
        tf_model_file: Text,
        model_data_example: RasaModelData,
        label_data: RasaModelData,
        entity_tag_specs: List[EntityTagSpec],
        meta: Dict[Text, Any],
        finetune_mode: bool,
    ) -> "RasaModel":

        predict_data_example = RasaModelData(
            label_key=model_data_example.label_key,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                if TEXT in feature_name
            },
        )

        return cls.model_class().load(
            tf_model_file,
            model_data_example,
            predict_data_example,
            data_signature=model_data_example.get_signature(),
            label_data=label_data,
            entity_tag_specs=entity_tag_specs,
            config=copy.deepcopy(meta),
            finetune_mode=finetune_mode,
        )

    def _instantiate_model_class(self, model_data: RasaModelData) -> "RasaModel":
        return self.model_class()(
            data_signature=model_data.get_signature(),
            label_data=self._label_data,
            entity_tag_specs=self._entity_tag_specs,
            config=self.component_config,
        )


class DIET(TransformerRasaModel):
    """Dual Intent and Entity Transformer (DIET) implementation."""

    def __init__(
        self,
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
        label_data: RasaModelData,
        entity_tag_specs: Optional[List[EntityTagSpec]],
        config: Dict[Text, Any],
    ) -> None:
        """Instantiate `DIET`.

        Args:
          data_signature: the model's signature
          label_data: label data
          entity_tag_specs: specification of entity specs
          config: configuration
        """
        # create entity tag spec before calling super otherwise building the model
        # will fail
        super().__init__("DIET", config, data_signature, label_data)
        self._entity_tag_specs = self._ordered_tag_specs(entity_tag_specs)

        self.predict_data_signature = {
            feature_name: features
            for feature_name, features in data_signature.items()
            if TEXT in feature_name
        }

        # tf training
        self._create_metrics()
        self._update_metrics_to_log()

        # needed for efficient prediction
        self.all_labels_embed: Optional[tf.Tensor] = None

        self._prepare_layers()

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
        if TEXT not in self.data_signature:
            raise InvalidConfigException(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if self.config[INTENT_CLASSIFICATION]:
            if LABEL not in self.data_signature:
                raise InvalidConfigException(
                    f"No label features specified. "
                    f"Cannot train '{self.__class__.__name__}' model."
                )

            if self.config[SHARE_HIDDEN_LAYERS]:
                different_sentence_signatures = False
                different_sequence_signatures = False
                if (
                    SENTENCE in self.data_signature[TEXT]
                    and SENTENCE in self.data_signature[LABEL]
                ):
                    different_sentence_signatures = (
                        self.data_signature[TEXT][SENTENCE]
                        != self.data_signature[LABEL][SENTENCE]
                    )
                if (
                    SEQUENCE in self.data_signature[TEXT]
                    and SEQUENCE in self.data_signature[LABEL]
                ):
                    different_sequence_signatures = (
                        self.data_signature[TEXT][SEQUENCE]
                        != self.data_signature[LABEL][SEQUENCE]
                    )

                if different_sentence_signatures or different_sequence_signatures:
                    raise RasaModelConfigException(
                        "If hidden layer weights are shared, data signatures "
                        "for text_features and label_features must coincide."
                    )

        if self.config[ENTITY_RECOGNITION] and (
            ENTITIES not in self.data_signature
            or ENTITY_ATTRIBUTE_TYPE not in self.data_signature[ENTITIES]
        ):
            logger.debug(
                f"You specified '{self.__class__.__name__}' to train entities, but "
                f"no entities are present in the training data. Skipping training of "
                f"entities."
            )
            self.config[ENTITY_RECOGNITION] = False

    def _create_metrics(self) -> None:
        # self.metrics will have the same order as they are created
        # so create loss metrics first to output losses first
        self.mask_loss = tf.keras.metrics.Mean(name="m_loss")
        self.intent_loss = tf.keras.metrics.Mean(name="i_loss")
        self.entity_loss = tf.keras.metrics.Mean(name="e_loss")
        self.entity_group_loss = tf.keras.metrics.Mean(name="g_loss")
        self.entity_role_loss = tf.keras.metrics.Mean(name="r_loss")
        # create accuracy metrics second to output accuracies second
        self.mask_acc = tf.keras.metrics.Mean(name="m_acc")
        self.intent_acc = tf.keras.metrics.Mean(name="i_acc")
        self.entity_f1 = tf.keras.metrics.Mean(name="e_f1")
        self.entity_group_f1 = tf.keras.metrics.Mean(name="g_f1")
        self.entity_role_f1 = tf.keras.metrics.Mean(name="r_f1")

    def _update_metrics_to_log(self) -> None:
        debug_log_level = logging.getLogger("rasa").level == logging.DEBUG

        if self.config[MASKED_LM]:
            self.metrics_to_log.append("m_acc")
            if debug_log_level:
                self.metrics_to_log.append("m_loss")
        if self.config[INTENT_CLASSIFICATION]:
            self.metrics_to_log.append("i_acc")
            if debug_log_level:
                self.metrics_to_log.append("i_loss")
        if self.config[ENTITY_RECOGNITION]:
            for tag_spec in self._entity_tag_specs:
                if tag_spec.num_tags != 0:
                    name = tag_spec.tag_name
                    self.metrics_to_log.append(f"{name[0]}_f1")
                    if debug_log_level:
                        self.metrics_to_log.append(f"{name[0]}_loss")

        self._log_metric_info()

    def _log_metric_info(self) -> None:
        metric_name = {
            "t": "total",
            "i": "intent",
            "e": "entity",
            "m": "mask",
            "r": "role",
            "g": "group",
        }
        logger.debug("Following metrics will be logged during training: ")
        for metric in self.metrics_to_log:
            parts = metric.split("_")
            name = f"{metric_name[parts[0]]} {parts[1]}"
            logger.debug(f"  {metric} ({name})")

    def _prepare_layers(self) -> None:
        # For user text, prepare layers that combine different feature types, embed
        # everything using a transformer and optionally also do masked language
        # modeling.
        self.text_name = TEXT
        self._tf_layers[
            f"sequence_layer.{self.text_name}"
        ] = rasa_layers.RasaSequenceLayer(
            self.text_name, self.data_signature[self.text_name], self.config
        )
        if self.config[MASKED_LM]:
            self._prepare_mask_lm_loss(self.text_name)

        # Intent labels are treated similarly to user text but without the transformer,
        # without masked language modelling, and with no dropout applied to the
        # individual features, only to the overall label embedding after all label
        # features have been combined.
        if self.config[INTENT_CLASSIFICATION]:
            self.label_name = TEXT if self.config[SHARE_HIDDEN_LAYERS] else LABEL

            # disable input dropout applied to sparse and dense label features
            label_config = self.config.copy()
            label_config.update(
                {SPARSE_INPUT_DROPOUT: False, DENSE_INPUT_DROPOUT: False}
            )

            self._tf_layers[
                f"feature_combining_layer.{self.label_name}"
            ] = rasa_layers.RasaFeatureCombiningLayer(
                self.label_name, self.label_signature[self.label_name], label_config
            )

            self._prepare_ffnn_layer(
                self.label_name,
                self.config[HIDDEN_LAYERS_SIZES][self.label_name],
                self.config[DROP_RATE],
            )

            self._prepare_label_classification_layers(predictor_attribute=TEXT)

        if self.config[ENTITY_RECOGNITION]:
            self._prepare_entity_recognition_layers()

    def _prepare_mask_lm_loss(self, name: Text) -> None:
        # for embedding predicted tokens at masked positions
        self._prepare_embed_layers(f"{name}_lm_mask")

        # for embedding the true tokens that got masked
        self._prepare_embed_layers(f"{name}_golden_token")

        # mask loss is additional loss
        # set scaling to False, so that it doesn't overpower other losses
        self._prepare_dot_product_loss(f"{name}_mask", scale_loss=False)

    def _create_bow(
        self,
        sequence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        sentence_features: List[Union[tf.Tensor, tf.SparseTensor]],
        sequence_feature_lengths: tf.Tensor,
        name: Text,
    ) -> tf.Tensor:

        x, _ = self._tf_layers[f"feature_combining_layer.{name}"](
            (sequence_features, sentence_features, sequence_feature_lengths),
            training=self._training,
        )

        # convert to bag-of-words by summing along the sequence dimension
        x = tf.reduce_sum(x, axis=1)

        return self._tf_layers[f"ffnn.{name}"](x, self._training)

    def _create_all_labels(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_label_ids = self.tf_label_data[LABEL_KEY][LABEL_SUB_KEY][0]

        sequence_feature_lengths = self._get_sequence_feature_lengths_or_zeros(
            self.tf_label_data, LABEL
        )

        # Note that `tf_label_data` is a nested defaultdict and hence the subkeys
        # will default to empty lists if there are no `SEQUENCE` or `SENTENCE` subkeys.
        x = self._create_bow(
            self.tf_label_data[LABEL][SEQUENCE],
            self.tf_label_data[LABEL][SENTENCE],
            sequence_feature_lengths,
            self.label_name,
        )
        all_labels_embed = self._tf_layers[f"embed.{LABEL}"](x)

        return all_label_ids, all_labels_embed

    def _mask_loss(
        self,
        outputs: tf.Tensor,
        inputs: tf.Tensor,
        seq_ids: tf.Tensor,
        mlm_mask_boolean: tf.Tensor,
        name: Text,
    ) -> tf.Tensor:
        # make sure there is at least one element in the mask
        mlm_mask_boolean = tf.cond(
            tf.reduce_any(mlm_mask_boolean),
            lambda: mlm_mask_boolean,
            lambda: tf.scatter_nd([[0, 0, 0]], [True], tf.shape(mlm_mask_boolean)),
        )

        mlm_mask_boolean = tf.squeeze(mlm_mask_boolean, -1)

        # Pick elements that were masked, throwing away the batch & sequence dimension
        # and effectively switching from shape (batch_size, sequence_length, units) to
        # (num_masked_elements, units).
        outputs = tf.boolean_mask(outputs, mlm_mask_boolean)
        inputs = tf.boolean_mask(inputs, mlm_mask_boolean)
        ids = tf.boolean_mask(seq_ids, mlm_mask_boolean)

        tokens_predicted_embed = self._tf_layers[f"embed.{name}_lm_mask"](outputs)
        tokens_true_embed = self._tf_layers[f"embed.{name}_golden_token"](inputs)

        # To limit the otherwise computationally expensive loss calculation, we
        # constrain the label space in MLM (i.e. token space) to only those tokens that
        # were masked in this batch. Hence the reduced list of token embeddings
        # (tokens_true_embed) and the reduced list of labels (ids) are passed as
        # all_labels_embed and all_labels, respectively. In the future, we could be less
        # restrictive and construct a slightly bigger label space which could include
        # tokens not masked in the current batch too.
        return self._tf_layers[f"loss.{name}_mask"](
            inputs_embed=tokens_predicted_embed,
            labels_embed=tokens_true_embed,
            labels=ids,
            all_labels_embed=tokens_true_embed,
            all_labels=ids,
        )

    def _calculate_label_loss(
        self, text_features: tf.Tensor, label_features: tf.Tensor, label_ids: tf.Tensor
    ) -> tf.Tensor:
        all_label_ids, all_labels_embed = self._create_all_labels()

        text_embed = self._tf_layers[f"embed.{TEXT}"](text_features)
        label_embed = self._tf_layers[f"embed.{LABEL}"](label_features)

        return self._tf_layers[f"loss.{LABEL}"](
            text_embed, label_embed, label_ids, all_labels_embed, all_label_ids
        )

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        """Calculates the loss for the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The loss of the given batch.
        """
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)

        # compute the sequence lengths of the `SEQUENCE` features, if they are
        # present, and a sequence of just 0s otherwise
        sequence_feature_lengths = self._get_sequence_feature_lengths_or_zeros(
            tf_batch_data, TEXT
        )

        (
            text_transformed,
            text_in,
            mask_combined_sequence_sentence,
            text_seq_ids,
            mlm_mask_boolean_text,
            _,
        ) = self._tf_layers[f"sequence_layer.{self.text_name}"](
            (
                tf_batch_data[TEXT][SEQUENCE],
                tf_batch_data[TEXT][SENTENCE],
                sequence_feature_lengths,
            ),
            training=self._training,
        )

        losses = []

        # Lengths of sequences in case of sentence-level features are always 1, but they
        # can effectively be 0 if sentence-level features aren't present.
        sentence_feature_lengths = self._get_sentence_feature_lengths_or_zeros(
            tf_batch_data, TEXT
        )

        combined_sequence_sentence_feature_lengths = (
            sequence_feature_lengths + sentence_feature_lengths
        )

        if self.config[MASKED_LM]:
            loss, acc = self._mask_loss(
                text_transformed, text_in, text_seq_ids, mlm_mask_boolean_text, TEXT
            )
            self.mask_loss.update_state(loss)
            self.mask_acc.update_state(acc)
            losses.append(loss)

        if self.config[INTENT_CLASSIFICATION]:
            loss = self._batch_loss_intent(
                combined_sequence_sentence_feature_lengths,
                text_transformed,
                tf_batch_data,
            )
            losses.append(loss)

        if self.config[ENTITY_RECOGNITION]:
            losses += self._batch_loss_entities(
                mask_combined_sequence_sentence,
                sequence_feature_lengths,
                text_transformed,
                tf_batch_data,
            )

        return tf.math.add_n(losses)

    def _batch_loss_intent(
        self,
        combined_sequence_sentence_feature_lengths_text: tf.Tensor,
        text_transformed: tf.Tensor,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
    ) -> tf.Tensor:
        # get sentence features vector for intent classification
        sentence_vector = self._last_token(
            text_transformed, combined_sequence_sentence_feature_lengths_text
        )

        sequence_feature_lengths_label = self._get_sequence_feature_lengths_or_zeros(
            tf_batch_data, LABEL
        )

        label_ids = tf_batch_data[LABEL_KEY][LABEL_SUB_KEY][0]
        # Note that `tf_batch_data` is a nested defaultdict and hence the subkeys
        # will default to empty lists if there are no `SEQUENCE` or `SENTENCE` subkeys.
        label = self._create_bow(
            tf_batch_data[LABEL][SEQUENCE],
            tf_batch_data[LABEL][SENTENCE],
            sequence_feature_lengths_label,
            self.label_name,
        )
        loss, acc = self._calculate_label_loss(sentence_vector, label, label_ids)

        self._update_label_metrics(loss, acc)

        return loss

    def _update_label_metrics(self, loss: tf.Tensor, acc: tf.Tensor) -> None:

        self.intent_loss.update_state(loss)
        self.intent_acc.update_state(acc)

    def _batch_loss_entities(
        self,
        mask_combined_sequence_sentence: tf.Tensor,
        sequence_feature_lengths: tf.Tensor,
        text_transformed: tf.Tensor,
        tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]],
    ) -> List[tf.Tensor]:
        losses = []

        entity_tags = None

        for tag_spec in self._entity_tag_specs:
            if tag_spec.num_tags == 0:
                continue

            tag_ids = tf_batch_data[ENTITIES][tag_spec.tag_name][0]
            # add a zero (no entity) for the sentence features to match the shape of
            # inputs
            tag_ids = tf.pad(tag_ids, [[0, 0], [0, 1], [0, 0]])

            loss, f1, _logits = self._calculate_entity_loss(
                text_transformed,
                tag_ids,
                mask_combined_sequence_sentence,
                sequence_feature_lengths,
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

    def _update_entity_metrics(
        self, loss: tf.Tensor, f1: tf.Tensor, tag_name: Text
    ) -> None:
        if tag_name == ENTITY_ATTRIBUTE_TYPE:
            self.entity_loss.update_state(loss)
            self.entity_f1.update_state(f1)
        elif tag_name == ENTITY_ATTRIBUTE_GROUP:
            self.entity_group_loss.update_state(loss)
            self.entity_group_f1.update_state(f1)
        elif tag_name == ENTITY_ATTRIBUTE_ROLE:
            self.entity_role_loss.update_state(loss)
            self.entity_role_f1.update_state(f1)

    def prepare_for_predict(self) -> None:
        """Prepares the model for prediction."""
        if self.config[INTENT_CLASSIFICATION]:
            _, self.all_labels_embed = self._create_all_labels()

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor, ...], Tuple[np.ndarray, ...]]
    ) -> Dict[Text, tf.Tensor]:
        """Predicts the output of the given batch.

        Args:
            batch_in: The batch.

        Returns:
            The output to predict.
        """
        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )

        sequence_feature_lengths = self._get_sequence_feature_lengths_or_zeros(
            tf_batch_data, TEXT
        )
        sentence_feature_lengths = self._get_sentence_feature_lengths_or_zeros(
            tf_batch_data, TEXT,
        )

        # Note that `tf_batch_data` is a nested defaultdict and hence the subkeys
        # will default to empty lists if there are no `SEQUENCE` or `SENTENCE` subkeys.
        text_transformed, _, _, _, _, attention_weights = self._tf_layers[
            f"sequence_layer.{self.text_name}"
        ](
            (
                tf_batch_data[TEXT][SEQUENCE],
                tf_batch_data[TEXT][SENTENCE],
                sequence_feature_lengths,
            ),
            training=self._training,
        )
        predictions = {
            DIAGNOSTIC_DATA: {
                "attention_weights": attention_weights,
                "text_transformed": text_transformed,
            }
        }

        if self.config[INTENT_CLASSIFICATION]:
            predictions.update(
                self._batch_predict_intents(
                    sequence_feature_lengths + sentence_feature_lengths,
                    text_transformed,
                )
            )

        if self.config[ENTITY_RECOGNITION]:
            predictions.update(
                self._batch_predict_entities(sequence_feature_lengths, text_transformed)
            )

        return predictions

    def _batch_predict_entities(
        self, sequence_feature_lengths: tf.Tensor, text_transformed: tf.Tensor
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
            pred_ids, confidences = self._tf_layers[f"crf.{name}"](
                _logits, sequence_feature_lengths
            )

            predictions[f"e_{name}_ids"] = pred_ids
            predictions[f"e_{name}_scores"] = confidences

            if name == ENTITY_ATTRIBUTE_TYPE:
                # use the entity tags as additional input for the role
                # and group CRF
                entity_tags = tf.one_hot(
                    tf.cast(pred_ids, tf.int32), depth=tag_spec.num_tags
                )

        return predictions

    def _batch_predict_intents(
        self,
        combined_sequence_sentence_feature_lengths: tf.Tensor,
        text_transformed: tf.Tensor,
    ) -> Dict[Text, tf.Tensor]:

        if self.all_labels_embed is None:
            raise RasaModelConfigException(
                "The model was not prepared for prediction. "
                "Call `prepare_for_predict` first."
            )

        # get sentence feature vector for intent classification
        sentence_vector = self._last_token(
            text_transformed, combined_sequence_sentence_feature_lengths
        )
        sentence_vector_embed = self._tf_layers[f"embed.{TEXT}"](sentence_vector)

        _, scores = self._tf_layers[
            f"loss.{LABEL}"
        ].get_similarities_and_confidences_from_embeddings(
            sentence_vector_embed[:, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, :, :],
        )

        return {"i_scores": scores}
