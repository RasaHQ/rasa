# WARNING: This module will be dropped before Rasa Open Source 3.0 is released.
#          Please don't do any changes in this module and rather adapt `TEDPolicyGraphComponent` from
#          the regular `rasa.core.policies.ted_policy` module. This module is a
#          workaround to defer breaking changes due to the architecture revamp in 3.0.
# flake8: noqa
import logging
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Text, Any, Optional, List, Tuple, Type, Union, TYPE_CHECKING

import numpy as np
from rasa.core.constants import DIALOGUE, DEFAULT_POLICY_PRIORITY
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import (
    MaxHistoryTrackerFeaturizer,
    TrackerFeaturizer,
)
from rasa.core.policies import Policy
import rasa.utils.train_utils
from rasa.core.policies.policy import PolicyPrediction
from rasa.nlu.constants import TOKENS_NAMES
from rasa.nlu.extractors.extractor import EntityTagSpec, EntityExtractor
from rasa.shared.constants import DIAGNOSTIC_DATA
from rasa.shared.core.constants import SLOTS, ACTIVE_LOOP, ACTION_LISTEN_NAME
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import EntitiesAdded, Event
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.training_data.message import Message
from rasa.utils.tensorflow.model_data import RasaModelData, FeatureArray, Data
from rasa.utils.tensorflow.model_data_utils import convert_to_data_format
from rasa.shared.nlu.constants import (
    ACTION_TEXT,
    ACTION_NAME,
    INTENT,
    TEXT,
    ENTITIES,
    ENTITY_ATTRIBUTE_TYPE,
    ENTITY_TAGS,
    EXTRACTOR,
    SPLIT_ENTITIES_BY_COMMA,
    SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
)

from rasa.utils.tensorflow.constants import (
    LABEL,
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
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    NEGATIVE_MARGIN_SCALE,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    DROP_RATE_DIALOGUE,
    DROP_RATE_LABEL,
    DROP_RATE,
    DROP_RATE_ATTENTION,
    CONNECTION_DENSITY,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    CROSS_ENTROPY,
    AUTO,
    BALANCED,
    TENSORBOARD_LOG_DIR,
    TENSORBOARD_LOG_LEVEL,
    CHECKPOINT_MODEL,
    ENCODING_DIMENSION,
    UNIDIRECTIONAL_ENCODER,
    SEQUENCE,
    SEQUENCE_LENGTH,
    DENSE_DIMENSION,
    CONCAT_DIMENSION,
    SPARSE_INPUT_DROPOUT,
    DENSE_INPUT_DROPOUT,
    MASKED_LM,
    MASK,
    HIDDEN_LAYERS_SIZES,
    FEATURIZERS,
    ENTITY_RECOGNITION,
    CONSTRAIN_SIMILARITIES,
    MODEL_CONFIDENCE,
    SOFTMAX,
    BILOU_FLAG,
    IDS,
)
from rasa.utils.tensorflow.models import RasaModel
import rasa.utils.io
import rasa.shared.utils.io
from rasa.shared.utils import io as shared_io_utils
import rasa.utils.io as io_utils
import tensorflow as tf

if TYPE_CHECKING:
    from rasa.shared.nlu.training_data.features import Features
    from rasa.core.policies.ted_policy import TED

logger = logging.getLogger(__name__)

E2E_CONFIDENCE_THRESHOLD = "e2e_confidence_threshold"
LABEL_KEY = LABEL
LABEL_SUB_KEY = IDS
LENGTH = "length"
INDICES = "indices"
SENTENCE_FEATURES_TO_ENCODE = [INTENT, TEXT, ACTION_NAME, ACTION_TEXT]
SEQUENCE_FEATURES_TO_ENCODE = [TEXT, ACTION_TEXT, f"{LABEL}_{ACTION_TEXT}"]
LABEL_FEATURES_TO_ENCODE = [
    f"{LABEL}_{ACTION_NAME}",
    f"{LABEL}_{ACTION_TEXT}",
    f"{LABEL}_{INTENT}",
]
STATE_LEVEL_FEATURES = [ENTITIES, SLOTS, ACTIVE_LOOP]
PREDICTION_FEATURES = STATE_LEVEL_FEATURES + SENTENCE_FEATURES_TO_ENCODE + [DIALOGUE]


class TEDPolicy(Policy):
    """Transformer Embedding Dialogue (TED) Policy.

    The model architecture is described in
    detail in https://arxiv.org/abs/1910.00486.
    In summary, the architecture comprises of the
    following steps:
        - concatenate user input (user intent and entities), previous system actions,
          slots and active forms for each time step into an input vector to
          pre-transformer embedding layer;
        - feed it to transformer;
        - apply a dense layer to the output of the transformer to get embeddings of a
          dialogue for each time step;
        - apply a dense layer to create embeddings for system actions for each time
          step;
        - calculate the similarity between the dialogue embedding and embedded system
          actions. This step is based on the StarSpace
          (https://arxiv.org/abs/1709.03856) idea.
    """

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # ## Architecture of the used neural network
        # Hidden layer sizes for layers before the embedding layers for user message
        # and labels.
        # The number of hidden layers is equal to the length of the corresponding list.
        HIDDEN_LAYERS_SIZES: {TEXT: [], ACTION_TEXT: [], f"{LABEL}_{ACTION_TEXT}": []},
        # Dense dimension to use for sparse features.
        DENSE_DIMENSION: {
            TEXT: 128,
            ACTION_TEXT: 128,
            f"{LABEL}_{ACTION_TEXT}": 128,
            INTENT: 20,
            ACTION_NAME: 20,
            f"{LABEL}_{ACTION_NAME}": 20,
            ENTITIES: 20,
            SLOTS: 20,
            ACTIVE_LOOP: 20,
        },
        # Default dimension to use for concatenating sequence and sentence features.
        CONCAT_DIMENSION: {TEXT: 128, ACTION_TEXT: 128, f"{LABEL}_{ACTION_TEXT}": 128},
        # Dimension size of embedding vectors before the dialogue transformer encoder.
        ENCODING_DIMENSION: 50,
        # Number of units in transformer encoders
        TRANSFORMER_SIZE: {
            TEXT: 128,
            ACTION_TEXT: 128,
            f"{LABEL}_{ACTION_TEXT}": 128,
            DIALOGUE: 128,
        },
        # Number of layers in transformer encoders
        NUM_TRANSFORMER_LAYERS: {
            TEXT: 1,
            ACTION_TEXT: 1,
            f"{LABEL}_{ACTION_TEXT}": 1,
            DIALOGUE: 1,
        },
        # Number of attention heads in transformer
        NUM_HEADS: 4,
        # If 'True' use key relative embeddings in attention
        KEY_RELATIVE_ATTENTION: False,
        # If 'True' use value relative embeddings in attention
        VALUE_RELATIVE_ATTENTION: False,
        # Max position for relative embeddings. Only in effect if key- or value relative
        # attention are turned on
        MAX_RELATIVE_POSITION: 5,
        # Use a unidirectional or bidirectional encoder
        # for `text`, `action_text`, and `label_action_text`.
        UNIDIRECTIONAL_ENCODER: False,
        # ## Training parameters
        # Initial and final batch sizes:
        # Batch size will be linearly increased for each epoch.
        BATCH_SIZES: [64, 256],
        # Strategy used whenc creating batches.
        # Can be either 'sequence' or 'balanced'.
        BATCH_STRATEGY: BALANCED,
        # Number of epochs to train
        EPOCHS: 1,
        # Set random seed to any 'int' to get reproducible results
        RANDOM_SEED: None,
        # Initial learning rate for the optimizer
        LEARNING_RATE: 0.001,
        # ## Parameters for embeddings
        # Dimension size of embedding vectors
        EMBEDDING_DIMENSION: 20,
        # The number of incorrect labels. The algorithm will minimize
        # their similarity to the user input during training.
        NUM_NEG: 20,
        # Type of similarity measure to use, either 'auto' or 'cosine' or 'inner'.
        SIMILARITY_TYPE: AUTO,
        # The type of the loss function, either 'cross_entropy' or 'margin'.
        LOSS_TYPE: CROSS_ENTROPY,
        # Number of top actions to normalize scores for. Applicable with
        # loss type 'cross_entropy' and 'softmax' confidences. Set to 0
        # to turn off normalization.
        RANKING_LENGTH: 10,
        # Indicates how similar the algorithm should try to make embedding vectors
        # for correct labels.
        # Should be 0.0 < ... < 1.0 for 'cosine' similarity type.
        MAX_POS_SIM: 0.8,
        # Maximum negative similarity for incorrect labels.
        # Should be -1.0 < ... < 1.0 for 'cosine' similarity type.
        MAX_NEG_SIM: -0.2,
        # If 'True' the algorithm only minimizes maximum similarity over
        # incorrect intent labels, used only if 'loss_type' is set to 'margin'.
        USE_MAX_NEG_SIM: True,
        # If 'True' scale loss inverse proportionally to the confidence
        # of the correct prediction
        SCALE_LOSS: True,
        # ## Regularization parameters
        # The scale of regularization
        REGULARIZATION_CONSTANT: 0.001,
        # The scale of how important is to minimize the maximum similarity
        # between embeddings of different labels,
        # used only if 'loss_type' is set to 'margin'.
        NEGATIVE_MARGIN_SCALE: 0.8,
        # Dropout rate for embedding layers of dialogue features.
        DROP_RATE_DIALOGUE: 0.1,
        # Dropout rate for embedding layers of utterance level features.
        DROP_RATE: 0.0,
        # Dropout rate for embedding layers of label, e.g. action, features.
        DROP_RATE_LABEL: 0.0,
        # Dropout rate for attention.
        DROP_RATE_ATTENTION: 0.0,
        # Fraction of trainable weights in internal layers.
        CONNECTION_DENSITY: 0.2,
        # If 'True' apply dropout to sparse input tensors
        SPARSE_INPUT_DROPOUT: True,
        # If 'True' apply dropout to dense input tensors
        DENSE_INPUT_DROPOUT: True,
        # If 'True' random tokens of the input message will be masked. Since there is no
        # related loss term used inside TED, the masking effectively becomes just input
        # dropout applied to the text of user utterances.
        MASKED_LM: False,
        # ## Evaluation parameters
        # How often calculate validation accuracy.
        # Small values may hurt performance.
        EVAL_NUM_EPOCHS: 20,
        # How many examples to use for hold out validation set
        # Large values may hurt performance, e.g. model accuracy.
        # Set to 0 for no validation.
        EVAL_NUM_EXAMPLES: 0,
        # If you want to use tensorboard to visualize training and validation metrics,
        # set this option to a valid output directory.
        TENSORBOARD_LOG_DIR: None,
        # Define when training metrics for tensorboard should be logged.
        # Either after every epoch or for every training step.
        # Valid values: 'epoch' and 'batch'
        TENSORBOARD_LOG_LEVEL: "epoch",
        # Perform model checkpointing
        CHECKPOINT_MODEL: False,
        # Only pick e2e prediction if the policy is confident enough
        E2E_CONFIDENCE_THRESHOLD: 0.5,
        # Specify what features to use as sequence and sentence features.
        # By default all features in the pipeline are used.
        FEATURIZERS: [],
        # If set to true, entities are predicted in user utterances.
        ENTITY_RECOGNITION: True,
        # if 'True' applies sigmoid on all similarity terms and adds
        # it to the loss function to ensure that similarity values are
        # approximately bounded. Used inside softmax loss only.
        CONSTRAIN_SIMILARITIES: False,
        # Model confidence to be returned during inference. Possible values -
        # 'softmax' and 'linear_norm'.
        MODEL_CONFIDENCE: SOFTMAX,
        # 'BILOU_flag' determines whether to use BILOU tagging or not.
        # If set to 'True' labelling is more rigorous, however more
        # examples per entity are required.
        # Rule of thumb: you should have more than 100 examples per entity.
        BILOU_FLAG: True,
        # Split entities by comma, this makes sense e.g. for a list of
        # ingredients in a recipe, but it doesn't make sense for the parts of
        # an address
        SPLIT_ENTITIES_BY_COMMA: SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE,
    }

    @staticmethod
    def _standard_featurizer(max_history: Optional[int] = None) -> TrackerFeaturizer:
        return MaxHistoryTrackerFeaturizer(
            SingleStateFeaturizer(), max_history=max_history
        )

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = DEFAULT_POLICY_PRIORITY,
        max_history: Optional[int] = None,
        model: Optional[RasaModel] = None,
        fake_features: Optional[Dict[Text, List["Features"]]] = None,
        entity_tag_specs: Optional[List[EntityTagSpec]] = None,
        should_finetune: bool = False,
        **kwargs: Any,
    ) -> None:
        """Declares instance variables with default values."""
        self.split_entities_config = rasa.utils.train_utils.init_split_entities(
            kwargs.get(SPLIT_ENTITIES_BY_COMMA, SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE),
            self.defaults.get(
                SPLIT_ENTITIES_BY_COMMA, SPLIT_ENTITIES_BY_COMMA_DEFAULT_VALUE
            ),
        )

        # TODO: check if the else statement can be removed.
        #  More context here -
        #  https://github.com/RasaHQ/rasa/issues/5786#issuecomment-840762751
        if not featurizer:
            featurizer = self._standard_featurizer(max_history)
        else:
            if isinstance(featurizer, MaxHistoryTrackerFeaturizer) and max_history:
                featurizer.max_history = max_history

        super().__init__(
            featurizer, priority, should_finetune=should_finetune, **kwargs
        )
        self._load_params(**kwargs)

        self.model = model

        self._entity_tag_specs = entity_tag_specs

        self.fake_features = fake_features or defaultdict(list)
        # TED is only e2e if only text is present in fake features, which represent
        # all possible input features for current version of this trained ted
        self.only_e2e = TEXT in self.fake_features and INTENT not in self.fake_features

        self._label_data: Optional[RasaModelData] = None
        self.data_example: Optional[Dict[Text, Dict[Text, List[FeatureArray]]]] = None

        self.tmp_checkpoint_dir = None
        if self.config[CHECKPOINT_MODEL]:
            self.tmp_checkpoint_dir = Path(rasa.utils.io.create_temporary_directory())

    @staticmethod
    def model_class() -> Type["TED"]:
        """Gets the class of the model architecture to be used by the policy.

        Returns:
            Required class.
        """
        from rasa.core.policies.ted_policy import TED

        return TED

    @classmethod
    def _metadata_filename(cls) -> Optional[Text]:
        return "ted_policy"

    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        new_config = rasa.utils.train_utils.check_core_deprecated_options(kwargs)
        self.config = rasa.utils.train_utils.override_defaults(
            self.defaults, new_config
        )

        self._auto_update_configuration()

    def _auto_update_configuration(self) -> None:
        """Takes care of deprecations and compatibility of parameters."""
        self.config = rasa.utils.train_utils.update_confidence_type(self.config)
        rasa.utils.train_utils.validate_configuration_settings(self.config)
        self.config = rasa.utils.train_utils.update_similarity_type(self.config)
        self.config = rasa.utils.train_utils.update_evaluation_parameters(self.config)

    def _create_label_data(
        self, domain: Domain, interpreter: NaturalLanguageInterpreter
    ) -> Tuple[RasaModelData, List[Dict[Text, List["Features"]]]]:
        # encode all label_ids with policies' featurizer
        state_featurizer = self.featurizer.state_featurizer
        encoded_all_labels = state_featurizer.encode_all_labels(domain, interpreter)

        attribute_data, _ = convert_to_data_format(
            encoded_all_labels, featurizers=self.config[FEATURIZERS]
        )

        label_data = self._assemble_label_data(attribute_data, domain)

        return label_data, encoded_all_labels

    def _assemble_label_data(
        self, attribute_data: Data, domain: Domain
    ) -> RasaModelData:
        """Constructs data regarding labels to be fed to the model.

        The resultant model data can possibly contain one or both of the
        keys - [`label_action_name`, `label_action_text`] but will definitely
        contain the `label` key.
        `label_action_*` will contain the sequence, sentence and mask features
        for corresponding labels and `label` will contain the numerical label ids.

        Args:
            attribute_data: Feature data for all labels.
            domain: Domain of the assistant.

        Returns:
            Features of labels ready to be fed to the model.
        """
        label_data = RasaModelData()
        label_data.add_data(attribute_data, key_prefix=f"{LABEL_KEY}_")
        label_data.add_lengths(
            f"{LABEL}_{ACTION_TEXT}",
            SEQUENCE_LENGTH,
            f"{LABEL}_{ACTION_TEXT}",
            SEQUENCE,
        )
        label_ids = np.arange(domain.num_actions)
        label_data.add_features(
            LABEL_KEY,
            LABEL_SUB_KEY,
            [FeatureArray(np.expand_dims(label_ids, -1), number_of_dimensions=2)],
        )
        return label_data

    @staticmethod
    def _should_extract_entities(
        entity_tags: List[List[Dict[Text, List["Features"]]]]
    ) -> bool:
        for turns_tags in entity_tags:
            for turn_tags in turns_tags:
                # if turn_tags are empty or all entity tag indices are `0`
                # it means that all the inputs only contain NO_ENTITY_TAG
                if turn_tags and np.any(turn_tags[ENTITY_TAGS][0].features):
                    return True
        return False

    def _create_data_for_entities(
        self, entity_tags: Optional[List[List[Dict[Text, List["Features"]]]]]
    ) -> Optional[Data]:
        if not self.config[ENTITY_RECOGNITION]:
            return None

        # check that there are real entity tags
        if entity_tags and self._should_extract_entities(entity_tags):
            entity_tags_data, _ = convert_to_data_format(entity_tags)
            return entity_tags_data

        # there are no "real" entity tags
        logger.debug(
            f"Entity recognition cannot be performed, "
            f"set '{ENTITY_RECOGNITION}' config parameter to 'False'."
        )
        self.config[ENTITY_RECOGNITION] = False

        return None

    def _create_model_data(
        self,
        tracker_state_features: List[List[Dict[Text, List["Features"]]]],
        label_ids: Optional[np.ndarray] = None,
        entity_tags: Optional[List[List[Dict[Text, List["Features"]]]]] = None,
        encoded_all_labels: Optional[List[Dict[Text, List["Features"]]]] = None,
    ) -> RasaModelData:
        """Combine all model related data into RasaModelData.

        Args:
            tracker_state_features: a dictionary of attributes
                (INTENT, TEXT, ACTION_NAME, ACTION_TEXT, ENTITIES, SLOTS, ACTIVE_LOOP)
                to a list of features for all dialogue turns in all training trackers
            label_ids: the label ids (e.g. action ids) for every dialogue turn in all
                training trackers
            entity_tags: a dictionary of entity type (ENTITY_TAGS) to a list of features
                containing entity tag ids for text user inputs otherwise empty dict
                for all dialogue turns in all training trackers
            encoded_all_labels: a list of dictionaries containing attribute features
                for label ids

        Returns:
            RasaModelData
        """
        model_data = RasaModelData(label_key=LABEL_KEY, label_sub_key=LABEL_SUB_KEY)

        if label_ids is not None and encoded_all_labels is not None:
            label_ids = np.array(
                [np.expand_dims(seq_label_ids, -1) for seq_label_ids in label_ids]
            )
            model_data.add_features(
                LABEL_KEY,
                LABEL_SUB_KEY,
                [FeatureArray(label_ids, number_of_dimensions=3)],
            )

            attribute_data, self.fake_features = convert_to_data_format(
                tracker_state_features, featurizers=self.config[FEATURIZERS]
            )

            entity_tags_data = self._create_data_for_entities(entity_tags)
            if entity_tags_data is not None:
                model_data.add_data(entity_tags_data)
        else:
            # method is called during prediction
            attribute_data, _ = convert_to_data_format(
                tracker_state_features,
                self.fake_features,
                featurizers=self.config[FEATURIZERS],
            )

        model_data.add_data(attribute_data)
        model_data.add_lengths(TEXT, SEQUENCE_LENGTH, TEXT, SEQUENCE)
        model_data.add_lengths(ACTION_TEXT, SEQUENCE_LENGTH, ACTION_TEXT, SEQUENCE)

        # add the dialogue lengths
        attribute_present = next(iter(list(attribute_data.keys())))
        dialogue_lengths = np.array(
            [
                np.size(np.squeeze(f, -1))
                for f in model_data.data[attribute_present][MASK][0]
            ]
        )
        model_data.data[DIALOGUE][LENGTH] = [
            FeatureArray(dialogue_lengths, number_of_dimensions=1)
        ]

        # make sure all keys are in the same order during training and prediction
        model_data.sort()

        return model_data

    @staticmethod
    def _get_trackers_for_training(
        trackers: List[TrackerWithCachedStates],
    ) -> List[TrackerWithCachedStates]:
        """Filters out the list of trackers which should not be used for training.

        Args:
            trackers: All trackers available for training.

        Returns:
            Trackers which should be used for training.
        """
        # By default, we train on all available trackers.
        return trackers

    def _prepare_for_training(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> Tuple[RasaModelData, np.ndarray]:
        """Prepares data to be fed into the model.

        Args:
            training_trackers: List of training trackers to be featurized.
            domain: Domain of the assistant.
            interpreter: NLU interpreter to be used for featurizing states.
            **kwargs: Any other arguments.

        Returns:
            Featurized data to be fed to the model and corresponding label ids.
        """
        training_trackers = self._get_trackers_for_training(training_trackers)
        # dealing with training data
        tracker_state_features, label_ids, entity_tags = self._featurize_for_training(
            training_trackers,
            domain,
            interpreter,
            bilou_tagging=self.config[BILOU_FLAG],
            **kwargs,
        )

        if not tracker_state_features:
            return RasaModelData(), label_ids

        self._label_data, encoded_all_labels = self._create_label_data(
            domain, interpreter
        )

        # extract actual training data to feed to model
        model_data = self._create_model_data(
            tracker_state_features, label_ids, entity_tags, encoded_all_labels
        )

        if self.config[ENTITY_RECOGNITION]:
            self._entity_tag_specs = self.featurizer.state_featurizer.entity_tag_specs

        # keep one example for persisting and loading
        self.data_example = model_data.first_data_example()

        return model_data, label_ids

    def run_training(
        self, model_data: RasaModelData, label_ids: Optional[np.ndarray] = None
    ) -> None:
        """Feeds the featurized training data to the model.

        Args:
            model_data: Featurized training data.
            label_ids: Label ids corresponding to the data points in `model_data`.
                These may or may not be used by the function depending
                on how the policy is trained.
        """
        if not self.finetune_mode:
            # This means the model wasn't loaded from a
            # previously trained model and hence needs
            # to be instantiated.
            self.model = self.model_class()(
                model_data.get_signature(),
                self.config,
                isinstance(self.featurizer, MaxHistoryTrackerFeaturizer),
                self._label_data,
                self._entity_tag_specs,
            )
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(self.config[LEARNING_RATE])
            )
        (
            data_generator,
            validation_data_generator,
        ) = rasa.utils.train_utils.create_data_generators(
            model_data,
            self.config[BATCH_SIZES],
            self.config[EPOCHS],
            self.config[BATCH_STRATEGY],
            self.config[EVAL_NUM_EXAMPLES],
            self.config[RANDOM_SEED],
        )
        callbacks = rasa.utils.train_utils.create_common_callbacks(
            self.config[EPOCHS],
            self.config[TENSORBOARD_LOG_DIR],
            self.config[TENSORBOARD_LOG_LEVEL],
            self.tmp_checkpoint_dir,
        )
        self.model.fit(
            data_generator,
            epochs=self.config[EPOCHS],
            validation_data=validation_data_generator,
            validation_freq=self.config[EVAL_NUM_EPOCHS],
            callbacks=callbacks,
            verbose=False,
            shuffle=False,  # we use custom shuffle inside data generator
        )

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        """Trains the policy on given training trackers.

        Args:
            training_trackers: List of training trackers to be used
                for training the model.
            domain: Domain of the assistant.
            interpreter: NLU Interpreter to be used for featurizing the states.
            **kwargs: Any other argument.
        """
        if not training_trackers:
            shared_io_utils.raise_warning(
                f"Skipping training of `{self.__class__.__name__}` "
                f"as no data was provided. You can exclude this "
                f"policy in the configuration "
                f"file to avoid this warning.",
                category=UserWarning,
            )
            return

        model_data, label_ids = self._prepare_for_training(
            training_trackers, domain, interpreter, **kwargs
        )

        if model_data.is_empty():
            shared_io_utils.raise_warning(
                f"Skipping training of `{self.__class__.__name__}` "
                f"as no data was provided. You can exclude this "
                f"policy in the configuration "
                f"file to avoid this warning.",
                category=UserWarning,
            )
            return

        self.run_training(model_data, label_ids)

    def _featurize_tracker_for_e2e(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
    ) -> List[List[Dict[Text, List["Features"]]]]:
        # construct two examples in the batch to be fed to the model -
        # one by featurizing last user text
        # and second - an optional one (see conditions below),
        # the first example in the constructed batch either does not contain user input
        # or uses intent or text based on whether TED is e2e only.
        tracker_state_features = self._featurize_for_prediction(
            tracker, domain, interpreter, use_text_for_last_user_input=self.only_e2e,
        )
        # the second - text, but only after user utterance and if not only e2e
        if (
            tracker.latest_action_name == ACTION_LISTEN_NAME
            and TEXT in self.fake_features
            and not self.only_e2e
        ):
            tracker_state_features += self._featurize_for_prediction(
                tracker, domain, interpreter, use_text_for_last_user_input=True,
            )
        return tracker_state_features

    def _pick_confidence(
        self, confidences: np.ndarray, similarities: np.ndarray, domain: Domain
    ) -> Tuple[np.ndarray, bool]:
        # the confidences and similarities have shape (batch-size x number of actions)
        # batch-size can only be 1 or 2;
        # in the case batch-size==2, the first example contain user intent as features,
        # the second - user text as features
        if confidences.shape[0] > 2:
            raise ValueError(
                "We cannot pick prediction from batches of size more than 2."
            )
        # we use heuristic to pick correct prediction
        if confidences.shape[0] == 2:
            # we use similarities to pick appropriate input,
            # since it seems to be more accurate measure,
            # policy is trained to maximize the similarity not the confidence
            non_e2e_action_name = domain.action_names_or_texts[
                np.argmax(confidences[0])
            ]
            logger.debug(f"User intent lead to '{non_e2e_action_name}'.")
            e2e_action_name = domain.action_names_or_texts[np.argmax(confidences[1])]
            logger.debug(f"User text lead to '{e2e_action_name}'.")
            if (
                np.max(confidences[1]) > self.config[E2E_CONFIDENCE_THRESHOLD]
                # TODO maybe compare confidences is better
                and np.max(similarities[1]) > np.max(similarities[0])
            ):
                logger.debug(f"TED predicted '{e2e_action_name}' based on user text.")
                return confidences[1], True

            logger.debug(f"TED predicted '{non_e2e_action_name}' based on user intent.")
            return confidences[0], False

        # by default the first example in a batch is the one to use for prediction
        predicted_action_name = domain.action_names_or_texts[np.argmax(confidences[0])]
        basis_for_prediction = "text" if self.only_e2e else "intent"
        logger.debug(
            f"TED predicted '{predicted_action_name}' "
            f"based on user {basis_for_prediction}."
        )
        return confidences[0], self.only_e2e

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take after seeing the tracker.

        Args:
            tracker: the :class:`rasa.core.trackers.DialogueStateTracker`
            domain: the :class:`rasa.shared.core.domain.Domain`
            interpreter: Interpreter which may be used by the policies to create
                additional features.

        Returns:
             The policy's prediction (e.g. the probabilities for the actions).
        """
        if self.model is None:
            return self._prediction(self._default_predictions(domain))

        # create model data from tracker
        tracker_state_features = self._featurize_tracker_for_e2e(
            tracker, domain, interpreter
        )
        model_data = self._create_model_data(tracker_state_features)
        outputs: Dict[Text, np.ndarray] = self.model.run_inference(model_data)

        # take the last prediction in the sequence
        similarities = outputs["similarities"][:, -1, :]
        confidences = outputs["scores"][:, -1, :]
        # take correct prediction from batch
        confidence, is_e2e_prediction = self._pick_confidence(
            confidences, similarities, domain
        )

        if self.config[RANKING_LENGTH] > 0 and self.config[MODEL_CONFIDENCE] == SOFTMAX:
            # TODO: This should be removed in 3.0 when softmax as
            #  model confidence and normalization is completely deprecated.
            _, confidence = rasa.utils.train_utils.rank_and_mask(
                confidence, self.config[RANKING_LENGTH], renormalize=True
            )

        optional_events = self._create_optional_event_for_entities(
            outputs, is_e2e_prediction, interpreter, tracker
        )

        return self._prediction(
            confidence.tolist(),
            is_end_to_end_prediction=is_e2e_prediction,
            optional_events=optional_events,
            diagnostic_data=outputs.get(DIAGNOSTIC_DATA),
        )

    def _create_optional_event_for_entities(
        self,
        prediction_output: Dict[Text, tf.Tensor],
        is_e2e_prediction: bool,
        interpreter: NaturalLanguageInterpreter,
        tracker: DialogueStateTracker,
    ) -> Optional[List[Event]]:
        if tracker.latest_action_name != ACTION_LISTEN_NAME or not is_e2e_prediction:
            # entities belong only to the last user message
            # and only if user text was used for prediction,
            # a user message always comes after action listen
            return None

        if not self.config[ENTITY_RECOGNITION]:
            # entity recognition is not turned on, no entities can be predicted
            return None

        # The batch dimension of entity prediction is not the same as batch size,
        # rather it is the number of last (if max history featurizer else all)
        # text inputs in the batch
        # therefore, in order to pick entities from the latest user message
        # we need to pick entities from the last batch dimension of entity prediction
        predicted_tags, confidence_values = rasa.utils.train_utils.entity_label_to_tags(
            prediction_output,
            self._entity_tag_specs,
            self.config[BILOU_FLAG],
            prediction_index=-1,
        )

        if ENTITY_ATTRIBUTE_TYPE not in predicted_tags:
            # no entities detected
            return None

        # entities belong to the last message of the tracker
        # convert the predicted tags to actual entities
        text = tracker.latest_message.text
        parsed_message = interpreter.featurize_message(Message(data={TEXT: text}))
        tokens = parsed_message.get(TOKENS_NAMES[TEXT])
        entities = EntityExtractor.convert_predictions_into_entities(
            text,
            tokens,
            predicted_tags,
            self.split_entities_config,
            confidences=confidence_values,
        )

        # add the extractor name
        for entity in entities:
            entity[EXTRACTOR] = "TEDPolicy"

        return [EntitiesAdded(entities)]

    def persist(self, path: Union[Text, Path]) -> None:
        """Persists the policy to a storage."""
        if self.model is None:
            logger.debug(
                "Method `persist(...)` was called without a trained model present. "
                "Nothing to persist then!"
            )
            return

        model_path = Path(path)
        model_filename = self._metadata_filename()
        tf_model_file = model_path / f"{model_filename}.tf_model"

        rasa.shared.utils.io.create_directory_for_file(tf_model_file)

        self.featurizer.persist(path)

        if self.config[CHECKPOINT_MODEL]:
            shutil.move(self.tmp_checkpoint_dir, model_path / "checkpoints")
        self.model.save(str(tf_model_file))

        self.persist_model_utilities(model_path)

    def persist_model_utilities(self, model_path: Path) -> None:
        """Persists model's utility attributes like model weights, etc.

        Args:
            model_path: Path where model is to be persisted
        """
        model_filename = self._metadata_filename()
        io_utils.json_pickle(
            model_path / f"{model_filename}.priority.pkl", self.priority
        )
        io_utils.pickle_dump(model_path / f"{model_filename}.meta.pkl", self.config)
        io_utils.pickle_dump(
            model_path / f"{model_filename}.data_example.pkl", self.data_example,
        )
        io_utils.pickle_dump(
            model_path / f"{model_filename}.fake_features.pkl", self.fake_features,
        )
        io_utils.pickle_dump(
            model_path / f"{model_filename}.label_data.pkl",
            dict(self._label_data.data),
        )
        entity_tag_specs = (
            [tag_spec._asdict() for tag_spec in self._entity_tag_specs]
            if self._entity_tag_specs
            else []
        )
        rasa.shared.utils.io.dump_obj_as_json_to_file(
            model_path / f"{model_filename}.entity_tag_specs.json", entity_tag_specs,
        )

    @classmethod
    def _load_model_utilities(cls, model_path: Path) -> Dict[Text, Any]:
        """Loads model's utility attributes.

        Args:
            model_path: Path where model is to be persisted.
        """
        tf_model_file = model_path / f"{cls._metadata_filename()}.tf_model"
        loaded_data = io_utils.pickle_load(
            model_path / f"{cls._metadata_filename()}.data_example.pkl"
        )
        label_data = io_utils.pickle_load(
            model_path / f"{cls._metadata_filename()}.label_data.pkl"
        )
        fake_features = io_utils.pickle_load(
            model_path / f"{cls._metadata_filename()}.fake_features.pkl"
        )
        label_data = RasaModelData(data=label_data)
        meta = io_utils.pickle_load(model_path / f"{cls._metadata_filename()}.meta.pkl")
        priority = io_utils.json_unpickle(
            model_path / f"{cls._metadata_filename()}.priority.pkl"
        )
        entity_tag_specs = rasa.shared.utils.io.read_json_file(
            model_path / f"{cls._metadata_filename()}.entity_tag_specs.json"
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

        return {
            "tf_model_file": tf_model_file,
            "loaded_data": loaded_data,
            "fake_features": fake_features,
            "label_data": label_data,
            "meta": meta,
            "priority": priority,
            "entity_tag_specs": entity_tag_specs,
        }

    @classmethod
    def load(
        cls,
        path: Union[Text, Path],
        should_finetune: bool = False,
        epoch_override: int = defaults[EPOCHS],
        **kwargs: Any,
    ) -> "TEDPolicy":
        """Loads a policy from the storage.

        Args:
            path: Path on disk where policy is persisted.
            should_finetune: Whether to load the policy for finetuning.
            epoch_override: Override the number of epochs in persisted
                configuration for further finetuning.
            **kwargs: Any other arguments

        Returns:
            Loaded policy

        Raises:
            `PolicyModelNotFound` if the model is not found in the supplied `path`.
        """
        model_path = Path(path)

        if not model_path.exists():
            logger.warning(
                f"Failed to load {cls.__class__.__name__} model. Path "
                f"'{model_path.absolute()}' doesn't exist."
            )
            return cls()

        featurizer = TrackerFeaturizer.load(path)

        if not (model_path / f"{cls._metadata_filename()}.data_example.pkl").is_file():
            return cls(featurizer=featurizer)

        model_utilities = cls._load_model_utilities(model_path)

        model_utilities["meta"] = cls._update_loaded_params(model_utilities["meta"])

        if should_finetune:
            model_utilities["meta"][EPOCHS] = epoch_override

        (
            model_data_example,
            predict_data_example,
        ) = cls._construct_model_initialization_data(model_utilities["loaded_data"])

        model = cls._load_tf_model(
            model_utilities,
            model_data_example,
            predict_data_example,
            featurizer,
            should_finetune,
        )

        return cls._load_policy_with_model(
            model, featurizer, model_utilities, should_finetune
        )

    @classmethod
    def _load_policy_with_model(
        cls,
        model: "TED",
        featurizer: TrackerFeaturizer,
        model_utilities: Dict[Text, Any],
        should_finetune: bool,
    ) -> "TEDPolicy":
        return cls(
            featurizer=featurizer,
            priority=model_utilities["priority"],
            model=model,
            fake_features=model_utilities["fake_features"],
            entity_tag_specs=model_utilities["entity_tag_specs"],
            should_finetune=should_finetune,
            **model_utilities["meta"],
        )

    @classmethod
    def _load_tf_model(
        cls,
        model_utilities: Dict[Text, Any],
        model_data_example: RasaModelData,
        predict_data_example: RasaModelData,
        featurizer: TrackerFeaturizer,
        should_finetune: bool,
    ) -> "TED":
        model = cls.model_class().load(
            str(model_utilities["tf_model_file"]),
            model_data_example,
            predict_data_example,
            data_signature=model_data_example.get_signature(),
            config=model_utilities["meta"],
            max_history_featurizer_is_used=isinstance(
                featurizer, MaxHistoryTrackerFeaturizer
            ),
            label_data=model_utilities["label_data"],
            entity_tag_specs=model_utilities["entity_tag_specs"],
            finetune_mode=should_finetune,
        )
        return model

    @classmethod
    def _construct_model_initialization_data(
        cls, loaded_data: Dict[Text, Dict[Text, List[FeatureArray]]]
    ) -> Tuple[RasaModelData, RasaModelData]:
        model_data_example = RasaModelData(
            label_key=LABEL_KEY, label_sub_key=LABEL_SUB_KEY, data=loaded_data
        )
        predict_data_example = RasaModelData(
            label_key=LABEL_KEY,
            label_sub_key=LABEL_SUB_KEY,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                if feature_name
                # we need to remove label features for prediction if they are present
                in PREDICTION_FEATURES
            },
        )
        return model_data_example, predict_data_example

    @classmethod
    def _update_loaded_params(cls, meta: Dict[Text, Any]) -> Dict[Text, Any]:
        meta = rasa.utils.train_utils.override_defaults(cls.defaults, meta)
        meta = rasa.utils.train_utils.update_confidence_type(meta)
        meta = rasa.utils.train_utils.update_similarity_type(meta)

        return meta
