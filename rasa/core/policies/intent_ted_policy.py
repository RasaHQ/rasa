import copy
import logging
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from typing import Any, List, Optional, Text, Dict, Tuple, Union

import rasa.utils.io as io_utils
from rasa.shared.core.domain import Domain
from rasa.core.featurizers.tracker_featurizers import (
    TrackerFeaturizer,
    MaxHistoryTrackerFeaturizer,
    IntentMaxHistoryFeaturizer,
)
from rasa.core.featurizers.single_state_featurizer import (
    IntentTokenizerSingleStateFeaturizer,
)
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.core.policies.policy import Policy, SupportedData
from rasa.core.policies.ted_policy import TEDPolicy, TED
from rasa.core.constants import UNLIKELY_INTENT_POLICY_PRIORITY, DIALOGUE
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.utils import train_utils
import rasa.shared.utils.io
from rasa.utils.tensorflow.model_data import (
    RasaModelData,
    FeatureSignature,
    FeatureArray,
)
from rasa.utils.tensorflow.model_data_utils import convert_to_data_format
from rasa.utils.tensorflow.models import RasaModel
from rasa.core.policies.policy import PolicyPrediction

from rasa.utils.tensorflow.constants import (
    LABEL,
    DENSE_DIMENSION,
    ENCODING_DIMENSION,
    UNIDIRECTIONAL_ENCODER,
    TRANSFORMER_SIZE,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
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
    WEIGHT_SPARSITY,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    SOFTMAX,
    AUTO,
    BALANCED,
    TENSORBOARD_LOG_DIR,
    TENSORBOARD_LOG_LEVEL,
    CHECKPOINT_MODEL,
    FEATURIZERS,
    ENTITY_RECOGNITION,
    IGNORE_INTENTS_LIST,
)
from rasa.core.policies.ted_policy import (
    STATE_LEVEL_FEATURES,
    SENTENCE_FEATURES_TO_ENCODE,
    SEQUENCE_FEATURES_TO_ENCODE,
    SEQUENCE_LENGTH,
    SEQUENCE,
)
from rasa.shared.nlu.constants import INTENT, TEXT, ENTITIES, ACTION_TEXT, ACTION_NAME
from rasa.shared.core.constants import ACTION_LISTEN_NAME, SLOTS, ACTIVE_LOOP
from rasa.shared.core.events import UserUttered
from rasa.utils.tensorflow.constants import HIDDEN_LAYERS_SIZES, CONCAT_DIMENSION
from rasa.utils.tensorflow import layers
from rasa.utils.tensorflow.model_data import (
    RasaModelData,
    FeatureSignature,
    FeatureArray,
    Data,
)
from rasa.shared.core.constants import ACTION_UNLIKELY_INTENT_NAME

logger = logging.getLogger(__name__)

DIALOGUE_FEATURES = f"{DIALOGUE}_features"
LABEL_FEATURES = f"{LABEL}_features"
LABEL_IDS = f"{LABEL}_ids"
LABEL_KEY = LABEL
LABEL_SUB_KEY = "ids"

SAVE_MODEL_FILE_NAME = "intent_ted_policy"


class IntentTEDPolicy(TEDPolicy):
    """
    TODO: add description
    """

    SUPPORTS_ONLINE_TRAINING = True

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # Hidden layer sizes for layers before the embedding layers for user message
        # and labels.
        # The number of hidden layers is equal to the length of the corresponding
        # list.
        HIDDEN_LAYERS_SIZES: {TEXT: [], ACTION_TEXT: []},
        DENSE_DIMENSION: {
            TEXT: 128,
            ACTION_TEXT: 128,
            ENTITIES: 128,
            SLOTS: 128,
            ACTIVE_LOOP: 128,
            f"{LABEL}_{ACTION_TEXT}": 20,
            INTENT: 20,
            ACTION_NAME: 20,
            f"{LABEL}_{ACTION_NAME}": 20,
            f"{LABEL}_{INTENT}": 20,
        },
        CONCAT_DIMENSION: {TEXT: 128, ACTION_TEXT: 128},
        ENCODING_DIMENSION: 50,
        # Number of units in sequence transformer
        TRANSFORMER_SIZE: 128,
        # Number of sequence transformer layers
        NUM_TRANSFORMER_LAYERS: 1,
        # Number of units in dialogue transformer
        f"{DIALOGUE}_{TRANSFORMER_SIZE}": 128,
        # Number of dialogue transformer layers
        f"{DIALOGUE}_{NUM_TRANSFORMER_LAYERS}": 1,
        # Number of attention heads in transformer
        NUM_HEADS: 4,
        # If 'True' use key relative embeddings in attention
        KEY_RELATIVE_ATTENTION: False,
        # If 'True' use value relative embeddings in attention
        VALUE_RELATIVE_ATTENTION: False,
        # Max position for relative embeddings
        MAX_RELATIVE_POSITION: None,
        # Use a unidirectional or bidirectional encoder.
        UNIDIRECTIONAL_ENCODER: True,
        # ## Training parameters
        # Initial and final batch sizes:
        # Batch size will be linearly increased for each epoch.
        BATCH_SIZES: [64, 256],
        # Strategy used when creating batches.
        # Can be either 'sequence' or 'balanced'.
        BATCH_STRATEGY: BALANCED,
        # Number of epochs to train
        EPOCHS: 1,
        # Set random seed to any 'int' to get reproducible results
        RANDOM_SEED: None,
        # ## Parameters for embeddings
        # Dimension size of embedding vectors
        EMBEDDING_DIMENSION: 20,
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
        MAX_NEG_SIM: -0.2,
        # If 'True' the algorithm only minimizes maximum similarity over
        # incorrect intent labels, used only if 'loss_type' is set to 'margin'.
        USE_MAX_NEG_SIM: True,
        # If 'True' scale loss inverse proportionally to the confidence
        # of the correct prediction
        SCALE_LOSS: False,
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
        DROP_RATE_ATTENTION: 0,
        # Sparsity of the weights in dense layers
        WEIGHT_SPARSITY: 0.8,
        # ## Evaluation parameters
        # How often calculate validation accuracy.
        # Small values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EPOCHS: 20,
        # How many examples to use for hold out validation set
        # Large values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EXAMPLES: 0,
        # If you want to use tensorboard to visualize training and validation metrics,
        # set this option to a valid output directory.
        TENSORBOARD_LOG_DIR: None,
        # Define when training metrics for tensorboard should be logged.
        # Either after every epoch or for every training step.
        # Valid values: 'epoch' and 'minibatch'
        TENSORBOARD_LOG_LEVEL: "epoch",
        # Perform model checkpointing
        CHECKPOINT_MODEL: False,
        # Specify what features to use as sequence and sentence features.
        # By default all features in the pipeline are used.
        FEATURIZERS: [],
        # If set to true, entities are predicted in user utterances.
        ENTITY_RECOGNITION: False,
        # List of intents to ignore
        IGNORE_INTENTS_LIST: [],
    }

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = UNLIKELY_INTENT_POLICY_PRIORITY,
        max_history: Optional[int] = None,
        model: Optional[RasaModel] = None,
        fake_features: Optional[Dict[Text, List["Features"]]] = None,
        entity_tag_specs: Optional[List[EntityTagSpec]] = None,
        should_finetune: bool = False,
        intent_thresholds: Dict[int, float] = None,
        all_labels: List[Text] = None,
        **kwargs: Any,
    ) -> None:
        """Declare instance variables with default values."""
        super().__init__(
            featurizer,
            priority,
            max_history,
            model,
            fake_features,
            entity_tag_specs,
            should_finetune,
            **kwargs,
        )

        self._all_labels = all_labels
        self.intent_thresholds = intent_thresholds
        self.ignore_intent_list = self.config[IGNORE_INTENTS_LIST]

        # Set all invalid configuration parameters
        self.config[ENTITY_RECOGNITION] = False

    @staticmethod
    def supported_data() -> SupportedData:
        return SupportedData.ML_DATA

    @staticmethod
    def _standard_featurizer(max_history: Optional[int] = None) -> TrackerFeaturizer:
        return IntentMaxHistoryFeaturizer(
            IntentTokenizerSingleStateFeaturizer(), max_history=max_history
        )

    def _assemble_label_data(
        self, attribute_data: Data, domain: Domain
    ) -> RasaModelData:
        label_data = RasaModelData()
        label_data.add_data(attribute_data, key_prefix=f"{LABEL_KEY}_")
        label_data.add_lengths(
            f"{LABEL}_{INTENT}", SEQUENCE_LENGTH, f"{LABEL}_{INTENT}", SEQUENCE,
        )
        label_ids = np.arange(len(domain.intents))
        label_data.add_features(
            LABEL_KEY,
            LABEL_SUB_KEY,
            [FeatureArray(np.expand_dims(label_ids, -1), number_of_dimensions=2)],
        )
        return label_data

    def run_post_training_procedures(
        self, model_data: RasaModelData, label_ids: np.ndarray
    ) -> None:

        self.intent_thresholds = self.model.compute_thresholds(model_data, label_ids)

    def _collect_action_metadata(
        self, domain: Domain, similarities: np.array
    ) -> Dict[Text, Dict[Text, float]]:
        """

        Args:
            domain:
            similarities:

        Returns:

        """
        metadata = {}
        for intent in domain.intents:
            intent_index = domain.intents.index(intent)
            metadata[intent] = {
                "score": similarities[0][intent_index],
                "threshold": self.intent_thresholds[intent_index],
            }
        return metadata

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
        tracker_state_features = self._featurize_for_prediction(
            tracker, domain, interpreter
        )

        model_data = self._create_model_data(tracker_state_features)
        output = self.model.run_inference(model_data)

        # take the last prediction in the sequence
        similarities = output["similarities"][:, -1, :]

        # Check for unlikely intent
        is_unlikely_intent = self._check_unlikely_intent(domain, similarities, tracker)

        confidences = list(np.zeros(domain.num_actions))

        if is_unlikely_intent:
            confidences[domain.index_for_action(ACTION_UNLIKELY_INTENT_NAME)] = 1.0

        return self._prediction(
            confidences,
            action_metadata=self._collect_action_metadata(domain, similarities),
        )

    def _should_check_for_intent(self, intent: Text, domain: Domain) -> bool:

        if domain.intents.index(intent) not in self.intent_thresholds:
            # This means the intent was never present in a story
            return False
        if intent in self.config[IGNORE_INTENTS_LIST]:
            return False

        return True

    def _check_unlikely_intent(
        self, domain: Domain, similarities: np.array, tracker: DialogueStateTracker
    ) -> bool:
        """Check if the latest user event is probable according to IntentTED predictions.

        If the similarity prediction for the intent of
        latest user event is lower than the threshold
        calculated for that intent during training, the
        corresponding user intent is unlikely.

        Args:
            domain: Domain of the assistant.
            similarities: Predicted similarities for all labels.
            tracker: Current conversation tracker
        """
        predicted_intent_scores = {
            index: similarities[0][index] for index, intent in enumerate(domain.intents)
        }
        sorted_intent_scores = sorted(
            [
                (intent_label, score)
                for intent_label, score in predicted_intent_scores.items()
            ],
            key=lambda x: x[1],
        )
        # Get the last intent prediction from tracker
        last_user_event: Optional[UserUttered] = tracker.get_last_event_for(UserUttered)
        if last_user_event:
            query_label = last_user_event.intent_name
            query_label_id = domain.intents.index(query_label)
            query_label_score = similarities[0][query_label_id]

            logger.debug(f"Querying for intent {query_label}")

            if self._should_check_for_intent(query_label, domain):

                logger.debug(
                    f"Score for user intent {query_label} likely to occur here is "
                    f"{query_label_score}, while threshold is {self.intent_thresholds[query_label_id]}"
                )
                logger.debug(
                    f"Top 5 intents(in ascending order) that are likely here are: {sorted_intent_scores[-5:]}"
                )

                # If score for query intent is below threshold and
                # the query intent is not the top likely intent
                if (
                    query_label_score < self.intent_thresholds[query_label_id]
                    and query_label_id != sorted_intent_scores[-1][0]
                ):
                    return True


class IntentTED(TED):
    def _prepare_label_classification_layers(self, predictor_attribute: Text) -> None:
        """Prepares layers & loss for the final label prediction step."""
        self._prepare_embed_layers(predictor_attribute)
        self._prepare_embed_layers(LABEL)

        self._prepare_dot_product_loss(
            LABEL, self.config[SCALE_LOSS], loss_layer=layers.MultiLabelDotProductLoss
        )

    def compute_thresholds(
        self, model_data: RasaModelData, label_ids: np.ndarray
    ) -> Dict[int, float]:
        """

        Args:
            model_data:
            label_ids:

        Returns:

        """
        self._training = False

        batch_size = (
            self.config[BATCH_SIZES]
            if isinstance(self.config[BATCH_SIZES], int)
            else self.config[BATCH_SIZES][0]
        )
        outputs = self.model.run_inference(model_data, batch_size=batch_size)

        thresholds = {}

        # Collect all the probabilities for each label id
        for index, all_pos_labels in enumerate(label_ids):
            first_pos_label_id = all_pos_labels[0]

            if first_pos_label_id not in thresholds:
                thresholds[first_pos_label_id] = []

            thresholds[first_pos_label_id].append(
                outputs["similarities"][index, 0, first_pos_label_id]
            )

        # Pick the minimum of all similarities as the threshold
        for label_id in thresholds:
            thresholds[label_id] = min(thresholds[label_id])

        return thresholds
