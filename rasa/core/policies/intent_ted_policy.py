import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Any, List, Optional, Text, Dict, Type, Union, TYPE_CHECKING
from collections import defaultdict

from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.constants import SLOTS, ACTIVE_LOOP, ACTION_UNLIKELY_INTENT_NAME
from rasa.shared.core.events import UserUttered
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.shared.nlu.constants import (
    INTENT,
    TEXT,
    ENTITIES,
    ACTION_NAME,
)
from rasa.nlu.extractors.extractor import EntityTagSpec
from rasa.core.featurizers.tracker_featurizers import (
    TrackerFeaturizer,
    IntentMaxHistoryTrackerFeaturizer,
)
from rasa.core.featurizers.single_state_featurizer import (
    IntentTokenizerSingleStateFeaturizer,
)
from rasa.core.constants import UNLIKELY_INTENT_POLICY_PRIORITY, DIALOGUE
from rasa.core.policies.policy import PolicyPrediction
from rasa.core.policies.ted_policy import (
    LABEL_KEY,
    LABEL_SUB_KEY,
    TEDPolicy,
    TED,
    SEQUENCE_LENGTH,
    SEQUENCE,
    PREDICTION_FEATURES,
)
from rasa.utils import train_utils
from rasa.utils.tensorflow.models import RasaModel
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
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    EMBEDDING_DIMENSION,
    DROP_RATE_DIALOGUE,
    DROP_RATE_LABEL,
    DROP_RATE,
    DROP_RATE_ATTENTION,
    CONNECTION_DENSITY,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    INNER,
    BALANCED,
    TENSORBOARD_LOG_DIR,
    TENSORBOARD_LOG_LEVEL,
    CHECKPOINT_MODEL,
    FEATURIZERS,
    ENTITY_RECOGNITION,
    IGNORE_INTENTS_LIST,
    BILOU_FLAG,
    LEARNING_RATE,
    CROSS_ENTROPY,
    SPARSE_INPUT_DROPOUT,
    DENSE_INPUT_DROPOUT,
    MASKED_LM,
    HIDDEN_LAYERS_SIZES,
    CONCAT_DIMENSION,
)
from rasa.utils.tensorflow import layers
from rasa.utils.tensorflow.model_data import (
    RasaModelData,
    FeatureArray,
    Data,
)

import rasa.utils.io as io_utils
from rasa.core.exceptions import RasaCoreException

if TYPE_CHECKING:
    from rasa.shared.nlu.training_data.features import Features


logger = logging.getLogger(__name__)

SAVE_MODEL_FILE_NAME = "intent_ted_policy"


class IntentTEDPolicy(TEDPolicy):
    """`IntentTEDPolicy` has the same model architecture as `TEDPolicy`.

    The difference is at a task level.
    Instead of predicting the next probable action, this policy
    predicts whether the last predicted intent is a likely intent
    according to the training stories and conversation context.
    """

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # ## Architecture of the used neural network
        # Hidden layer sizes for layers before the embedding layers for user message
        # and labels.
        # The number of hidden layers is equal to the length of the corresponding list.
        HIDDEN_LAYERS_SIZES: {TEXT: []},
        # Dense dimension to use for sparse features.
        DENSE_DIMENSION: {
            TEXT: 128,
            INTENT: 20,
            ACTION_NAME: 20,
            ENTITIES: 20,
            SLOTS: 20,
            ACTIVE_LOOP: 20,
            f"{LABEL}_{INTENT}": 20,
        },
        # Default dimension to use for concatenating sequence and sentence features.
        CONCAT_DIMENSION: {TEXT: 128},
        # Dimension size of embedding vectors before the dialogue transformer encoder.
        ENCODING_DIMENSION: 50,
        # Number of units in transformer encoders
        TRANSFORMER_SIZE: {TEXT: 128, DIALOGUE: 128,},
        # Number of layers in transformer encoders
        NUM_TRANSFORMER_LAYERS: {TEXT: 1, DIALOGUE: 1,},
        # Number of attention heads in transformer
        NUM_HEADS: 4,
        # If 'True' use key relative embeddings in attention
        KEY_RELATIVE_ATTENTION: False,
        # If 'True' use value relative embeddings in attention
        VALUE_RELATIVE_ATTENTION: False,
        # Max position for relative embeddings
        MAX_RELATIVE_POSITION: None,
        # Use a unidirectional or bidirectional encoder
        # for `text`, `action_text`, and `label_action_text`.
        UNIDIRECTIONAL_ENCODER: False,
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
        # Initial learning rate for the optimizer
        LEARNING_RATE: 0.001,
        # ## Parameters for embeddings
        # Dimension size of embedding vectors
        EMBEDDING_DIMENSION: 20,
        # The number of incorrect labels. The algorithm will minimize
        # their similarity to the user input during training.
        NUM_NEG: 20,
        # Number of intents to store in predicted action metadata.
        RANKING_LENGTH: 10,
        # If 'True' scale loss inverse proportionally to the confidence
        # of the correct prediction
        SCALE_LOSS: True,
        # ## Regularization parameters
        # The scale of regularization
        REGULARIZATION_CONSTANT: 0.001,
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
        # Valid values: 'epoch' and 'batch'
        TENSORBOARD_LOG_LEVEL: "epoch",
        # Perform model checkpointing
        CHECKPOINT_MODEL: False,
        # Specify what features to use as sequence and sentence features.
        # By default all features in the pipeline are used.
        FEATURIZERS: [],
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
        label_thresholds: Dict[int, float] = None,
        **kwargs: Any,
    ) -> None:
        """Declares instance variables with default values."""
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

        self.label_thresholds = label_thresholds
        self.ignore_intent_list = self.config[IGNORE_INTENTS_LIST]

        # Set all invalid / non configurable parameters
        self.config[ENTITY_RECOGNITION] = False
        self.config[BILOU_FLAG] = False
        self.config[SIMILARITY_TYPE] = INNER
        self.config[LOSS_TYPE] = CROSS_ENTROPY

    @staticmethod
    def _standard_featurizer(max_history: Optional[int] = None) -> TrackerFeaturizer:
        return IntentMaxHistoryTrackerFeaturizer(
            IntentTokenizerSingleStateFeaturizer(), max_history=max_history
        )

    @staticmethod
    def model_class() -> Type["IntentTED"]:
        """Gets the class of the model architecture to be used by the policy.

        Returns:
            Required class.
        """
        return IntentTED

    def _auto_update_configuration(self) -> None:
        self.config = train_utils.update_evaluation_parameters(self.config)
        self.config = train_utils.update_deprecated_sparsity_to_density(self.config)

    @classmethod
    def _metadata_filename(cls) -> Optional[Text]:
        return SAVE_MODEL_FILE_NAME

    def _assemble_label_data(
        self, attribute_data: Data, domain: Domain
    ) -> RasaModelData:
        """Constructs data regarding labels to be fed to the model.

        The resultant model data should contain the keys `label_intent`, `label`.
        `label_intent` will contain the sequence, sentence and mask features
        for all intent labels and `label` will contain the numerical label ids.

        Args:
            attribute_data: Feature data for all intent labels.
            domain: Domain of the assistant.

        Returns:
            Features of labels ready to be fed to the model.
        """
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

    @staticmethod
    def _prepare_data_for_prediction(model_data: RasaModelData) -> RasaModelData:
        """Transforms training model data to data usable for making model predictions.

        Transformation involves filtering out all features which
        are not useful at prediction time. This is important
        because the prediction signature will not contain these
        attributes and hence prediction will break.

        Args:
            model_data: Data used during model training.

        Returns:
            Transformed data usable for making predictions.
        """
        filtered_data: Dict[Text, Dict[Text, Any]] = {
            key: features
            for key, features in model_data.data.items()
            if key in PREDICTION_FEATURES
        }
        return RasaModelData(data=filtered_data)

    def calculate_label_thresholds_post_training(
        self, model_data: RasaModelData, label_ids: np.ndarray
    ) -> None:
        """Calculates label thresholds for prediction of `action_unlikely_intent`.

        Args:
            model_data: Data used for training the model.
            label_ids: Numerical IDs of labels for each data point used during training.
        """
        # `model_data` contains data attributes like `label` which were
        # used during training. These attributes are not present in
        # the `predict_data_signature`. Prediction through the model
        # will break if `model_data` is passed as it is through the model.
        # Hence, we first filter out the attributes inside `model_data`
        # to keep only those which should be present during prediction.
        model_prediction_data = self._prepare_data_for_prediction(model_data)
        self.label_thresholds = self.model.compute_thresholds(
            model_prediction_data, label_ids
        )

    def run_training(
        self, model_data: RasaModelData, label_ids: Optional[np.ndarray] = None
    ) -> None:
        """Feeds the featurized training data to the model.

        Args:
            model_data: Featurized training data.
            label_ids: Label ids corresponding to the data points in `model_data`.

        Raises:
            `RasaCoreException` if `label_ids` is None as it's needed for
                running post training procedures.
        """
        if label_ids is None:
            raise RasaCoreException(
                f"Incorrect usage of `run_training` "
                f"method of `{self.__class__.__name__}`."
                f"`label_ids` cannot be left to `None`."
            )
        super().run_training(model_data, label_ids)
        self.calculate_label_thresholds_post_training(model_data, label_ids)

    def _collect_action_metadata(
        self, domain: Domain, similarities: np.array
    ) -> Dict[Text, Dict[Text, float]]:
        """Adds any metadata to be attached to the predicted action.

        Similarities for all intents and their thresholds are attached as metadata.

        Args:
            domain: Domain of the assistant.
            similarities: Predicted similarities for each intent.

        Returns:
            Metadata to be attached.
        """
        metadata = {}
        for intent_index, intent in enumerate(domain.intents):
            if intent_index in self.label_thresholds:
                metadata[intent] = {
                    "score": similarities[0][intent_index],
                    "threshold": self.label_thresholds[intent_index],
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
            tracker: Tracker containing past conversation events.
            domain: Domain of the assistant.
            interpreter: Interpreter which may be used by the policies to create
                additional features.

        Returns:
             The policy's prediction (e.g. the probabilities for the actions).
        """
        if self.model is None:
            return self._prediction(self._default_predictions(domain))

        # Prediction through the policy is skipped if:
        # 1. Last event in the tracker was not of type `UserUttered`.
        # This is to prevent the ensemble of policies from being stuck
        # in a loop.
        # 2. If the tracker does not contain any event of type `UserUttered` till now.
        if not tracker.get_last_event_for(UserUttered) or (
            tracker.events and not isinstance(tracker.events[-1], UserUttered)
        ):
            logger.debug(
                f"Skipping predictions for {self.__class__.__name__} "
                f"as the last event in tracker is not of type `UserUttered`."
            )
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
        query_intent = tracker.get_last_event_for(UserUttered).intent_name
        is_unlikely_intent = self._check_unlikely_intent(
            domain, similarities, query_intent
        )

        confidences = list(np.zeros(domain.num_actions))

        if is_unlikely_intent:
            confidences[domain.index_for_action(ACTION_UNLIKELY_INTENT_NAME)] = 1.0

        return self._prediction(
            confidences,
            action_metadata=self._collect_action_metadata(domain, similarities),
        )

    def _should_check_for_intent(self, intent: Text, domain: Domain) -> bool:
        """Checks if the intent should raise `action_unlikely_intent`.

        Args:
            intent: Intent to be queried.
            domain: Domain of the assistant.

        Returns:
            Whether intent should raise `action_unlikely_intent` or not.
        """
        if domain.intents.index(intent) not in self.label_thresholds:
            # This means the intent was never present in a story
            logger.debug(
                f"Query intent index {domain.intents.index(intent)} not "
                f"found in label thresholds - {self.label_thresholds}."
                f"Check for `action_unlikely_intent` prediction will be skipped."
            )
            return False
        if intent in self.config[IGNORE_INTENTS_LIST]:
            logger.debug(
                f"Query intent {intent} found in {IGNORE_INTENTS_LIST}. "
                f"Check for `action_unlikely_intent` prediction will be skipped."
            )
            return False

        return True

    def _check_unlikely_intent(
        self, domain: Domain, similarities: np.array, query_intent: Text
    ) -> bool:
        """Checks if the query intent is probable according to model's predictions.

        If the similarity prediction for the intent of
        is lower than the threshold calculated for that
        intent during training, the corresponding user
        intent is unlikely.

        Args:
            domain: Domain of the assistant.
            similarities: Predicted similarities for all intents.
            query_intent: Intent to be queried.

        Returns:
            Whether query intent is likely or not.
        """
        logger.debug(f"Querying for intent {query_intent}")

        if not self._should_check_for_intent(query_intent, domain):
            return False

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
        query_intent_id = domain.intents.index(query_intent)
        query_intent_similarity = similarities[0][query_intent_id]

        logger.debug(
            f"Score for intent `{query_intent}` is "
            f"{query_intent_similarity}, while "
            f"threshold is {self.label_thresholds[query_intent_id]}"
        )
        logger.debug(
            f"Top 5 intents(in ascending order) that "
            f"are likely here are: {sorted_intent_scores[-5:]}"
        )

        # If score for query intent is below threshold and
        # the query intent is not the top likely intent
        if (
            query_intent_similarity < self.label_thresholds[query_intent_id]
            and query_intent_id != sorted_intent_scores[-1][0]
        ):
            logger.debug(
                f"Intent {query_intent}-{query_intent_id} unlikely to occur here."
            )
            return True

        return False

    def persist_model_utilities(self, model_path: Path) -> None:
        """Persists model's utility attributes like model weights, etc.

        Args:
            model_path: Path where model is to be persisted
        """
        super().persist_model_utilities(model_path)
        io_utils.pickle_dump(
            model_path / f"{self._metadata_filename()}.label_thresholds.pkl",
            self.label_thresholds,
        )

    @classmethod
    def _load_model_utilities(cls, model_path: Path) -> Dict[Text, Any]:
        """Loads model's utility attributes.

        Args:
            model_path: Path where model is to be persisted.
        """
        model_utilties = super()._load_model_utilities(model_path)
        label_thresholds = io_utils.pickle_load(
            model_path / f"{cls._metadata_filename()}.label_thresholds.pkl"
        )
        model_utilties.update({"label_thresholds": label_thresholds})
        return model_utilties

    @classmethod
    def _update_loaded_params(cls, meta: Dict[Text, Any]) -> Dict[Text, Any]:
        meta = train_utils.override_defaults(cls.defaults, meta)
        return meta

    @classmethod
    def _load_policy_with_model(
        cls,
        model: "IntentTED",
        featurizer: TrackerFeaturizer,
        model_utilities: Dict[Text, Any],
        should_finetune: bool,
    ) -> "IntentTEDPolicy":
        return cls(
            featurizer=featurizer,
            priority=model_utilities["priority"],
            model=model,
            fake_features=model_utilities["fake_features"],
            entity_tag_specs=model_utilities["entity_tag_specs"],
            should_finetune=should_finetune,
            label_thresholds=model_utilities["label_thresholds"],
            **model_utilities["meta"],
        )


class IntentTED(TED):
    """Follows TED's model architecture from https://arxiv.org/abs/1910.00486.

    However, it has been re-purposed to predict multiple
    labels (intents) instead of a single label (action).
    """

    def _prepare_label_classification_layers(self, predictor_attribute: Text) -> None:
        """Prepares layers & loss for the final label prediction step."""
        self._prepare_embed_layers(predictor_attribute)
        self._prepare_embed_layers(LABEL)
        self._prepare_dot_product_loss(LABEL, self.config[SCALE_LOSS])

    def _prepare_dot_product_loss(
        self, name: Text, scale_loss: bool, prefix: Text = "loss",
    ) -> None:
        self._tf_layers[f"{prefix}.{name}"] = self.dot_product_loss_layer(
            self.config[NUM_NEG],
            scale_loss,
            similarity_type=self.config[SIMILARITY_TYPE],
        )

    @property
    def dot_product_loss_layer(self) -> tf.keras.layers.Layer:
        """Returns the dot-product loss layer to use.

        Multiple intents can be valid simultaneously, so `IntentTED` uses the
        `MultiLabelDotProductLoss`.

        Returns:
            The loss layer that is used by `_prepare_dot_product_loss`.
        """
        return layers.MultiLabelDotProductLoss

    @staticmethod
    def _get_labels_embed(
        label_ids: tf.Tensor, all_labels_embed: tf.Tensor
    ) -> tf.Tensor:
        # instead of processing labels again, gather embeddings from
        # all_labels_embed using label ids

        indices = tf.cast(label_ids[:, :, 0], tf.int32)

        # Find padding indices. They should have a value -1
        padding_indices = tf.where(tf.equal(indices, -1))

        # Create a tensor of ones which will serve as updates to original `indices`
        updates_to_indices = tf.ones((tf.shape(padding_indices)[0]), dtype=tf.int32)

        # Add the tensor of 1s to indices with padding.
        # So, effectively -1s become 0. This is fine because
        # we don't change the original label indices but only
        # make them 'compatible' for the `tf.gather` op below.
        indices_to_gather = tf.cast(
            tf.tensor_scatter_nd_add(indices, padding_indices, updates_to_indices),
            tf.int32,
        )

        labels_embed = tf.gather(all_labels_embed, indices_to_gather)

        return labels_embed

    def compute_thresholds(
        self, model_data: RasaModelData, label_ids: np.ndarray
    ) -> Dict[int, float]:
        """Computes prediction thresholds for each intent.

        These thresholds are used at inference time to predict
        whether a query intent is likely or not.

        Args:
            model_data: Data used during model training.
            label_ids: Numerical IDs of labels corresponding to data points used during
            training.

        Returns:
            Computed thresholds for each intent label present in `label_ids`.
        """
        self._training = False

        batch_size = (
            self.config[BATCH_SIZES]
            if isinstance(self.config[BATCH_SIZES], int)
            else self.config[BATCH_SIZES][0]
        )
        outputs = self.run_inference(
            model_data, batch_size=batch_size, output_keys_expected=["similarities"]
        )

        # Collect scores across all data points
        label_id_scores = self._collect_label_id_similarities_from_outputs(
            label_ids, outputs
        )

        return self._pick_threshold_from_similarities(label_id_scores)

    @staticmethod
    def _pick_threshold_from_similarities(
        label_id_similarities: Dict[int, List[float]]
    ) -> Dict[int, float]:
        """Computes threshold for predicted similarities.

        The threshold for an intent is computed as the minimum
        of all similarities predicted for that particular intent.

        Args:
            label_id_similarities: Similarities predicted for each label/

        Returns:
            Computed thresholds for each intent label.
        """
        return {
            label_id: min(label_id_similarities[label_id])
            for label_id in label_id_similarities
        }

    @staticmethod
    def _collect_label_id_similarities_from_outputs(
        label_ids: np.ndarray, outputs: Dict[Text, Union[np.ndarray, Dict[Text, Any]]]
    ) -> Dict[int, List[float]]:
        """Collects similarities predicted for each label id.

        Args:
            label_ids: Numerical IDs of labels for each data point used during training.
            outputs: Model's predictions for each data point.

        Returns:
            Similarities grouped by intent label id.
        """
        label_id_scores: Dict[int, List[float]] = defaultdict(list)
        for index, all_pos_labels in enumerate(label_ids):

            # If a data point (tracker) has multiple label ids
            # assigned to it, the tracker featurizer distributes
            # those multiple label ids across multiple data points,
            # one for each label id. Hence, here when we iterate over
            # all data points, we consider only the first label id.
            # Other positive label ids will be taken care of as part of
            # some other data point, where the input tracker is the same
            # as this one but first positive label id is different.
            # This prevents over-counting across label ids.
            first_pos_label_id = all_pos_labels[0]
            label_id_scores[first_pos_label_id].append(
                outputs["similarities"][index, 0, first_pos_label_id]
            )
        return label_id_scores
