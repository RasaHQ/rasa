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
from rasa.core.constants import DEFAULT_POLICY_PRIORITY, DIALOGUE
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
)
from rasa.core.policies.ted_policy import (
    LENGTH,
    STATE_LEVEL_FEATURES,
    FEATURES_TO_ENCODE,
)
from rasa.shared.nlu.constants import INTENT, TEXT, ENTITIES, ACTION_TEXT, ACTION_NAME
from rasa.shared.core.constants import ACTION_LISTEN_NAME, SLOTS, ACTIVE_LOOP
from rasa.shared.core.events import UserUttered
from rasa.utils.tensorflow.constants import HIDDEN_LAYERS_SIZES, CONCAT_DIMENSION

logger = logging.getLogger(__name__)

DIALOGUE_FEATURES = f"{DIALOGUE}_features"
LABEL_FEATURES = f"{LABEL}_features"
LABEL_IDS = f"{LABEL}_ids"
LABEL_KEY = LABEL
LABEL_SUB_KEY = "ids"

SAVE_MODEL_FILE_NAME = "intent_ted_policy"


class IntentTEDPolicy(TEDPolicy):
    """Transformer Embedding Dialogue (TED) Policy is described in
    https://arxiv.org/abs/1910.00486.
    This policy has a pre-defined architecture, which comprises the
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
        # Number of units in transformer
        TRANSFORMER_SIZE: 128,
        # Number of transformer layers
        NUM_TRANSFORMER_LAYERS: 1,
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
        # Strategy used whenc creating batches.
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
        # Whether to use augmented stories for threshold computation
        "use_augmentation_for_thresholds": True,
        # What to use for thresholds
        "use_probability_thresholds": True,
        # Whether to flag retrieval intents
        "ignore_retrieval_intents": True,
        # Other intents to ignore
        "ignore_intents": [],
        # revert unseen intent events
        "revert_unseen_intents": True,
    }

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = DEFAULT_POLICY_PRIORITY,
        max_history: Optional[int] = None,
        model: Optional[RasaModel] = None,
        zero_state_features: Optional[Dict[Text, List["Features"]]] = None,
        intent_thresholds: Dict[int, float] = None,
        **kwargs: Any,
    ) -> None:
        """Declare instance variables with default values."""
        super().__init__(
            featurizer, priority, max_history, model, zero_state_features, **kwargs
        )

        self.intent_thresholds = intent_thresholds
        self.use_augmentation_for_thresholds = self.config[
            "use_augmentation_for_thresholds"
        ]
        self.use_probability_thresholds = self.config["use_probability_thresholds"]
        self.ignore_intent_list = self.config["ignore_intents"]
        self.ignore_retrieval_intents = self.config["ignore_retrieval_intents"]

    @staticmethod
    def supported_data() -> SupportedData:
        return SupportedData.ML_AND_RULE_DATA

    @staticmethod
    def _standard_featurizer(max_history: Optional[int] = None) -> TrackerFeaturizer:
        return IntentMaxHistoryFeaturizer(
            IntentTokenizerSingleStateFeaturizer(), max_history=max_history
        )

    def _create_label_data(
        self, domain: Domain, interpreter: NaturalLanguageInterpreter
    ) -> Tuple[RasaModelData, List[Dict[Text, List["Features"]]]]:
        # encode all label_ids with policies' featurizer
        state_featurizer = self.featurizer.state_featurizer

        # TODO: Change TED also to do this before label data is created.
        #  Currently call to this is made inside featurize_trackers which masks
        #  the problem that labels may also need to be featurized from states prepared from domain.
        state_featurizer.prepare_from_domain(domain)

        encoded_all_labels = state_featurizer.encode_all_intents(domain, interpreter)
        # print("encoded all labels", encoded_all_labels)

        attribute_data, _ = convert_to_data_format(encoded_all_labels)

        # print("Attribute data after conversion", attribute_data)

        label_data = RasaModelData()
        label_data.add_data(attribute_data, key_prefix=f"{LABEL_KEY}_")

        label_ids = np.arange(len(domain.intents))
        label_data.add_features(
            LABEL_KEY,
            LABEL_SUB_KEY,
            [FeatureArray(np.expand_dims(label_ids, -1), number_of_dimensions=2)],
        )

        return label_data, encoded_all_labels

    def _separate_augmented_trackers(
        self, all_trackers: List[DialogueStateTracker]
    ) -> Tuple[List[DialogueStateTracker], List[DialogueStateTracker]]:
        """Separate

        Args:
            all_trackers:

        Returns:
        """

        augmented, non_augmented = [], []
        for tracker in all_trackers:
            augmented.append(tracker) if tracker.is_augmented else non_augmented.append(
                tracker
            )

        return augmented, non_augmented

    def train(
        self,
        all_trackers: List[DialogueStateTracker],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        """Train the policy on given training trackers."""

        # Filter non-augmented trackers from augmented ones.
        augmented_trackers, non_augmented_trackers = self._separate_augmented_trackers(
            all_trackers
        )

        # print(len(non_augmented_trackers), len(augmented_trackers))

        self._label_data, encoded_all_labels = self._create_label_data(
            domain, interpreter
        )

        # print("Label data signature", self._label_data.get_signature())

        model_train_data, train_label_ids = self._featurize_for_model(
            domain, encoded_all_labels, interpreter, non_augmented_trackers, **kwargs
        )

        if model_train_data.is_empty():
            logger.error(
                f"Can not train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the policy."
            )
            return

        # keep one example for persisting and loading
        self.data_example = model_train_data.first_data_example()

        self.model = IntentTED(
            model_train_data.get_signature(),
            self.config,
            isinstance(self.featurizer, MaxHistoryTrackerFeaturizer),
            self._label_data,
        )

        self.model.fit(
            model_train_data,
            self.config[EPOCHS],
            self.config[BATCH_SIZES],
            self.config[EVAL_NUM_EXAMPLES],
            self.config[EVAL_NUM_EPOCHS],
            batch_strategy=self.config[BATCH_STRATEGY],
        )

        if self.use_augmentation_for_thresholds:

            # print("Computing feats")
            # Featurize augmented trackers for threshold computation
            model_threshold_data, threshold_label_ids = self._featurize_for_model(
                domain, encoded_all_labels, interpreter, augmented_trackers, **kwargs
            )

            # print("Computing thresholds")
            self.intent_thresholds = self.model.compute_thresholds(
                model_threshold_data,
                threshold_label_ids,
                self.use_probability_thresholds,
            )
        else:
            self.intent_thresholds = self.model.compute_thresholds(
                model_train_data, train_label_ids, self.use_probability_thresholds
            )

    def _featurize_for_model(
        self, domain, encoded_all_labels, interpreter, trackers, **kwargs: Any
    ):
        # dealing with training data
        tracker_state_features, label_ids = self.featurize_for_training(
            trackers, domain, interpreter, **kwargs
        )
        # extract actual training data to feed to model
        model_data = self._create_model_data(
            tracker_state_features, label_ids, encoded_all_labels
        )
        return model_data, label_ids

    def persist(self, path: Union[Text, Path]) -> None:
        """Persists the policy to a storage."""

        if self.model is None:
            logger.debug(
                "Method `persist(...)` was called "
                "without a trained model present. "
                "Nothing to persist then!"
            )
            return

        model_path = Path(path)
        tf_model_file = model_path / f"{SAVE_MODEL_FILE_NAME}.tf_model"

        rasa.shared.utils.io.create_directory_for_file(tf_model_file)

        self.featurizer.persist(path)

        if self.model.checkpoint_model:
            self.model.copy_best(str(tf_model_file))
        else:
            self.model.save(str(tf_model_file))

        io_utils.json_pickle(
            model_path / f"{SAVE_MODEL_FILE_NAME}.priority.pkl", self.priority
        )
        io_utils.pickle_dump(
            model_path / f"{SAVE_MODEL_FILE_NAME}.meta.pkl", self.config
        )
        io_utils.pickle_dump(
            model_path / f"{SAVE_MODEL_FILE_NAME}.data_example.pkl", self.data_example
        )
        io_utils.pickle_dump(
            model_path / f"{SAVE_MODEL_FILE_NAME}.zero_state_features.pkl",
            self.zero_state_features,
        )
        io_utils.pickle_dump(
            model_path / f"{SAVE_MODEL_FILE_NAME}.label_data.pkl",
            dict(self._label_data.data),
        )

        io_utils.pickle_dump(
            model_path / f"{SAVE_MODEL_FILE_NAME}.intent_thresholds.pkl",
            dict(self.intent_thresholds),
        )

    @classmethod
    def load(cls, path: Union[Text, Path]) -> "TEDPolicy":
        """Loads a policy from the storage.
        **Needs to load its featurizer**
        """
        model_path = Path(path)

        if not model_path.exists():
            raise Exception(
                f"Failed to load TED policy model. Path "
                f"'{model_path.absolute()}' doesn't exist."
            )

        tf_model_file = model_path / f"{SAVE_MODEL_FILE_NAME}.tf_model"

        featurizer = TrackerFeaturizer.load(path)

        if not (model_path / f"{SAVE_MODEL_FILE_NAME}.data_example.pkl").is_file():
            return cls(featurizer=featurizer)

        loaded_data = io_utils.pickle_load(
            model_path / f"{SAVE_MODEL_FILE_NAME}.data_example.pkl"
        )
        label_data = io_utils.pickle_load(
            model_path / f"{SAVE_MODEL_FILE_NAME}.label_data.pkl"
        )
        zero_state_features = io_utils.pickle_load(
            model_path / f"{SAVE_MODEL_FILE_NAME}.zero_state_features.pkl"
        )
        intent_thresholds = io_utils.pickle_load(
            model_path / f"{SAVE_MODEL_FILE_NAME}.intent_thresholds.pkl"
        )
        label_data = RasaModelData(data=label_data)
        meta = io_utils.pickle_load(model_path / f"{SAVE_MODEL_FILE_NAME}.meta.pkl")
        priority = io_utils.json_unpickle(
            model_path / f"{SAVE_MODEL_FILE_NAME}.priority.pkl"
        )

        model_data_example = RasaModelData(
            label_key=LABEL_KEY, label_sub_key=LABEL_SUB_KEY, data=loaded_data
        )
        meta = train_utils.update_similarity_type(meta)

        model = IntentTED.load(
            str(tf_model_file),
            model_data_example,
            data_signature=model_data_example.get_signature(),
            config=meta,
            max_history_tracker_featurizer_used=isinstance(
                featurizer, MaxHistoryTrackerFeaturizer
            ),
            label_data=label_data,
        )

        # build the graph for prediction
        predict_data_example = RasaModelData(
            label_key=LABEL_KEY,
            label_sub_key=LABEL_SUB_KEY,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                if feature_name
                # we need to remove label features for prediction if they are present
                in STATE_LEVEL_FEATURES + FEATURES_TO_ENCODE + [DIALOGUE]
            },
        )
        model.build_for_predict(predict_data_example)

        return cls(
            featurizer=featurizer,
            priority=priority,
            model=model,
            zero_state_features=zero_state_features,
            intent_thresholds=intent_thresholds,
            **meta,
        )

    @staticmethod
    def _default_predictions(domain: Domain) -> List[float]:
        """Creates a list of zeros.

        Args:
            domain: the :class:`rasa.shared.core.domain.Domain`
        Returns:
            the list of the length of the number of actions
        """

        return [0.0] * len(domain.intents)

    def _should_check_for_intent(self, intent: Text, domain: Domain):

        if self.ignore_retrieval_intents and intent in domain.retrieval_intents:
            return False
        if domain.index_for_intent(intent) not in self.intent_thresholds:
            # This means the intent was never present in a story
            return False
        if intent in self.ignore_intent_list:
            return False

        return True

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> Tuple[List[float], Optional[bool]]:

        if self.model is None:
            return self._default_predictions(domain), False

        # create model data from tracker
        tracker_state_features = []
        if (
            INTENT in self.zero_state_features
            or not tracker.latest_action_name == ACTION_LISTEN_NAME
        ):
            # the first example in a batch uses intent
            # or current prediction is not after user utterance
            tracker_state_features += self.featurizer.create_state_features(
                [tracker], domain, interpreter, use_text_for_last_user_input=False
            )
        if (
            TEXT in self.zero_state_features
            and tracker.latest_action_name == ACTION_LISTEN_NAME
        ):
            # the second - text, but only after user utterance
            tracker_state_features += self.featurizer.create_state_features(
                [tracker], domain, interpreter, use_text_for_last_user_input=True
            )

        model_data = self._create_model_data(tracker_state_features)

        output = self.model.predict(model_data)

        # take the last prediction in the sequence
        confidences = output["intent_scores"].numpy()[:, -1, :]

        intent_confidences = {}
        for index, intent in enumerate(domain.intents):
            intent_confidences[intent] = confidences[0][index]

        # print("Confidences", intent_confidences)

        # Get the last intent prediction from tracker
        last_user_event: Optional[UserUttered] = tracker.get_last_event_for(UserUttered)
        if last_user_event:
            # If this is not the first intent
            query_intent = last_user_event.intent_name
            query_intent_id = domain.index_for_intent(query_intent)
            query_intent_prob = confidences[0][query_intent_id]

            if self._should_check_for_intent(query_intent, domain):
                # Check is only valid in this case, for all
                # other cases it means that this intent did not appear in stories.
                logger.debug(
                    f"User intent {query_intent} is probable with "
                    f"{query_intent_prob}, while threshold is {self.intent_thresholds[query_intent_id]}"
                )
                # print(
                #     f"User intent {query_intent} is probable with "
                #     f"{query_intent_prob}, while threshold is {self.intent_thresholds[query_intent_id]}"
                # )
                if query_intent_prob < self.intent_thresholds[query_intent_id]:

                    # Mark the corresponding user turn as interesting
                    last_user_event.set_as_not_probable()
                    # print("marked as improbable")

        return confidences.tolist(), False  # pytype: disable=bad-return-type


class IntentTED(TED):
    def _prepare_layers(self) -> None:

        # print("preparing layers")

        # print("data layers")
        for name in self.data_signature.keys():
            self._prepare_sparse_dense_layer_for(name, self.data_signature)
            self._prepare_encoding_layers(name)

        # print("label layers")
        for name in self.label_signature.keys():
            self._prepare_sparse_dense_layer_for(name, self.label_signature)
            self._prepare_encoding_layers(name)

        self._prepare_transformer_layer(
            DIALOGUE, self.config[DROP_RATE_DIALOGUE], self.config[DROP_RATE_ATTENTION]
        )

        self._prepare_embed_layers(DIALOGUE)
        self._prepare_embed_layers(LABEL)

        # print("layers uptil now - ", self._tf_layers.keys())

        self._prepare_multi_label_dot_product_loss(LABEL, self.config[SCALE_LOSS])

    def _create_all_labels_embed(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_label_ids = self.tf_label_data[LABEL_KEY][LABEL_SUB_KEY][0]

        all_labels_encoded = {
            key: self._encode_features_per_attribute(self.tf_label_data, key)
            for key in self.tf_label_data.keys()
            if key != LABEL_KEY
        }

        x = all_labels_encoded.pop(f"{LABEL_KEY}_{INTENT}")

        # additional sequence axis is artifact of our RasaModelData creation
        # TODO check whether this should be solved in data creation
        x = tf.squeeze(x, axis=1)

        all_labels_embed = self._tf_layers[f"embed.{LABEL}"](x)

        return all_label_ids, all_labels_embed

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)

        all_label_ids, all_labels_embed = self._create_all_labels_embed()

        batch_label_ids = tf_batch_data[LABEL_KEY][LABEL_SUB_KEY][
            0
        ]  # This can have multiple ids
        # print("Batch label ids shape", batch_label_ids.shape)

        batch_labels_embed = self._get_labels_embed(batch_label_ids, all_labels_embed)

        # print("Batch labels embed shape", batch_labels_embed.shape)

        dialogue_in = self._process_batch_data(tf_batch_data)
        dialogue_embed, dialogue_mask = self._emebed_dialogue(
            dialogue_in, tf_batch_data
        )
        dialogue_mask = tf.squeeze(dialogue_mask, axis=-1)

        loss, acc = self._tf_layers[f"loss.{LABEL}"](
            dialogue_embed,
            batch_labels_embed,
            batch_label_ids,
            all_labels_embed,
            all_label_ids,
            dialogue_mask,
        )

        self.action_loss.update_state(loss)
        self.action_acc.update_state(acc)

        return loss

    def batch_infer_during_training(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:

        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)

        _, all_labels_embed = self._create_all_labels_embed()

        dialogue_in = self._process_batch_data(tf_batch_data)
        dialogue_embed, dialogue_mask = self._emebed_dialogue(
            dialogue_in, tf_batch_data
        )
        dialogue_mask = tf.squeeze(dialogue_mask, axis=-1)

        sim_all = self._tf_layers[f"loss.{LABEL}"].sim(
            dialogue_embed[:, :, tf.newaxis, :],
            all_labels_embed[tf.newaxis, tf.newaxis, :, :],
            dialogue_mask,
        )

        scores = self._tf_layers[f"loss.{LABEL}"].confidence_from_sim(
            sim_all, self.config[SIMILARITY_TYPE]
        )

        return {"intent_scores": scores, "sim_all": sim_all}

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:

        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )

        if self.all_labels_embed is None:
            _, self.all_labels_embed = self._create_all_labels_embed()

        dialogue_in = self._process_batch_data(tf_batch_data)
        dialogue_embed, dialogue_mask = self._emebed_dialogue(
            dialogue_in, tf_batch_data
        )
        dialogue_mask = tf.squeeze(dialogue_mask, axis=-1)

        sim_all = self._tf_layers[f"loss.{LABEL}"].sim(
            dialogue_embed[:, :, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, tf.newaxis, :, :],
            dialogue_mask,
        )

        scores = self._tf_layers[f"loss.{LABEL}"].confidence_from_sim(
            sim_all, self.config[SIMILARITY_TYPE]
        )

        return {"intent_scores": scores, "sim_all": sim_all}

    def compute_thresholds(
        self, model_data: RasaModelData, label_ids, use_probability_thresholds: bool
    ):
        self._training = False

        scores = None
        sims = None

        # Todo: Make this use the config batch size
        batch_size = 64
        progress_bar = tqdm(
            range(0, label_ids.shape[0], batch_size),
            desc="Calculating Thresholds",
            disable=rasa.shared.utils.io.is_logging_disabled(),
        )
        for index in progress_bar:
            batch_in = model_data.prepare_batch(start=index, end=index + batch_size)
            batch_output = self.batch_infer_during_training(batch_in)
            if index == 0:
                scores = batch_output["intent_scores"].numpy()
                sims = batch_output["sim_all"].numpy()
            else:
                scores = np.vstack([scores, batch_output["intent_scores"].numpy()])
                sims = np.vstack([sims, batch_output["sim_all"].numpy()])
            # print(index)

        thresholds = {}

        # Collect all the probabilities for each label id
        for index, all_pos_labels in enumerate(label_ids):
            first_pos_label_id = all_pos_labels[0]

            if first_pos_label_id not in thresholds:
                thresholds[first_pos_label_id] = []

            thresholds[first_pos_label_id].append(
                scores[index, 0, first_pos_label_id]
                if use_probability_thresholds
                else sims[index, 0, first_pos_label_id]
            )

        for label_id in thresholds:
            thresholds[label_id] = (
                min(0.5, min(thresholds[label_id]))
                if use_probability_thresholds
                else min(0.0, min(thresholds[label_id]))
            )
            # thresholds[label_id] = min(thresholds[label_id])

        print(thresholds)

        return thresholds
