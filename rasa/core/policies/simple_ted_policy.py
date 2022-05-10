from __future__ import annotations

import contextlib
import logging
import typing
from collections import defaultdict
from pathlib import Path
from typing import Any, List, Optional, Text, Dict, Tuple, Union

import numpy as np
import scipy.sparse
import tensorflow as tf

import rasa.shared.utils.io
import rasa.utils.io as io_utils
from rasa.core.constants import DIALOGUE, POLICY_PRIORITY, DEFAULT_POLICY_PRIORITY
from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.core.featurizers.tracker_featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    MaxHistoryTrackerFeaturizer,
)
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.engine.graph import ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.core.constants import ACTIVE_LOOP, SLOTS
from rasa.shared.core.domain import Domain
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import ACTION_TEXT, ACTION_NAME, INTENT, TEXT, ENTITIES
from rasa.utils import train_utils
from rasa.utils.tensorflow import layers
from rasa.utils.tensorflow.constants import (
    SEQUENCE_LENGTH,
    LABEL,
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
    CONNECTION_DENSITY,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
    SOFTMAX,
    AUTO,
    BALANCED,
    TENSORBOARD_LOG_DIR,
    TENSORBOARD_LOG_LEVEL,
    CHECKPOINT_MODEL,
    ENCODING_DIMENSION,
    UNIDIRECTIONAL_ENCODER,
    SEQUENCE,
    SENTENCE,
    DENSE_DIMENSION,
    USE_GPU,
    CONSTRAIN_SIMILARITIES,
    MODEL_CONFIDENCE,
    CROSS_ENTROPY,
    LEARNING_RATE,
)
from rasa.utils.tensorflow.model_data import (
    RasaModelData,
    FeatureSignature,
    FeatureArray,
)
from rasa.utils.tensorflow.model_data_utils_21x import convert_to_data_format
from rasa.utils.tensorflow.models import RasaModel, TransformerRasaModel
from rasa.utils.tensorflow.transformer import TransformerEncoder

if typing.TYPE_CHECKING:
    from rasa.shared.nlu.training_data.features import Features


logger = logging.getLogger(__name__)

MASK = "mask"
LABEL_KEY = LABEL
LABEL_SUB_KEY = "ids"
LENGTH = "length"
POSSIBLE_FEATURE_TYPES = [SEQUENCE, SENTENCE]
FEATURES_TO_ENCODE = [INTENT, TEXT, ACTION_NAME, ACTION_TEXT]
LABEL_FEATURES_TO_ENCODE = [f"{LABEL}_{ACTION_NAME}", f"{LABEL}_{ACTION_TEXT}"]
STATE_LEVEL_FEATURES = [ENTITIES, SLOTS, ACTIVE_LOOP]

SAVE_MODEL_FILE_NAME = "ted_policy"


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.POLICY_WITH_END_TO_END_SUPPORT, is_trainable=True
)
class SimpleTEDPolicy(Policy):
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

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the default config (see parent class for full docstring)."""
        # please make sure to update the docs when changing a default parameter
        return {
            # ## Architecture of the used neural network
            # Hidden layer sizes for layers before the dialogue and label embedding layers.
            # The number of hidden layers is equal to the length of the corresponding
            # list.
            # TODO add 2 parallel NNs: transformer for text and ffnn for names
            DENSE_DIMENSION: 20,
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
            MAX_RELATIVE_POSITION: 5,
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
            CONNECTION_DENSITY: 0.2,
            # if 'True' applies sigmoid on all similarity terms and adds
            # it to the loss function to ensure that similarity values are
            # approximately bounded. Used inside cross-entropy loss only.
            CONSTRAIN_SIMILARITIES: False,
            # Model confidence to be returned during inference. Currently, the only
            # possible value is `softmax`.
            MODEL_CONFIDENCE: SOFTMAX,
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
            # Determines the importance of policies, higher values take precedence
            POLICY_PRIORITY: DEFAULT_POLICY_PRIORITY,
            USE_GPU: True,
        }

    @staticmethod
    def _standard_featurizer(max_history: Optional[int] = None) -> TrackerFeaturizer:
        return MaxHistoryTrackerFeaturizer(
            SingleStateFeaturizer(), max_history=max_history
        )

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        model: Optional[RasaModel] = None,
        featurizer: Optional[TrackerFeaturizer] = None,
        zero_state_features: Optional[Dict[Text, List["Features"]]] = None,
    ) -> None:
        """Declares instance variables with default values."""
        super().__init__(
            config, model_storage, resource, execution_context, featurizer=featurizer
        )

        if isinstance(featurizer, FullDialogueTrackerFeaturizer):
            self.is_full_dialogue_featurizer_used = True
        else:
            self.is_full_dialogue_featurizer_used = False

        self._load_params(config)

        self.model = model

        self.zero_state_features = zero_state_features or defaultdict(list)

        self._label_data: Optional[RasaModelData] = None
        self.data_example: Optional[Dict[Text, List[np.ndarray]]] = None

        self.tmp_checkpoint_dir = None
        if self.config[CHECKPOINT_MODEL]:
            self.tmp_checkpoint_dir = Path(rasa.utils.io.create_temporary_directory())

    def _load_params(self, config: Dict[Text, Any]) -> None:
        new_config = rasa.utils.train_utils.check_core_deprecated_options(config)
        self.config = new_config
        self._auto_update_configuration()

    def _auto_update_configuration(self) -> None:
        """Takes care of deprecations and compatibility of parameters."""
        self.config = rasa.utils.train_utils.update_confidence_type(self.config)
        rasa.utils.train_utils.validate_configuration_settings(self.config)
        self.config = rasa.utils.train_utils.update_similarity_type(self.config)
        self.config = rasa.utils.train_utils.update_evaluation_parameters(self.config)

    def _create_label_data(
        self,
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization],
    ) -> Tuple[RasaModelData, List[Dict[Text, List["Features"]]]]:
        # encode all label_ids with policies' featurizer
        state_featurizer = self.featurizer.state_featurizer
        encoded_all_labels = (
            state_featurizer.encode_all_labels(domain, precomputations)
            if state_featurizer is not None
            else []
        )

        attribute_data, _ = convert_to_data_format(encoded_all_labels)

        label_data = RasaModelData()
        label_data.add_data(attribute_data, key_prefix=f"{LABEL_KEY}_")

        label_ids = np.arange(domain.num_actions)
        label_data.add_features(
            LABEL_KEY,
            LABEL_SUB_KEY,
            [FeatureArray(np.expand_dims(label_ids, -1), number_of_dimensions=2)],
        )

        label_data.sort()

        return label_data, encoded_all_labels

    def _create_model_data(
        self,
        tracker_state_features: List[List[Dict[Text, List[Features]]]],
        label_ids: Optional[np.ndarray] = None,
        encoded_all_labels: Optional[List[Dict[Text, List[Features]]]] = None,
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

            attribute_data, self.zero_state_features = convert_to_data_format(
                tracker_state_features,
            )

        else:
            # method is called during prediction
            attribute_data, _ = convert_to_data_format(
                tracker_state_features,
                self.zero_state_features,
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

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization] = None,
        **kwargs: Any,
    ) -> Resource:
        """Train the policy on given training trackers."""

        if not training_trackers:
            logger.error(
                f"Can not train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the policy."
            )
            return

        # dealing with training data
        tracker_state_features, label_ids, _ = self._featurize_for_training(
            training_trackers, domain, precomputations=precomputations, **kwargs
        )

        self._label_data, encoded_all_labels = self._create_label_data(
            domain, precomputations
        )

        # extract actual training data to feed to model
        model_data = self._create_model_data(
            tracker_state_features=tracker_state_features,
            label_ids=label_ids,
            encoded_all_labels=encoded_all_labels,
        )

        if model_data.is_empty():
            logger.error(
                f"Can not train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the policy."
            )
            return

        # keep one example for persisting and loading
        self.data_example = model_data.first_data_example()

        self.model = SimpleTED(
            model_data.get_signature(),
            self.config,
            isinstance(self.featurizer, MaxHistoryTrackerFeaturizer),
            self._label_data,
        )

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(self.config[LEARNING_RATE]),
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

        self.persist()

        return self._resource

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        precomputations: Optional[MessageContainerForCoreFeaturization] = None,
        **kwargs: Any,
    ) -> PolicyPrediction:
        """Predicts the next action the bot should take.

        See the docstring of the parent class `Policy` for more information.
        """
        if self.model is None:
            return self._prediction(self._default_predictions(domain))

        # create model data from tracker
        tracker_state_features = self.featurizer.create_state_features(
            [tracker], domain, precomputations
        )
        model_data = self._create_model_data(
            tracker_state_features=tracker_state_features
        )
        output: Dict[Text, np.ndarray] = self.model.run_inference(model_data)

        # remove batch dimension and take the last prediction in the sequence
        confidence = output["action_scores"][0, -1, :]

        if self.config[LOSS_TYPE] == SOFTMAX and self.config[RANKING_LENGTH] > 0:
            confidence = train_utils.normalize(confidence, self.config[RANKING_LENGTH])

        return self._prediction(confidence.tolist())

    def persist(self) -> None:
        """Persists the policy to a storage."""
        if self.model is None:
            logger.debug(
                "Method `persist(...)` was called without a trained model present. "
                "Nothing to persist then!"
            )
            return

        with self._model_storage.write_to(self._resource) as model_path:
            model_filename = self._metadata_filename()
            tf_model_file = model_path / f"{model_filename}.tf_model"

            rasa.shared.utils.io.create_directory_for_file(tf_model_file)

            self.featurizer.persist(model_path)

            if self.config[CHECKPOINT_MODEL] and self.tmp_checkpoint_dir:
                self.model.load_weights(self.tmp_checkpoint_dir / "checkpoint.tf_model")
                # Save an empty file to flag that this model has been
                # produced using checkpointing
                checkpoint_marker = model_path / f"{model_filename}.from_checkpoint.pkl"
                checkpoint_marker.touch()

            self.model.save(str(tf_model_file))

            self.persist_model_utilities(model_path)

    @classmethod
    def _metadata_filename(cls) -> Optional[Text]:
        return "simple_ted_policy"

    def persist_model_utilities(self, model_path: Path) -> None:
        """Persists model's utility attributes like model weights, etc.

        Args:
            model_path: Path where model is to be persisted
        """
        model_filename = self._metadata_filename()
        rasa.utils.io.json_pickle(
            model_path / f"{model_filename}.priority.pkl", self.priority
        )
        rasa.utils.io.pickle_dump(
            model_path / f"{model_filename}.meta.pkl", self.config
        )
        rasa.utils.io.pickle_dump(
            model_path / f"{model_filename}.data_example.pkl", self.data_example
        )
        rasa.utils.io.pickle_dump(
            model_path / f"{model_filename}.fake_features.pkl", self.zero_state_features
        )
        rasa.utils.io.pickle_dump(
            model_path / f"{model_filename}.label_data.pkl",
            dict(self._label_data.data) if self._label_data is not None else {},
        )

    @classmethod
    def load(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
        **kwargs: Any,
    ) -> SimpleTEDPolicy:
        """Loads a policy from the storage (see parent class for full docstring)."""
        try:
            with model_storage.read_from(resource) as model_path:
                return cls._load(
                    model_path, config, model_storage, resource, execution_context
                )
        except ValueError:
            logger.debug(
                f"Failed to load {cls.__class__.__name__} from model storage. Resource "
                f"'{resource.name}' doesn't exist."
            )
            return cls(config, model_storage, resource, execution_context)

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
                in STATE_LEVEL_FEATURES + FEATURES_TO_ENCODE + [DIALOGUE]
            },
        )
        return model_data_example, predict_data_example

    @classmethod
    def _load(
        cls,
        model_path: Path,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> SimpleTEDPolicy:
        featurizer = TrackerFeaturizer.load(model_path)

        if not (model_path / f"{cls._metadata_filename()}.data_example.pkl").is_file():
            return cls(
                config,
                model_storage,
                resource,
                execution_context,
                featurizer=featurizer,
            )

        model_utilities = cls._load_model_utilities(model_path)

        config = cls._update_loaded_params(config)
        # if execution_context.is_finetuning and EPOCH_OVERRIDE in config:
        #     config[EPOCHS] = config.get(EPOCH_OVERRIDE)

        (
            model_data_example,
            predict_data_example,
        ) = cls._construct_model_initialization_data(model_utilities["loaded_data"])

        with (contextlib.nullcontext() if config["use_gpu"] else tf.device("/cpu:0")):
            model = SimpleTED.load(
                str(model_utilities["tf_model_file"]),
                model_data_example,
                predict_data_example,
                data_signature=model_data_example.get_signature(),
                config=model_utilities["model_config"],
                label_data=model_utilities["label_data"],
                # TODO: save this and check whether used featurizer is correct...
                max_history_tracker_featurizer_used=isinstance(
                    featurizer, MaxHistoryTrackerFeaturizer
                ),
                finetune_mode=execution_context.is_finetuning,
            )

        return cls(
            config,
            model_storage,
            resource,
            execution_context,
            model=model,
            featurizer=featurizer,
            zero_state_features=model_utilities["zero_state_features"],
        )

    @classmethod
    def _load_model_utilities(cls, model_path: Path) -> Dict[Text, Any]:
        """Loads model's utility attributes.

        Args:
            model_path: Path where model is to be persisted.
        """
        tf_model_file = model_path / f"{cls._metadata_filename()}.tf_model"
        loaded_data = rasa.utils.io.pickle_load(
            model_path / f"{cls._metadata_filename()}.data_example.pkl"
        )
        label_data = rasa.utils.io.pickle_load(
            model_path / f"{cls._metadata_filename()}.label_data.pkl"
        )
        fake_features = rasa.utils.io.pickle_load(
            model_path / f"{cls._metadata_filename()}.fake_features.pkl"
        )
        label_data = RasaModelData(data=label_data)
        priority = rasa.utils.io.json_unpickle(
            model_path / f"{cls._metadata_filename()}.priority.pkl"
        )
        model_config = rasa.utils.io.pickle_load(
            model_path / f"{cls._metadata_filename()}.meta.pkl"
        )

        return {
            "tf_model_file": tf_model_file,
            "loaded_data": loaded_data,
            "zero_state_features": fake_features,
            "label_data": label_data,
            "priority": priority,
            "model_config": model_config,
        }

    @classmethod
    def _update_loaded_params(cls, meta: Dict[Text, Any]) -> Dict[Text, Any]:
        meta = rasa.utils.train_utils.update_confidence_type(meta)
        meta = rasa.utils.train_utils.update_similarity_type(meta)

        return meta


class SimpleTED(TransformerRasaModel):
    def __init__(
        self,
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
        config: Dict[Text, Any],
        max_history_tracker_featurizer_used: bool,
        label_data: RasaModelData,
    ) -> None:
        super().__init__("SimpleTED", config, data_signature, label_data)

        self.max_history_tracker_featurizer_used = max_history_tracker_featurizer_used

        self.predict_data_signature = {
            feature_name: features
            for feature_name, features in data_signature.items()
            if feature_name in STATE_LEVEL_FEATURES + FEATURES_TO_ENCODE + [DIALOGUE]
        }

        # metrics
        self.action_loss = tf.keras.metrics.Mean(name="loss")
        self.action_acc = tf.keras.metrics.Mean(name="acc")
        self.metrics_to_log += ["loss", "acc"]

        # needed for efficient prediction
        self.all_labels_embed: Optional[tf.Tensor] = None

        self._prepare_layers()

    def _check_data(self) -> None:
        if not any(key in [INTENT, TEXT] for key in self.data_signature.keys()):
            raise ValueError(
                f"No user features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )

        if not any(
            key in [ACTION_NAME, ACTION_TEXT] for key in self.data_signature.keys()
        ):
            raise ValueError(
                f"No action features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if LABEL not in self.data_signature:
            raise ValueError(
                f"No label features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )

    def _prepare_layers(self) -> None:
        for name in self.data_signature.keys():
            self._prepare_sparse_dense_layer_for(name, self.data_signature)
            self._prepare_encoding_layers(name)

        for name in self.label_signature.keys():
            self._prepare_sparse_dense_layer_for(name, self.label_signature)
            self._prepare_encoding_layers(name)

        self._prepare_transformer_layer(
            DIALOGUE, self.config[DROP_RATE_DIALOGUE], self.config[DROP_RATE_ATTENTION]
        )

        self._prepare_embed_layers(DIALOGUE)
        self._prepare_embed_layers(LABEL)

        self._prepare_dot_product_loss(LABEL, self.config[SCALE_LOSS])

    def _prepare_sparse_dense_layer_for(
        self, name: Text, signature: Dict[Text, Dict[Text, List[FeatureSignature]]]
    ) -> None:
        """Prepare the sparse dense layer for the given attribute name. It is used to
        combine the sparse and dense features of the attribute at the beginning of
        the model.

        Args:
            name: the attribute name
            signature: data signature
        """
        for feature_type in POSSIBLE_FEATURE_TYPES:
            if name not in signature or feature_type not in signature[name]:
                # features for feature type are not present
                continue

            self._prepare_sparse_dense_dropout_layers(
                f"{name}_{feature_type}", self.config[DROP_RATE]
            )

            # use the same configurable dense dimension for all sparse features
            self._prepare_sparse_dense_layers(
                signature[name][feature_type],
                f"{name}_{feature_type}",
                self.config[DENSE_DIMENSION],
            )

    def _prepare_sparse_dense_dropout_layers(
        self, name: Text, drop_rate: float
    ) -> None:
        self._tf_layers[f"sparse_input_dropout.{name}"] = layers.SparseDropout(
            rate=drop_rate
        )
        self._tf_layers[f"dense_input_dropout.{name}"] = tf.keras.layers.Dropout(
            rate=drop_rate
        )

    def _prepare_sparse_dense_layers(
        self, data_signature: List[FeatureSignature], name: Text, dense_dim: int
    ) -> None:
        is_sparse_options = set(
            signature_item.is_sparse for signature_item in data_signature
        )

        if True in is_sparse_options:  # at least one sparse
            self._tf_layers[f"sparse_to_dense.{name}"] = layers.DenseForSparse(
                units=dense_dim,
                reg_lambda=self.config[REGULARIZATION_CONSTANT],
                name=name,
            )
            if False not in is_sparse_options:  # only sparse
                # create dense labels for the input to use in negative sampling
                self._tf_layers[f"sparse_to_dense_ids.{name}"] = layers.DenseForSparse(
                    units=2, trainable=False, name=f"sparse_to_dense_ids.{name}"
                )

    def _prepare_encoding_layers(self, name: Text) -> None:
        """Create ffnn layer for given attribute name. The layer is used just before
        all dialogue features are combined.

        Args:
            name: attribute name
        """
        feature_type = SENTENCE
        # create encoding layers only for the features which should be encoded;
        if name not in FEATURES_TO_ENCODE + LABEL_FEATURES_TO_ENCODE:
            return
        # check that there are SENTENCE features for the attribute name in data
        if name in FEATURES_TO_ENCODE and feature_type not in self.data_signature[name]:
            return
        #  same for label_data
        if (
            name in LABEL_FEATURES_TO_ENCODE
            and feature_type not in self.label_signature[name]
        ):
            return

        self._prepare_ffnn_layer(
            f"{name}_{feature_type}",
            [self.config[ENCODING_DIMENSION]],
            self.config[DROP_RATE_DIALOGUE],
        )

    def _prepare_transformer_layer(
        self,
        name: Text,
        drop_rate: float,
        drop_rate_attention: float,
        prefix: Text = "transformer",
    ):
        if self.config[NUM_TRANSFORMER_LAYERS] > 0:
            self._tf_layers[f"{prefix}.{name}"] = TransformerEncoder(
                num_layers=self.config[NUM_TRANSFORMER_LAYERS],
                units=self.config[TRANSFORMER_SIZE],
                num_heads=self.config[NUM_HEADS],
                filter_units=self.config[TRANSFORMER_SIZE] * 4,
                reg_lambda=self.config[REGULARIZATION_CONSTANT],
                dropout_rate=drop_rate,
                attention_dropout_rate=drop_rate_attention,
                density=self.config[CONNECTION_DENSITY],
                unidirectional=self.config[UNIDIRECTIONAL_ENCODER],
                use_key_relative_position=self.config[KEY_RELATIVE_ATTENTION],
                use_value_relative_position=self.config[VALUE_RELATIVE_ATTENTION],
                max_relative_position=self.config[MAX_RELATIVE_POSITION],
                name=f"{name}_encoder",
            )
        else:
            # create lambda so that it can be used later without the check
            self._tf_layers[f"{prefix}.{name}"] = lambda x, mask, training: x

    def _create_all_labels_embed(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_label_ids = self.tf_label_data[LABEL_KEY][LABEL_SUB_KEY][0]

        all_labels_encoded = {
            key: self._encode_features_per_attribute(self.tf_label_data, key)
            for key in self.tf_label_data.keys()
            if key != LABEL_KEY
        }

        if (
            all_labels_encoded.get(f"{LABEL_KEY}_{ACTION_TEXT}") is not None
            and all_labels_encoded.get(f"{LABEL_KEY}_{ACTION_NAME}") is not None
        ):
            x = all_labels_encoded.pop(
                f"{LABEL_KEY}_{ACTION_TEXT}"
            ) + all_labels_encoded.pop(f"{LABEL_KEY}_{ACTION_NAME}")
        elif all_labels_encoded.get(f"{LABEL_KEY}_{ACTION_TEXT}") is not None:
            x = all_labels_encoded.pop(f"{LABEL_KEY}_{ACTION_TEXT}")
        else:
            x = all_labels_encoded.pop(f"{LABEL_KEY}_{ACTION_NAME}")

        # additional sequence axis is artifact of our RasaModelData creation
        # TODO check whether this should be solved in data creation
        x = tf.squeeze(x, axis=1)

        all_labels_embed = self._tf_layers[f"embed.{LABEL}"](x)

        return all_label_ids, all_labels_embed

    def _emebed_dialogue(
        self, dialogue_in: tf.Tensor, sequence_lengths: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create dialogue level embedding and mask."""

        mask = self._compute_mask(sequence_lengths)

        dialogue_transformed, _ = self._tf_layers[f"transformer.{DIALOGUE}"](
            dialogue_in, 1 - mask, self._training
        )
        dialogue_transformed = tf.nn.gelu(
            dialogue_transformed, approximate=True, name="gelu"
        )

        if self.max_history_tracker_featurizer_used:
            # pick last vector if max history featurizer is used
            dialogue_transformed = tf.expand_dims(
                self._last_token(dialogue_transformed, sequence_lengths), 1
            )
            mask = tf.expand_dims(self._last_token(mask, sequence_lengths), 1)

        dialogue_embed = self._tf_layers[f"embed.{DIALOGUE}"](dialogue_transformed)

        return dialogue_embed, mask

    @staticmethod
    def _compute_mask(sequence_lengths: tf.Tensor) -> tf.Tensor:
        mask = tf.sequence_mask(
            sequence_lengths, dtype=tf.float32, name="sequence-mask"
        )
        # explicitly add last dimension to mask
        # to track correctly dynamic sequences
        return tf.expand_dims(mask, -1)

    def _encode_features_per_attribute(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> Optional[tf.Tensor]:
        """
        Encodes features for a given attribute
        Args:
            tf_batch_data: dictionary mapping every attribute to its features and masks
            attribute: the attribute we will encode features for (e.g., ACTION_NAME, INTENT)
        Returns:
            A tensor combining  all features for `attribute`
        """

        if not tf_batch_data[attribute]:
            return None

        attribute_mask = tf_batch_data[attribute][MASK][0]
        # TODO transformer has to be used to process sequence features

        attribute_features = self._combine_sparse_dense_features(
            tf_batch_data[attribute][SENTENCE],
            f"{attribute}_{SENTENCE}",
            mask=attribute_mask,
        )

        if attribute in FEATURES_TO_ENCODE + LABEL_FEATURES_TO_ENCODE:
            attribute_features = self._tf_layers[f"ffnn.{attribute}_{SENTENCE}"](
                attribute_features
            )

        return attribute_features * attribute_mask

    def _combine_sparse_dense_features(
        self,
        features: List[Union[np.ndarray, tf.Tensor, tf.SparseTensor]],
        name: Text,
        mask: Optional[tf.Tensor] = None,
        sparse_dropout: bool = False,
        dense_dropout: bool = False,
    ) -> Optional[tf.Tensor]:

        if not features:
            return None

        dense_features = []

        for f in features:
            if isinstance(f, tf.SparseTensor) or isinstance(f, scipy.sparse.spmatrix):
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

        if mask is None:
            return tf.concat(dense_features, axis=-1)

        return tf.concat(dense_features, axis=-1) * mask

    def _process_batch_data(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]]
    ) -> tf.Tensor:
        """Encodes batch dat.

        Ccombines intent and text and action name and action text if both are present.

        Args:
            tf_batch_data: dictionary mapping every attribute to its features and masks
        Returns:
             Tensor: encoding of all features in the batch, combined;
        """
        # encode each attribute present in tf_batch_data
        batch_encoded = {
            key: self._encode_features_per_attribute(tf_batch_data, key)
            for key in tf_batch_data.keys()
            if LABEL_KEY not in key and DIALOGUE not in key
        }
        # if both action text and action name are present, combine them; otherwise, return the one which is present

        if (
            batch_encoded.get(ACTION_TEXT) is not None
            and batch_encoded.get(ACTION_NAME) is not None
        ):
            batch_action = batch_encoded.pop(ACTION_TEXT) + batch_encoded.pop(
                ACTION_NAME
            )
        elif batch_encoded.get(ACTION_TEXT) is not None:
            batch_action = batch_encoded.pop(ACTION_TEXT)
        else:
            batch_action = batch_encoded.pop(ACTION_NAME)
        # same for user input
        if (
            batch_encoded.get(INTENT) is not None
            and batch_encoded.get(TEXT) is not None
        ):
            batch_user = batch_encoded.pop(INTENT) + batch_encoded.pop(TEXT)
        elif batch_encoded.get(TEXT) is not None:
            batch_user = batch_encoded.pop(TEXT)
        else:
            batch_user = batch_encoded.pop(INTENT)

        batch_features = [batch_user, batch_action]
        # once we have user input and previous action,
        # add all other attributes (SLOTS, ACTIVE_LOOP, etc.) to batch_features;
        for key in batch_encoded.keys():
            batch_features.append(batch_encoded.get(key))

        batch_features = tf.concat(batch_features, axis=-1)

        return batch_features

    @staticmethod
    def _get_labels_embed(
        label_ids: tf.Tensor, all_labels_embed: tf.Tensor
    ) -> tf.Tensor:
        # instead of processing labels again, gather embeddings from
        # all_labels_embed using label ids

        indices = tf.cast(label_ids[:, :, 0], tf.int32)
        labels_embed = tf.gather(all_labels_embed, indices)

        return labels_embed

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        tf_batch_data = self.batch_to_model_data_format(batch_in, self.data_signature)

        dialogue_lengths = tf.cast(tf_batch_data[DIALOGUE][LENGTH][0], tf.int32)

        all_label_ids, all_labels_embed = self._create_all_labels_embed()

        label_ids = tf_batch_data[LABEL_KEY][LABEL_SUB_KEY][0]
        labels_embed = self._get_labels_embed(label_ids, all_labels_embed)

        dialogue_in = self._process_batch_data(tf_batch_data)
        dialogue_embed, dialogue_mask = self._emebed_dialogue(
            dialogue_in, dialogue_lengths
        )
        dialogue_mask = tf.squeeze(dialogue_mask, axis=-1)

        loss, acc = self._tf_layers[f"loss.{LABEL}"](
            dialogue_embed,
            labels_embed,
            label_ids,
            all_labels_embed,
            all_label_ids,
            dialogue_mask,
        )

        self.action_loss.update_state(loss)
        self.action_acc.update_state(acc)

        return loss

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        tf_batch_data = self.batch_to_model_data_format(
            batch_in, self.predict_data_signature
        )

        dialogue_lengths = tf.cast(tf_batch_data[DIALOGUE][LENGTH][0], tf.int32)

        if self.all_labels_embed is None:
            _, self.all_labels_embed = self._create_all_labels_embed()

        dialogue_in = self._process_batch_data(tf_batch_data)
        dialogue_embed, dialogue_mask = self._emebed_dialogue(
            dialogue_in, dialogue_lengths
        )
        dialogue_mask = tf.squeeze(dialogue_mask, axis=-1)

        _, confidences = self._tf_layers[
            f"loss" f".{LABEL}"
        ].get_similarities_and_confidences_from_embeddings(
            input_embeddings=dialogue_embed[:, :, tf.newaxis, :],
            label_embeddings=self.all_labels_embed[tf.newaxis, tf.newaxis, :, :],
            mask=dialogue_mask,
        )

        return {"action_scores": confidences}
