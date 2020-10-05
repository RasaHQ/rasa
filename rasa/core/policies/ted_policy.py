import copy
import logging
from pathlib import Path
from collections import defaultdict

import numpy as np
import rasa.shared.utils.io
import tensorflow as tf
import tensorflow_addons as tfa
import typing
from typing import Any, List, Optional, Text, Dict, Tuple, Union

import rasa.utils.io as io_utils
from rasa.shared.core.domain import Domain
from rasa.core.featurizers.tracker_featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    MaxHistoryTrackerFeaturizer,
)
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.shared.nlu.constants import ACTION_TEXT, ACTION_NAME, INTENT, TEXT, ENTITIES
from rasa.shared.nlu.interpreter import NaturalLanguageInterpreter
from rasa.core.policies.policy import Policy
from rasa.core.constants import DEFAULT_POLICY_PRIORITY, DIALOGUE
from rasa.shared.core.constants import ACTIVE_LOOP, SLOTS
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.utils import train_utils
from rasa.utils.tensorflow.models import RasaModel, TransformerRasaModel
from rasa.utils.tensorflow.model_data import RasaModelData, FeatureSignature
from rasa.utils.tensorflow.model_data_utils import convert_to_data_format
from rasa.utils.tensorflow.constants import (
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
    ENCODING_DIMENSION,
    UNIDIRECTIONAL_ENCODER,
    SEQUENCE,
    SENTENCE,
    DENSE_DIMENSION,
)


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


class TEDPolicy(Policy):
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

    # please make sure to update the docs when changing a default parameter
    defaults = {
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
        zero_state_features: Optional[Dict[Text, List["Features"]]] = None,
        **kwargs: Any,
    ) -> None:
        """Declare instance variables with default values."""

        if not featurizer:
            featurizer = self._standard_featurizer(max_history)

        super().__init__(featurizer, priority)
        if isinstance(featurizer, FullDialogueTrackerFeaturizer):
            self.is_full_dialogue_featurizer_used = True
        else:
            self.is_full_dialogue_featurizer_used = False

        self._load_params(**kwargs)

        self.model = model

        self.zero_state_features = zero_state_features or defaultdict(list)

        self._label_data: Optional[RasaModelData] = None
        self.data_example: Optional[Dict[Text, List[np.ndarray]]] = None

    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        self.config = copy.deepcopy(self.defaults)
        self.config.update(kwargs)

        self.config = train_utils.check_deprecated_options(self.config)

        self.config = train_utils.update_similarity_type(self.config)
        self.config = train_utils.update_evaluation_parameters(self.config)

    def _create_label_data(
        self, domain: Domain, interpreter: NaturalLanguageInterpreter
    ) -> Tuple[RasaModelData, List[Dict[Text, List["Features"]]]]:
        # encode all label_ids with policies' featurizer
        state_featurizer = self.featurizer.state_featurizer
        encoded_all_labels = state_featurizer.encode_all_actions(domain, interpreter)

        attribute_data, _ = convert_to_data_format(encoded_all_labels)

        label_data = RasaModelData()
        label_data.add_data(attribute_data, key_prefix=f"{LABEL_KEY}_")

        label_ids = np.arange(domain.num_actions)
        label_data.add_features(
            LABEL_KEY, LABEL_SUB_KEY, [np.expand_dims(label_ids, -1)]
        )

        return label_data, encoded_all_labels

    def _create_model_data(
        self,
        tracker_state_features: List[List[Dict[Text, List["Features"]]]],
        label_ids: Optional[np.ndarray] = None,
        encoded_all_labels: Optional[List[Dict[Text, List["Features"]]]] = None,
    ) -> RasaModelData:
        """Combine all model related data into RasaModelData.

        Args:
            tracker_state_features: a dictionary of attributes (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
                ENTITIES, SLOTS, ACTIVE_LOOP) to a list of features for all dialogue
                turns in all training trackers
            label_ids: the label ids (e.g. action ids) for every dialogue turn in all
                training trackers
            encoded_all_labels: a list of dictionaries containing attribute features for labels ids

        Returns:
            RasaModelData
        """
        model_data = RasaModelData(label_key=LABEL_KEY, label_sub_key=LABEL_SUB_KEY)

        if label_ids is not None and encoded_all_labels is not None:

            label_ids = np.array(
                [np.expand_dims(seq_label_ids, -1) for seq_label_ids in label_ids]
            )
            model_data.add_features(LABEL_KEY, LABEL_SUB_KEY, [label_ids])

            attribute_data, self.zero_state_features = convert_to_data_format(
                tracker_state_features
            )
        else:
            # method is called during prediction
            attribute_data, _ = convert_to_data_format(
                tracker_state_features, self.zero_state_features
            )

        model_data.add_data(attribute_data)
        model_data.add_lengths(
            DIALOGUE, LENGTH, next(iter(list(attribute_data.keys()))), MASK
        )

        return model_data

    def train(
        self,
        training_trackers: List[TrackerWithCachedStates],
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> None:
        """Train the policy on given training trackers."""

        if not training_trackers:
            logger.error(
                f"Can not train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the policy."
            )
            return

        # dealing with training data
        tracker_state_features, label_ids = self.featurize_for_training(
            training_trackers, domain, interpreter, **kwargs
        )

        self._label_data, encoded_all_labels = self._create_label_data(
            domain, interpreter
        )

        # extract actual training data to feed to model
        model_data = self._create_model_data(
            tracker_state_features, label_ids, encoded_all_labels
        )
        if model_data.is_empty():
            logger.error(
                f"Can not train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the policy."
            )
            return

        # keep one example for persisting and loading
        self.data_example = model_data.first_data_example()

        self.model = TED(
            model_data.get_signature(),
            self.config,
            isinstance(self.featurizer, MaxHistoryTrackerFeaturizer),
            self._label_data,
        )

        self.model.fit(
            model_data,
            self.config[EPOCHS],
            self.config[BATCH_SIZES],
            self.config[EVAL_NUM_EXAMPLES],
            self.config[EVAL_NUM_EPOCHS],
            batch_strategy=self.config[BATCH_STRATEGY],
        )

    def predict_action_probabilities(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
        interpreter: NaturalLanguageInterpreter,
        **kwargs: Any,
    ) -> List[float]:
        """Predict the next action the bot should take.
        Return the list of probabilities for the next actions.
        """

        if self.model is None:
            return self._default_predictions(domain)

        # create model data from tracker
        tracker_state_features = self.featurizer.create_state_features(
            [tracker], domain, interpreter
        )
        model_data = self._create_model_data(tracker_state_features)

        output = self.model.predict(model_data)

        confidence = output["action_scores"].numpy()
        # remove batch dimension and take the last prediction in the sequence
        confidence = confidence[0, -1, :]

        if self.config[LOSS_TYPE] == SOFTMAX and self.config[RANKING_LENGTH] > 0:
            confidence = train_utils.normalize(confidence, self.config[RANKING_LENGTH])

        return confidence.tolist()

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
        label_data = RasaModelData(data=label_data)
        meta = io_utils.pickle_load(model_path / f"{SAVE_MODEL_FILE_NAME}.meta.pkl")
        priority = io_utils.json_unpickle(
            model_path / f"{SAVE_MODEL_FILE_NAME}.priority.pkl"
        )

        model_data_example = RasaModelData(
            label_key=LABEL_KEY, label_sub_key=LABEL_SUB_KEY, data=loaded_data
        )
        meta = train_utils.update_similarity_type(meta)

        model = TED.load(
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
                in STATE_LEVEL_FEATURES + FEATURES_TO_ENCODE + [DIALOGUE]
            },
        )
        model.build_for_predict(predict_data_example)

        return cls(
            featurizer=featurizer,
            priority=priority,
            model=model,
            zero_state_features=zero_state_features,
            **meta,
        )


# accessing _tf_layers with any key results in key-error, disable it
# pytype: disable=key-error


class TED(TransformerRasaModel):
    def __init__(
        self,
        data_signature: Dict[Text, Dict[Text, List[FeatureSignature]]],
        config: Dict[Text, Any],
        max_history_tracker_featurizer_used: bool,
        label_data: RasaModelData,
    ) -> None:
        super().__init__("TED", config, data_signature, label_data)

        self.max_history_tracker_featurizer_used = max_history_tracker_featurizer_used

        self.predict_data_signature = {
            feature_name: features
            for feature_name, features in data_signature.items()
            if feature_name in STATE_LEVEL_FEATURES + FEATURES_TO_ENCODE + [DIALOGUE]
        }

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        # metrics
        self.action_loss = tf.keras.metrics.Mean(name="loss")
        self.action_acc = tf.keras.metrics.Mean(name="acc")
        self.metrics_to_log += ["loss", "acc"]

        self.all_labels_embed = None  # needed for efficient prediction

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

        dialogue_transformed = self._tf_layers[f"transformer.{DIALOGUE}"](
            dialogue_in, 1 - mask, self._training
        )
        dialogue_transformed = tfa.activations.gelu(dialogue_transformed)

        if self.max_history_tracker_featurizer_used:
            # pick last vector if max history featurizer is used
            dialogue_transformed = tf.expand_dims(
                self._last_token(dialogue_transformed, sequence_lengths), 1
            )
            mask = tf.expand_dims(self._last_token(mask, sequence_lengths), 1)

        dialogue_embed = self._tf_layers[f"embed.{DIALOGUE}"](dialogue_transformed)

        return dialogue_embed, mask

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

    def _process_batch_data(
        self, tf_batch_data: Dict[Text, Dict[Text, List[tf.Tensor]]]
    ) -> tf.Tensor:
        """Encodes batch data; combines intent and text and action name and action text if both are present
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

        sim_all = self._tf_layers[f"loss.{LABEL}"].sim(
            dialogue_embed[:, :, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, tf.newaxis, :, :],
            dialogue_mask,
        )

        scores = self._tf_layers[f"loss.{LABEL}"].confidence_from_sim(
            sim_all, self.config[SIMILARITY_TYPE]
        )

        return {"action_scores": scores}


# pytype: enable=key-error
