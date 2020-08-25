import copy
import logging
import os
from pathlib import Path
from collections import defaultdict

import scipy.sparse
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import typing
from typing import Any, List, Optional, Text, Dict, Tuple, Union

import rasa.utils.io as io_utils
from rasa.core.domain import Domain
from rasa.core.featurizers.tracker_featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    MaxHistoryTrackerFeaturizer,
)
from rasa.core.featurizers.single_state_featurizer import SingleStateFeaturizer
from rasa.nlu.constants import ACTION_NAME, INTENT, ACTION_TEXT, TEXT, ENTITIES
from rasa.core.interpreter import NaturalLanguageInterpreter
from rasa.core.policies.policy import Policy
from rasa.core.constants import DEFAULT_POLICY_PRIORITY, DIALOGUE, FORM, SLOTS
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.generator import TrackerWithCachedStates
from rasa.utils import train_utils
from rasa.utils.tensorflow.models import RasaModel, TransformerRasaModel
from rasa.utils.tensorflow.model_data import RasaModelData, FeatureSignature, Data
from rasa.utils.tensorflow.constants import (
    LABEL,
    HIDDEN_LAYERS_SIZES,
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
    ENCODING_LAYER_SIZE,
    UNIDIRECTIONAL_ENCODER,
)


if typing.TYPE_CHECKING:
    from rasa.utils.features import Features


logger = logging.getLogger(__name__)

MASK = "mask"
LABEL_KEY = LABEL
LABEL_SUB_KEY = "ids"
LENGTH = "length"
SEQUENCE = "sequence"
SENTENCE = "sentence"
POSSIBLE_FEATURE_TYPES = [SEQUENCE, SENTENCE, MASK, LENGTH, LABEL_SUB_KEY]
FEATURES_TO_ENCODE = [INTENT, TEXT, ACTION_NAME, ACTION_TEXT]
STATE_LEVEL_FEATURES = [ENTITIES, SLOTS, FORM]

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

    SUPPORTS_ONLINE_TRAINING = True

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # ## Architecture of the used neural network
        # Hidden layer sizes for layers before the dialogue and label embedding layers.
        # The number of hidden layers is equal to the length of the corresponding
        # list.
        HIDDEN_LAYERS_SIZES: {DIALOGUE: [], LABEL: []},
        ENCODING_LAYER_SIZE: [50],
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
        BATCH_SIZES: [8, 32],
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
        zero_features: Optional[Dict[Text, List["Features"]]] = None,
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

        self.zero_features = zero_features or defaultdict(list)

        self._label_data: Optional[RasaModelData] = None
        self.data_example: Optional[Dict[Text, List[np.ndarray]]] = None

    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        self.config = copy.deepcopy(self.defaults)
        self.config.update(kwargs)

        self.config = train_utils.check_deprecated_options(self.config)

        self.config = train_utils.update_similarity_type(self.config)
        self.config = train_utils.update_evaluation_parameters(self.config)

    # data helpers
    # noinspection PyPep8Naming
    @staticmethod
    def _surface_attributes(
        features: List[List[Dict[Text, List["Features"]]]]
    ) -> Dict[Text, List[List[List["Features"]]]]:
        """Restructure the input.

        Args:
            features: a dictionary of attributes (INTENT, TEXT, ACTION_NAME,
                ACTION_TEXT, ENTITIES, SLOTS, FORM) to a list of features for all
                dialogue turns in all training trackers

        Returns:
            A dictionary of attributes to a list of features for all dialogue turns
            and all training trackers.
        """
        # collect all attributes
        attributes = set(
            attribute
            for features_in_tracker in features
            for features_in_dialogue in features_in_tracker
            for attribute in features_in_dialogue.keys()
        )

        attribute_to_features = defaultdict(list)

        for features_in_tracker in features:
            intermediate_features = defaultdict(list)

            for features_in_dialogue in features_in_tracker:
                for attribute in attributes:
                    # if attribute is not present in the example, populate it with None
                    intermediate_features[attribute].append(
                        features_in_dialogue.get(attribute)
                    )

            for key, value in intermediate_features.items():
                attribute_to_features[key].append(value)

        return attribute_to_features

    @staticmethod
    def _create_zero_features(
        features: List[List[List["Features"]]],
    ) -> List["Features"]:
        # all features should have the same types
        """
        Computes default feature values for an attribute;
        Args:
            features: list containing all feature values encountered
            in the dataset for an attribute;
        """

        example_features = next(
            iter(
                [
                    features_in_dialogue
                    for features_in_tracker in features
                    for features_in_dialogue in features_in_tracker
                    if features_in_dialogue is not None
                ]
            )
        )

        # create zero_features for nones
        zero_features = []
        for features in example_features:
            new_features = copy.deepcopy(features)
            if features.is_dense():
                new_features.features = np.zeros_like(features.features)
            if features.is_sparse():
                new_features.features = scipy.sparse.coo_matrix(
                    features.features.shape, features.features.dtype
                )
            zero_features.append(new_features)

        return zero_features

    def _convert_to_data_format(
        self,
        features: Union[
            List[List[Dict[Text, List["Features"]]]], List[Dict[Text, List["Features"]]]
        ],
        training: bool = True,
    ) -> Data:
        """Converts the input into "Data" format.

        Args:
            features: a dictionary of attributes (INTENT, TEXT, ACTION_NAME,
                ACTION_TEXT, ENTITIES, SLOTS, FORM) to a list of features for all
                dialogue turns in all training trackers

        Returns:
            Input in "Data" format.
        """

        remove_sequence_dimension = False
        # unify format of incoming features
        if isinstance(features[0], Dict):
            features = [[dicts] for dicts in features]
            remove_sequence_dimension = True

        features = self._surface_attributes(features)

        attribute_data = {}

        # During prediction we need to iterate over the zero features attributes to
        # have all keys in the resulting model data
        if training:
            attributes = list(features.keys())
        else:
            attributes = list(self.zero_features.keys())

        for attribute in attributes:
            features_in_tracker = (
                features[attribute] if attribute in features else [[None]]
            )

            # in case some features for a specific attribute and dialogue turn are
            # missing, replace them with a feature vector of zeros
            if training:
                self.zero_features[attribute] = self._create_zero_features(
                    features_in_tracker
                )

            (
                attribute_masks,
                _dense_features,
                _sparse_features,
            ) = self._map_tracker_features(
                features_in_tracker, self.zero_features[attribute]
            )

            sparse_features = defaultdict(list)
            dense_features = defaultdict(list)

            if remove_sequence_dimension:
                # remove added sequence dimension
                for key, values in _sparse_features.items():
                    sparse_features[key] = [value[0] for value in values]
                for key, values in _dense_features.items():
                    dense_features[key] = [value[0] for value in values]
            else:
                for key, values in _sparse_features.items():
                    sparse_features[key] = [
                        scipy.sparse.vstack(value) for value in values
                    ]
                for key, values in _dense_features.items():
                    dense_features[key] = [np.vstack(value) for value in values]

            # TODO not sure about expand_dims
            attribute_features = {MASK: [np.array(attribute_masks)]}

            feature_types = set()
            feature_types.update(list(dense_features.keys()))
            feature_types.update(list(sparse_features.keys()))

            for feature_type in feature_types:
                if feature_type == SEQUENCE:
                    # TODO we don't take sequence features because that makes us deal
                    #  with 4D sparse tensors
                    continue
                attribute_features[feature_type] = []
                if feature_type in sparse_features:
                    attribute_features[feature_type].append(
                        np.array(sparse_features[feature_type])
                    )
                if feature_type in dense_features:
                    attribute_features[feature_type].append(
                        np.array(dense_features[feature_type])
                    )

            attribute_data[attribute] = attribute_features

        return attribute_data

    def _map_tracker_features(
        self,
        features_in_tracker: List[List[List["Features"]]],
        zero_features: List["Features"],
    ) -> Tuple[
        List[np.ndarray],
        Dict[Text, List[List["Features"]]],
        Dict[Text, List[List["Features"]]],
    ]:
        """Create masks for all attributes of the given features and split the features
        into sparse and dense features.

        Args:
            features_in_tracker: all features
            zero_features: list of zero features

        Returns:
            - a list of attribute masks
            - a map of attribute to dense features
            - a map of attribute to sparse features
        """
        sparse_features = defaultdict(list)
        dense_features = defaultdict(list)
        attribute_masks = []

        for features_in_dialogue in features_in_tracker:
            dialogue_sparse_features = defaultdict(list)
            dialogue_dense_features = defaultdict(list)

            # create a mask for every state
            # to capture which turn has which input
            attribute_mask = np.expand_dims(
                np.ones(len(features_in_dialogue), np.float32), -1
            )

            for i, turn_features in enumerate(features_in_dialogue):

                if turn_features is None:
                    # use zero features and set mask to zero
                    attribute_mask[i] = 0
                    turn_features = zero_features

                for features in turn_features:
                    # all features should have the same types
                    if features.is_sparse():
                        dialogue_sparse_features[features.type].append(
                            features.features
                        )
                    else:
                        dialogue_dense_features[features.type].append(features.features)

            for key, value in dialogue_sparse_features.items():
                sparse_features[key].append(value)
            for key, value in dialogue_dense_features.items():
                dense_features[key].append(value)

            attribute_masks.append(attribute_mask)

        return attribute_masks, dense_features, sparse_features

    def _create_label_data(
        self, domain: Domain, interpreter: NaturalLanguageInterpreter
    ) -> Tuple[RasaModelData, List[Dict[Text, List["Features"]]]]:
        # encode all label_ids with policies' featurizer
        state_featurizer = self.featurizer.state_featurizer
        all_labels = state_featurizer.create_encoded_all_actions(domain, interpreter)

        attribute_data = self._convert_to_data_format(all_labels)

        label_data = RasaModelData()
        for attribute, attribute_features in attribute_data.items():
            for subkey, features in attribute_features.items():
                label_data.add_features(f"{LABEL_KEY}_{attribute}", subkey, features)

        label_ids = np.arange(domain.num_actions)
        # TODO add length of text sequence
        label_data.add_features(
            LABEL_KEY, LABEL_SUB_KEY, [np.expand_dims(label_ids, -1)]
        )

        return label_data, all_labels

    def _get_label_features(
        self, label_ids: List[List[int]], all_labels: List[Dict[Text, List["Features"]]]
    ) -> Data:
        label_ids = [label_id[0] for label_id in label_ids]
        label_data = [all_labels[label_id] for label_id in label_ids]
        label_attribute_data = self._convert_to_data_format(label_data)

        return label_attribute_data

    def _create_model_data(
        self,
        X: List[List[Dict[Text, List["Features"]]]],
        label_ids: Optional[List[List[int]]] = None,
        all_labels: Optional[List[Dict[Text, List["Features"]]]] = None,
        training: bool = True,
    ) -> RasaModelData:
        """Combine all model related data into RasaModelData.

        Args:
            X: a dictionary of attributes (INTENT, TEXT, ACTION_NAME, ACTION_TEXT,
                ENTITIES, SLOTS, FORM) to a list of features for all dialogue turns in
                all training trackers
            label_ids: the label ids (e.g. action ids) for every dialogue turn in all
                training trackers

        Returns:
            RasaModelData
        """

        # TODO the first turn in the story is `[{}] -> action_listen`,
        #  since it didn't create any features its attribute_mask will be 0

        model_data = RasaModelData(label_key=LABEL_KEY, label_sub_key=LABEL_SUB_KEY)
        if label_ids and all_labels:
            model_data.add_features(LABEL_KEY, LABEL_SUB_KEY, [np.array(label_ids)])
            attribute_data = self._get_label_features(label_ids, all_labels)
            for attribute, attribute_features in attribute_data.items():
                for subkey, features in attribute_features.items():
                    model_data.add_features(
                        f"{LABEL_KEY}_{attribute}", subkey, features
                    )

        attribute_data = self._convert_to_data_format(X, training)
        model_data.add_data(attribute_data)
        # TODO add dialogue and text lengths
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

        # dealing with training data
        X, label_ids = self.featurize_for_training(
            training_trackers, domain, interpreter, **kwargs
        )
        self._label_data, all_labels = self._create_label_data(domain, interpreter)

        # extract actual training data to feed to model
        model_data = self._create_model_data(X, label_ids, all_labels)
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
        data_X = self.featurizer.create_X([tracker], domain, interpreter)
        model_data = self._create_model_data(data_X, training=False)

        output = self.model.predict(model_data)

        confidence = output["action_scores"].numpy()
        # remove batch dimension and take the last prediction in the sequence
        confidence = confidence[0, -1, :]

        if self.config[LOSS_TYPE] == SOFTMAX and self.config[RANKING_LENGTH] > 0:
            confidence = train_utils.normalize(confidence, self.config[RANKING_LENGTH])

        return confidence.tolist()

    def persist(self, path: Text) -> None:
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

        io_utils.create_directory_for_file(tf_model_file)

        self.featurizer.persist(path)

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
            model_path / f"{SAVE_MODEL_FILE_NAME}.zero_features.pkl", self.zero_features
        )
        io_utils.pickle_dump(
            model_path / f"{SAVE_MODEL_FILE_NAME}.label_data.pkl",
            dict(self._label_data.data),
        )

    @classmethod
    def load(cls, path: Text) -> "TEDPolicy":
        """Loads a policy from the storage.
        **Needs to load its featurizer**
        """

        if not os.path.exists(path):
            raise Exception(
                f"Failed to load TED policy model. Path "
                f"'{os.path.abspath(path)}' doesn't exist."
            )

        model_path = Path(path)
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
        zero_features = io_utils.pickle_load(
            model_path / f"{SAVE_MODEL_FILE_NAME}.zero_features.pkl"
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
            zero_features=zero_features,
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

        self.all_labels_embed = None

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
        self._prepare_dot_product_loss(LABEL, self.config[SCALE_LOSS])

        for name in self.data_signature.keys():
            self._prepare_utterance_level_layers(name)

        for name in FEATURES_TO_ENCODE:
            self._prepare_encoding_layers(name)

        self._prepare_ffnn_layer(
            DIALOGUE,
            self.config[HIDDEN_LAYERS_SIZES][DIALOGUE],
            self.config[DROP_RATE_DIALOGUE],
        )
        self._prepare_ffnn_layer(
            LABEL, self.config[HIDDEN_LAYERS_SIZES][LABEL], self.config[DROP_RATE_LABEL]
        )

        self._prepare_transformer_layer(
            DIALOGUE, self.config[DROP_RATE_DIALOGUE], self.config[DROP_RATE_ATTENTION]
        )

        self._prepare_embed_layers(DIALOGUE)
        self._prepare_embed_layers(LABEL)

    def _prepare_utterance_level_layers(self, name: Text) -> None:
        for feature_type in POSSIBLE_FEATURE_TYPES:
            if (
                name not in self.data_signature
                or feature_type not in self.data_signature[name]
            ):
                continue

            self._prepare_sparse_dense_dropout_layers(
                f"{name}_{feature_type}", self.config[DROP_RATE]
            )

            if name not in STATE_LEVEL_FEATURES:
                self._prepare_sparse_dense_layers(
                    self.data_signature[name][feature_type],
                    f"{name}_{feature_type}",
                    10,
                )
            else:
                self._prepare_sparse_dense_layers(
                    self.data_signature[name][feature_type],
                    f"{name}_{feature_type}",
                    self.data_signature[name][feature_type][0][1],
                )

    def _prepare_encoding_layers(self, name: Text) -> None:
        feature_type = SENTENCE
        if (
            name not in self.data_signature
            or feature_type not in self.data_signature[name]
        ):
            return

        self._prepare_ffnn_layer(
            f"{name}_{feature_type}",
            self.config[ENCODING_LAYER_SIZE],
            self.config[DROP_RATE_DIALOGUE],
        )

    def _create_all_labels_embed(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_labels = []

        for key in self.tf_label_data.keys():
            mask = None
            if MASK in self.tf_label_data[key]:
                mask = self.tf_label_data[key][MASK][0]

            for sub_key in self.tf_label_data[key].keys():
                if sub_key not in [SEQUENCE, SENTENCE]:
                    continue

                label_features = self._combine_sparse_dense_features(
                    self.tf_label_data[key][sub_key], f"{key}_{sub_key}", mask=mask
                )
                all_labels.append(label_features)

        all_labels = tf.squeeze(tf.concat(all_labels, axis=-1), axis=1)
        all_labels_embed = self._embed_label(all_labels)

        return all_labels, all_labels_embed

    def _emebed_dialogue(
        self, dialogue_in: tf.Tensor, sequence_lengths: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create dialogue level embedding and mask."""

        mask = self._compute_mask(sequence_lengths)

        dialogue = self._tf_layers[f"ffnn.{DIALOGUE}"](dialogue_in, self._training)
        dialogue_transformed = self._tf_layers[f"transformer.{DIALOGUE}"](
            dialogue, 1 - mask, self._training
        )
        dialogue_transformed = tfa.activations.gelu(dialogue_transformed)

        if self.max_history_tracker_featurizer_used:
            # pick last label if max history featurizer is used
            # dialogue_transformed = dialogue_transformed[:, -1:, :]
            dialogue_transformed = tf.expand_dims(
                self._last_token(dialogue_transformed, sequence_lengths), 1
            )
            mask = self._last_token(mask, sequence_lengths)

        dialogue_embed = self._tf_layers[f"embed.{DIALOGUE}"](dialogue_transformed)

        return dialogue_embed, mask

    def _embed_label(self, label_in: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        label = self._tf_layers[f"ffnn.{LABEL}"](label_in, self._training)
        return self._tf_layers[f"embed.{LABEL}"](label)

    def _encode_features_per_attribute(
        self, batch: Dict[Text, Dict[Text, List[tf.Tensor]]], attribute: Text
    ) -> Optional[tf.Tensor]:
        if not batch[attribute]:
            return None

        mask = batch[attribute][MASK][0]
        attribute_features = self._combine_sparse_dense_features(
            batch[attribute][SENTENCE], f"{attribute}_{SENTENCE}", mask=mask
        )

        if attribute in FEATURES_TO_ENCODE:
            attribute_features = self._tf_layers[f"ffnn.{attribute}_{SENTENCE}"](
                attribute_features
            )

        return attribute_features

    def _preprocess_batch(
        self, batch: Dict[Text, Dict[Text, List[tf.Tensor]]]
    ) -> tf.Tensor:
        batch_encoded = {
            key: self._encode_features_per_attribute(batch, key)
            for key in batch.keys()
            if not LABEL_KEY in key and not DIALOGUE in key
        }

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

        for key in batch_encoded.keys():
            batch_features.append(batch_encoded.get(key))
            # # ignore features which are essentially empty
            # # (where there is nothing in the domain);
            # if not batch_encoded.get(key).shape[-1] == 0:
            #     batch_features.append(batch_encoded.get(key))

        batch_features = tf.concat(batch_features, axis=-1)

        return batch_features

    def _process_label_features(
        self, batch: Dict[Text, Dict[Text, List[tf.Tensor]]]
    ) -> tf.Tensor:
        label_in = []
        label_keys = [key for key in self.tf_label_data.keys() if LABEL in key]

        for key in label_keys:
            mask = None
            if MASK in batch[key]:
                mask = tf.expand_dims(batch[key][MASK][0], axis=-2)

            for sub_key in batch[key]:
                if sub_key not in [SEQUENCE, SENTENCE]:
                    continue

                label_features = self._combine_sparse_dense_features(
                    batch[key][sub_key], f"{key}_{sub_key}"
                )
                label_in.append(tf.expand_dims(label_features, axis=-2) * mask)

        return tf.squeeze(tf.concat(label_in, axis=-1))

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        batch = self.batch_to_model_data_format(batch_in, self.data_signature)

        sequence_lengths = tf.cast(
            tf.squeeze(batch[DIALOGUE][LENGTH], axis=0), tf.int32
        )

        label_in = self._process_label_features(batch)
        dialogue_in = self._preprocess_batch(batch)
        all_labels, all_labels_embed = self._create_all_labels_embed()

        dialogue_embed, mask = self._emebed_dialogue(dialogue_in, sequence_lengths)
        label_embed = self._embed_label(label_in)

        loss, acc = self._tf_layers[f"loss.{LABEL}"](
            dialogue_embed, label_embed, label_in, all_labels_embed, all_labels, mask
        )

        self.action_loss.update_state(loss)
        self.action_acc.update_state(acc)

        return loss

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        batch = self.batch_to_model_data_format(batch_in, self.predict_data_signature)

        sequence_lengths = tf.cast(
            tf.squeeze(batch[DIALOGUE][LENGTH], axis=0), tf.int32
        )

        if self.all_labels_embed is None:
            _, self.all_labels_embed = self._create_all_labels_embed()

        dialogue_in = self._preprocess_batch(batch)
        dialogue_embed, mask = self._emebed_dialogue(dialogue_in, sequence_lengths)

        sim_all = self._tf_layers[f"loss.{LABEL}"].sim(
            dialogue_embed[:, :, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, tf.newaxis, :, :],
            mask,
        )

        scores = self._tf_layers[f"loss.{LABEL}"].confidence_from_sim(
            sim_all, self.config[SIMILARITY_TYPE]
        )

        return {"action_scores": scores}


# pytype: enable=key-error
