import copy
import logging
import os
from pathlib import Path

from collections import defaultdict
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from scipy import sparse
import typing
from typing import Any, List, Optional, Text, Dict, Tuple, Union

import rasa.utils.io as io_utils
from rasa.core.domain import Domain
from rasa.core.featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    MaxHistoryTrackerFeaturizer,
    SingleStateFeaturizer,
)
from rasa.nlu.constants import ACTION_NAME, INTENT, ACTION_TEXT
from rasa.core.interpreter import NaturalLanguageInterpreter, RegexInterpreter
from rasa.core.policies.policy import Policy
from rasa.core.constants import DEFAULT_POLICY_PRIORITY, DIALOGUE
from rasa.core.trackers import DialogueStateTracker
from rasa.core.training.generator import TrackerWithCachedStates
from rasa.utils import train_utils
from rasa.utils.tensorflow import layers
from rasa.utils.tensorflow.transformer import TransformerEncoder
from rasa.utils.tensorflow.models import RasaModel
from rasa.utils.tensorflow.model_data import RasaModelData, FeatureSignature
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
)

if typing.TYPE_CHECKING:
    from rasa.utils.features import Features


logger = logging.getLogger(__name__)

DIALOGUE_FEATURES = f"{DIALOGUE}_features"
LABEL_FEATURES = f"{LABEL}_features"
LABEL_IDS = f"{LABEL}_ids"

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
        HIDDEN_LAYERS_SIZES: {DIALOGUE: [], LABEL: [], f"{DIALOGUE}_name_text": [100]},
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
    def _surface_attributes(list_of_list_of_dicts):
        # collect all attributes
        attributes = set(
            attribute
            for list_of_dicts in list_of_list_of_dicts
            for label in list_of_dicts
            for attribute in label.keys()
        )

        out = defaultdict(list)
        for list_of_dicts in list_of_list_of_dicts:
            seq_out = defaultdict(list)
            for label in list_of_dicts:
                for attribute in attributes:
                    # if attribute is not present in the example, populate it with None
                    seq_out[attribute].append(label.get(attribute))
            for key, value in seq_out.items():
                out[key].append(value)

        return out

    @staticmethod
    def _create_zero_features(list_of_all_features):
        # all features should have the same types

        example_features = next(
            iter(
                [
                    example_features
                    for all_features in list_of_all_features
                    for example_features in all_features
                    if example_features is not None
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
                new_features.features = sparse.coo_matrix(
                    features.features.shape, features.features.dtype
                )
            zero_features.append(new_features)

        return zero_features

    def _convert_to_data(self, list_of_dicts, double_list=False):
        # we might have either a dict of lists or a dict of of list of lists
        list_of_list_of_dicts = []
        if not double_list:
            for dicts in list_of_dicts:
                list_of_list_of_dicts.append([dicts])
        else:
            list_of_list_of_dicts = list_of_dicts

        dict_of_lists_lists = self._surface_attributes(list_of_list_of_dicts)

        attribute_data = {}
        for attribute, list_of_all_features in dict_of_lists_lists.items():

            attribute_mask = np.ones(len(list_of_all_features), np.float32)

            zero_features = self._create_zero_features(list_of_all_features)
            sparse_features = defaultdict(list)
            dense_features = defaultdict(list)
            feature_types = set()
            for i, all_features in enumerate(list_of_all_features):

                seq_sparse_features = defaultdict(list)
                seq_dense_features = defaultdict(list)

                for example_features in all_features:

                    if example_features is None:
                        # use zero features and set mask to zero
                        attribute_mask[i] = 0
                        example_features = zero_features

                    for features in example_features:
                        # all features should have the same types
                        feature_types.add(features.type)
                        if features.is_sparse():
                            seq_sparse_features[features.type].append(features.features)
                        else:
                            seq_dense_features[features.type].append(features.features)

                for key, value in seq_sparse_features.items():
                    sparse_features[key].append(value)
                for key, value in seq_dense_features.items():
                    dense_features[key].append(value)

            if not double_list:
                # remove added sequence dimension
                for key, values in sparse_features.items():
                    new_values = []
                    for value in values:
                        new_values.append(value[0])
                    sparse_features[key] = new_values
                for key, values in dense_features.items():
                    new_values = []
                    for value in values:
                        new_values.append(value[0])
                    dense_features[key] = new_values

            # TODO not sure about expand_dims
            attribute_features = {"mask": [np.expand_dims(attribute_mask, -1)]}
            for feature_type in feature_types:
                attribute_features[feature_type] = [
                    np.array(sparse_features[feature_type]),
                    np.array(dense_features[feature_type]),
                ]
            attribute_data[attribute] = attribute_features
        return attribute_data

    def _create_label_data(
        self, domain: Domain, interpreter: NaturalLanguageInterpreter
    ) -> RasaModelData:
        # encode all label_ids with policies' featurizer
        state_featurizer = self.featurizer.state_featurizer
        all_labels = state_featurizer.create_encoded_all_actions(domain, interpreter)

        attribute_data = self._convert_to_data(all_labels)

        label_data = RasaModelData()
        for attribute, attribute_features in attribute_data.items():
            label_data.add_features(attribute, attribute_features)

        label_ids = np.arange(domain.num_actions)
        # TODO not sure about expand_dims
        # TODO add length of text sequence
        label_data.add_features(LABEL_IDS, [np.expand_dims(label_ids, -1)])

        return label_data

    def _create_model_data(
        self, X: List[List[Dict[Text, List["Features"]]]], label_ids: List[List[int]],
    ) -> RasaModelData:
        """Combine all model related data into RasaModelData."""

        # TODO needs to be checked that double surfacing is working
        # TODO the first turn in the story is `[{}] -> action_listen`,
        #  since it didn't create any features its attribute_mask will be 0
        attribute_data = self._convert_to_data(X, True)

        model_data = RasaModelData(label_key=LABEL_IDS)
        for attribute, attribute_features in attribute_data.items():
            model_data.add_features(attribute, attribute_features)

        model_data.add_features(LABEL_IDS, [np.array(label_ids)])
        # TODO add dialogue and text lengths
        # model_data.add_features("dialog_lengths", [dialog_lengths])
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

        self._label_data = self._create_label_data(domain, interpreter)

        # extract actual training data to feed to model
        model_data = self._create_model_data(X, label_ids)
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
        model_data = self._create_model_data(data_X)

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
            model_path / f"{SAVE_MODEL_FILE_NAME}.label_data.pkl", self._label_data
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
        meta = io_utils.pickle_load(model_path / f"{SAVE_MODEL_FILE_NAME}.meta.pkl")
        priority = io_utils.json_unpickle(
            model_path / f"{SAVE_MODEL_FILE_NAME}.priority.pkl"
        )

        model_data_example = RasaModelData(label_key=LABEL_IDS, data=loaded_data)
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
            label_key=LABEL_IDS,
            data={
                feature_name: features
                for feature_name, features in model_data_example.items()
                if DIALOGUE in feature_name
            },
        )
        model.build_for_predict(predict_data_example)

        return cls(featurizer=featurizer, priority=priority, model=model, **meta)


# accessing _tf_layers with any key results in key-error, disable it
# pytype: disable=key-error


class TED(RasaModel):
    def __init__(
        self,
        data_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
        max_history_tracker_featurizer_used: bool,
        label_data: RasaModelData,
    ) -> None:
        super().__init__(
            name="TED",
            random_seed=config[RANDOM_SEED],
            tensorboard_log_dir=config[TENSORBOARD_LOG_DIR],
            tensorboard_log_level=config[TENSORBOARD_LOG_LEVEL],
        )

        self.config = config
        self.max_history_tracker_featurizer_used = max_history_tracker_featurizer_used

        # data
        self.data_signature = data_signature
        self._check_data()

        self.predict_data_signature = {
            feature_name: features
            for feature_name, features in data_signature.items()
            if DIALOGUE in feature_name
        }

        # optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        self.all_labels_embed = None

        label_batch = label_data.prepare_batch()
        self.tf_label_data = self.batch_to_model_data_format(
            label_batch, label_data.get_signature()
        )

        # metrics
        self.action_loss = tf.keras.metrics.Mean(name="loss")
        self.action_acc = tf.keras.metrics.Mean(name="acc")
        self.metrics_to_log += ["loss", "acc"]

        # set up tf layers
        self._tf_layers: Dict[Text : tf.keras.layers.Layer] = {}
        self._prepare_layers()

    def _check_data(self) -> None:
        if (
            f"{DIALOGUE_FEATURES}_user" not in self.data_signature
            and f"{DIALOGUE_FEATURES}_user_name" not in self.data_signature
        ):
            raise ValueError(
                f"No user features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )

        if (
            f"{DIALOGUE_FEATURES}_action" not in self.data_signature
            and f"{DIALOGUE_FEATURES}_action_name" not in self.data_signature
        ):
            raise ValueError(
                f"No action features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if (
            LABEL_FEATURES not in self.data_signature
            and f"{LABEL_FEATURES}_action_name" not in self.data_signature
        ):
            raise ValueError(
                f"No label features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )

    def _prepare_sparse_dense_layers(
        self,
        data_signature: List[FeatureSignature],
        name: Text,
        reg_lambda: float,
        dense_dim: int,
    ) -> None:
        sparse = False
        dense = False
        for is_sparse, shape in data_signature:
            if is_sparse:
                sparse = True
            else:
                dense = True

        if sparse:
            self._tf_layers[f"sparse_to_dense.{name}"] = layers.DenseForSparse(
                units=dense_dim, reg_lambda=reg_lambda, name=name
            )
            if not dense:
                # create dense labels for the input to use in negative sampling
                self._tf_layers[f"sparse_to_dense_ids.{name}"] = layers.DenseForSparse(
                    units=2, trainable=False, name=f"sparse_to_dense_ids.{name}"
                )

    def _prepare_layers(self) -> None:
        self._tf_layers[f"loss.{LABEL}"] = layers.DotProductLoss(
            self.config[NUM_NEG],
            self.config[LOSS_TYPE],
            self.config[MAX_POS_SIM],
            self.config[MAX_NEG_SIM],
            self.config[USE_MAX_NEG_SIM],
            self.config[NEGATIVE_MARGIN_SCALE],
            self.config[SCALE_LOSS],
            # set to 1 to get deterministic behaviour
            parallel_iterations=1 if self.random_seed is not None else 1000,
        )

        for feature_name in self.data_signature.keys():
            if feature_name.startswith(DIALOGUE_FEATURES) or feature_name.startswith(
                LABEL_FEATURES
            ):
                self._prepare_sparse_dense_layers(
                    self.data_signature[feature_name],
                    feature_name,
                    self.config[REGULARIZATION_CONSTANT],
                    100,
                )

            if feature_name.startswith(DIALOGUE_FEATURES) and not feature_name.endswith(
                "if_text"
            ):
                self._tf_layers[f"ffnn.{feature_name}"] = layers.Ffnn(
                    self.config[HIDDEN_LAYERS_SIZES][f"{DIALOGUE}_name_text"],
                    self.config[DROP_RATE_DIALOGUE],
                    self.config[REGULARIZATION_CONSTANT],
                    self.config[WEIGHT_SPARSITY],
                    layer_name_suffix=feature_name,
                )

        self._tf_layers[f"ffnn.{DIALOGUE}"] = layers.Ffnn(
            self.config[HIDDEN_LAYERS_SIZES][DIALOGUE],
            self.config[DROP_RATE_DIALOGUE],
            self.config[REGULARIZATION_CONSTANT],
            self.config[WEIGHT_SPARSITY],
            layer_name_suffix=DIALOGUE,
        )
        self._tf_layers[f"ffnn.{LABEL}"] = layers.Ffnn(
            self.config[HIDDEN_LAYERS_SIZES][LABEL],
            self.config[DROP_RATE_LABEL],
            self.config[REGULARIZATION_CONSTANT],
            self.config[WEIGHT_SPARSITY],
            layer_name_suffix=LABEL,
        )
        self._tf_layers["transformer"] = TransformerEncoder(
            self.config[NUM_TRANSFORMER_LAYERS],
            self.config[TRANSFORMER_SIZE],
            self.config[NUM_HEADS],
            self.config[TRANSFORMER_SIZE] * 4,
            self.config[REGULARIZATION_CONSTANT],
            dropout_rate=self.config[DROP_RATE_DIALOGUE],
            attention_dropout_rate=self.config[DROP_RATE_ATTENTION],
            sparsity=self.config[WEIGHT_SPARSITY],
            unidirectional=True,
            use_key_relative_position=self.config[KEY_RELATIVE_ATTENTION],
            use_value_relative_position=self.config[VALUE_RELATIVE_ATTENTION],
            max_relative_position=self.config[MAX_RELATIVE_POSITION],
            name=DIALOGUE + "_encoder",
        )
        self._tf_layers[f"embed.{DIALOGUE}"] = layers.Embed(
            self.config[EMBEDDING_DIMENSION],
            self.config[REGULARIZATION_CONSTANT],
            DIALOGUE,
            self.config[SIMILARITY_TYPE],
        )
        self._tf_layers[f"embed.{LABEL}"] = layers.Embed(
            self.config[EMBEDDING_DIMENSION],
            self.config[REGULARIZATION_CONSTANT],
            LABEL,
            self.config[SIMILARITY_TYPE],
        )

    def _combine_sparse_dense_features(
        self,
        features: List[Union[np.ndarray, tf.Tensor, tf.SparseTensor]],
        name: Text,
        sparse_dropout: bool = False,
    ) -> tf.Tensor:

        dense_features = []

        for f in features:
            if isinstance(f, tf.SparseTensor):
                if sparse_dropout:
                    _f = self._tf_layers[f"sparse_dropout.{name}"](f, self._training)
                else:
                    _f = f
                dense_features.append(self._tf_layers[f"sparse_to_dense.{name}"](_f))
            else:
                dense_features.append(f)

        return tf.concat(dense_features, axis=-1)

    def _create_all_labels_embed(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_labels = []
        for key in self.tf_label_data.keys():
            if key.startswith(LABEL_FEATURES):
                all_labels.append(
                    self._combine_sparse_dense_features(self.tf_label_data[key], key)
                )

        all_labels = tf.concat(all_labels, axis=-1)
        all_labels = tf.squeeze(all_labels, axis=1)
        all_labels_embed = self._embed_label(all_labels)

        return all_labels, all_labels_embed

    @staticmethod
    def _compute_mask(sequence_lengths: tf.Tensor) -> tf.Tensor:
        mask = tf.sequence_mask(sequence_lengths, dtype=tf.float32)
        return mask

    @staticmethod
    def _last_token(x: tf.Tensor, sequence_lengths: tf.Tensor) -> tf.Tensor:
        last_sequence_index = tf.maximum(0, sequence_lengths - 1)
        batch_index = tf.range(tf.shape(last_sequence_index)[0])

        indices = tf.stack([batch_index, last_sequence_index], axis=1)
        return tf.expand_dims(tf.gather_nd(x, indices), 1)

    def _emebed_dialogue(
        self, dialogue_in: tf.Tensor, sequence_lengths
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create dialogue level embedding and mask."""

        mask = self._compute_mask(sequence_lengths)

        dialogue = self._tf_layers[f"ffnn.{DIALOGUE}"](dialogue_in, self._training)
        dialogue_transformed = self._tf_layers["transformer"](
            dialogue, 1 - tf.expand_dims(mask, axis=-1), self._training
        )
        dialogue_transformed = tfa.activations.gelu(dialogue_transformed)

        if self.max_history_tracker_featurizer_used:
            # pick last label if max history featurizer is used
            # dialogue_transformed = dialogue_transformed[:, -1:, :]
            dialogue_transformed = self._last_token(
                dialogue_transformed, sequence_lengths
            )
            mask = self._last_token(mask, sequence_lengths)

        dialogue_embed = self._tf_layers[f"embed.{DIALOGUE}"](dialogue_transformed)

        return dialogue_embed, mask

    def _embed_label(self, label_in: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        label = self._tf_layers[f"ffnn.{LABEL}"](label_in, self._training)
        return self._tf_layers[f"embed.{LABEL}"](label)

    def _preprocess_batch(self, batch: Dict[Text, List[tf.Tensor]]) -> tf.Tensor:
        name_features = None
        text_features = None
        batch_features = []
        feats_to_combine = [f"{DIALOGUE_FEATURES}_user", f"{DIALOGUE_FEATURES}_action"]
        for feature in feats_to_combine:
            if not batch[feature] == []:
                text_features = self._combine_sparse_dense_features(
                    batch[feature], feature
                )
                text_features = self._tf_layers[f"ffnn.{feature}"](text_features)
                mask = tf.cast(
                    tf.math.equal(batch[f"{feature}_if_text"], 1), tf.float32
                )
                text_features = text_features * mask
            if not batch[f"{feature}_name"] == []:
                name_features = self._combine_sparse_dense_features(
                    batch[f"{feature}_name"], f"{feature}_name"
                )
                name_features = self._tf_layers[f"ffnn.{feature}_name"](name_features)
                mask = tf.cast(
                    tf.math.equal(batch[f"{feature}_if_text"], -1), tf.float32
                )
                name_features = name_features * mask

            if text_features is not None and name_features is not None:
                batch_features.append(text_features + name_features)
            else:
                batch_features.append(
                    text_features if text_features is not None else name_features
                )

        if not batch[f"{DIALOGUE_FEATURES}_entities"] == []:
            batch_features.append(batch[f"{DIALOGUE_FEATURES}_entities"])

        batch_features = tf.squeeze(tf.concat(batch_features, axis=-1), 0)
        return batch_features

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        batch = self.batch_to_model_data_format(batch_in, self.data_signature)
        sequence_lengths = tf.cast(
            tf.squeeze(batch["dialog_lengths"], axis=0), tf.int32
        )

        label_in = []
        for key in batch.keys():
            if key.startswith(LABEL_FEATURES):
                label_in.append(self._combine_sparse_dense_features(batch[key], key))
        label_in = tf.concat(label_in, axis=-1)

        all_labels, all_labels_embed = self._create_all_labels_embed()

        dialogue_in = self._preprocess_batch(batch)

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

        dialogue_in = self._preprocess_batch(batch)
        sequence_lengths = tf.expand_dims(tf.shape(dialogue_in)[1], axis=0)

        if self.all_labels_embed is None:
            _, self.all_labels_embed = self._create_all_labels_embed()

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
