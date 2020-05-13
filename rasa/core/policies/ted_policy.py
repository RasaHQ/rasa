import copy
import logging
import os
from pathlib import Path

import pickle
import scipy.sparse

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from typing import Any, List, Optional, Text, Dict, Tuple, Union

import rasa.utils.io as io_utils
from rasa.core.domain import Domain
from rasa.core.featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    E2ESingleStateFeaturizer,
    MaxHistoryTrackerFeaturizer,
)
from rasa.core.policies.policy import Policy
from rasa.core.constants import DEFAULT_POLICY_PRIORITY, DIALOGUE
from rasa.core.trackers import DialogueStateTracker
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
    UTTERANCE_TRANSFORMER_SIZE,
    UTTERANCE_NUM_TRANSFORMER_LAYERS,
    UTTERANCE_NUM_HEADS,
    ENTITY_RECOGNITION,
)


logger = logging.getLogger(__name__)

DIALOGUE_FEATURES = f"{DIALOGUE}_features"
DIALOGUE_MASK = f"{DIALOGUE}_mask"
LABEL_FEATURES = f"{LABEL}_features"
LABEL_IDS = f"{LABEL}_ids"
LABEL_MASK = f"{LABEL}_mask"
USER = "user"
BOT = "bot"

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
        # Number of units in transformer
        TRANSFORMER_SIZE: 512,
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
        BATCH_SIZES: [64, 128],
        # Strategy used whenc creating batches.
        # Can be either 'sequence' or 'balanced'.
        BATCH_STRATEGY: BALANCED,
        # Number of epochs to train
        EPOCHS: 1,
        # Set random seed to any 'int' to get reproducible results
        RANDOM_SEED: None,
        # ## Parameters for embeddings
        # Dimension size of embedding vectors
        EMBEDDING_DIMENSION: 100,
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
            E2ESingleStateFeaturizer(), max_history=max_history
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
    def _label_ids_for_Y(data_Y: np.ndarray) -> np.ndarray:
        """Prepare Y data for training: extract label_ids.

        label_ids are indices of labels, while `data_Y` contains one-hot encodings.
        """

        return data_Y.argmax(axis=-1)

    # noinspection PyPep8Naming
    def _label_features_for_Y(self, label_ids: np.ndarray) -> np.ndarray:
        """Prepare Y data for training: features for label_ids."""

        all_label_features = self._label_data.get(LABEL_FEATURES)

        is_full_dialogue_featurizer_used = len(label_ids.shape) == 2
        if is_full_dialogue_featurizer_used:
            return np.stack(
                [
                    np.stack(
                        [all_label_features[label_idx] for label_idx in seq_label_ids]
                    )
                    for seq_label_ids in label_ids
                ]
            )

        if len(all_label_features) == 2:
            Y_sparse = np.stack(
                [all_label_features[0][label_idx] for label_idx in label_ids]
            )
            Y_dense = np.stack(
                [all_label_features[1][label_idx] for label_idx in label_ids]
            )
        elif len(all_label_features) == 1:
            if isinstance(all_label_features[0], scipy.sparse.spmatrix):
                Y_sparse = np.stack(
                    [all_label_features[0][label_idx] for label_idx in label_ids]
                )
                Y_dense = np.array([])
            elif isinstance(all_label_features[0], np.ndarray):
                Y_sparse = np.array([])
                Y_dense = np.stack(
                    [all_label_features[1][label_idx] for label_idx in label_ids]
                )

        # max history featurizer is used
        return Y_sparse, Y_dense

    # noinspection PyPep8Naming
    def _create_model_data(
        self,
        data_X: Optional[np.ndarray],
        dialog_lengths: Optional[np.ndarray] = None,
        data_Y: Optional[np.ndarray] = None,
    ) -> RasaModelData:
        """Combine all model related data into RasaModelData."""

        label_ids = np.array([])
        Y_sparse, Y_dense = np.array([]), np.array([])

        if data_Y is not None:
            # label_ids = self._label_ids_for_Y(data_Y)
            label_ids = np.squeeze(data_Y, axis=-1)
            Y_sparse, Y_dense = self._label_features_for_Y(label_ids)
            # explicitly add last dimension to label_ids
            # to track correctly dynamic sequences
            label_ids = np.expand_dims(label_ids, -1)

        model_data = RasaModelData(label_key=LABEL_IDS)

        X_sparse = []
        X_dense = []
        X_entities = []

        for dial in data_X:
            sparse_state = []
            dense_state = []
            entities = []
            for state in dial:
                if state[0] is not None:
                    sparse_state.append(state[0].astype(np.float32))
                if state[1] is not None:
                    dense_state.append(state[1])
                if state[2] is not None:
                    entities.append(state[2])

            if not sparse_state == []:
                sparse_state = scipy.sparse.vstack(sparse_state)
            if not dense_state == []:
                dense_state = np.vstack(dense_state)
            if not entities == []:
                entities = np.vstack(entities)
            X_sparse.append(sparse_state)
            X_dense.append(dense_state)
            X_entities.append(entities)
        model_data.add_features(
            DIALOGUE_FEATURES,
            [np.array(X_sparse), np.array(X_dense), np.array(X_entities)],
        )
        model_data.add_features(LABEL_FEATURES, [Y_sparse, Y_dense])
        model_data.add_features(LABEL_IDS, [label_ids])
        if dialog_lengths is not None:
            model_data.add_features("dialog_lengths", [dialog_lengths])
        return model_data

    def collect_label_features(self, labels_example):
        sparse_features = []
        dense_features = []
        for feats in labels_example:
            if not feats[0] is None:
                sparse_features.append(feats[0].tocsr().astype(np.float32))
            if not feats[1] is None:
                dense_features.append(feats[1])

        sparse_features = scipy.sparse.vstack(sparse_features)
        dense_features = np.array(dense_features)
        return [sparse_features, dense_features]

    def _create_label_data(self, domain, **kwargs) -> RasaModelData:
        # encode all label_ids with policies' featurizer
        # sparse_features, dense_features =
        labels_idx_examples = self.featurizer.state_featurizer.create_encoded_all_actions(
            domain, kwargs
        )
        labels_idx_examples = sorted(labels_idx_examples, key=lambda x: x[0])
        labels_example = [example for (_, example) in labels_idx_examples]
        label_features = self.collect_label_features(labels_example)

        label_data = RasaModelData()
        label_data.add_features(LABEL_FEATURES, label_features)

        label_ids = np.array([idx for (idx, _) in labels_idx_examples])
        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        label_data.add_features(LABEL_IDS, [np.expand_dims(label_ids, -1)])
        return label_data

    def _create_label_data_e2e(self, label_data) -> RasaModelData:
        # encode all label_ids with policies' featurizer
        sparse_features = []
        dense_features = []
        for idx, feats in label_data:
            if feats[0] is not None:
                sparse_features.append(feats[0].astype(np.float32))
            if feats[1] is not None:
                dense_features.append(feats[1])

        sparse_features = scipy.sparse.vstack(sparse_features)
        # dense_features = np.vstack(dense_features).astype(np.float32)

        label_data = RasaModelData()
        label_data.add_features(LABEL_FEATURES, [label_features])
        return label_data

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any,
    ) -> None:
        """Train the policy on given training trackers."""

        # dealing with training data
        training_data = self.featurize_for_training(training_trackers, domain, **kwargs)

        self._label_data = self._create_label_data(domain)
        # self._label_data = self._create_label_data_e2e(label_data)
        
        # extract actual training data to feed to model
        model_data = self._create_model_data(
            training_data.X, np.array(training_data.true_length), training_data.y
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
        self, tracker: DialogueStateTracker, domain: Domain,
    ) -> List[float]:
        """Predict the next action the bot should take.

        Return the list of probabilities for the next actions.
        """

        if self.model is None:
            return self._default_predictions(domain)
        kwargs = {}

        # create model data from tracker
        data_X = self.featurizer.create_X([tracker], domain, **kwargs)
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
        print('FEATURIZER')
        print(self.featurizer)

        self.featurizer.persist(path)

        self.model.save(str(tf_model_file))

        io_utils.json_pickle(
            model_path / f"{SAVE_MODEL_FILE_NAME}.priority.pkl", self.priority
        )
        io_utils.pickle_dump(
            model_path / f"{SAVE_MODEL_FILE_NAME}.meta.pkl", self.config
        )
        # using pickle to be able to store and load both sparse and dense data_example
        io_utils.pickle_dump(
            model_path / f"{SAVE_MODEL_FILE_NAME}.data_example.pkl", self.data_example
        )
        io_utils.json_pickle(
            model_path / f"{SAVE_MODEL_FILE_NAME}.label_data.pkl", self._label_data
        )

    @classmethod
    def load(cls, path: Text) -> "TEDPolicy":
        """Loads a policy from the storage.

        **Needs to load its featurizer**
        """
        from rasa.nlu.model import Interpreter

        if not os.path.exists(path):
            raise Exception(
                f"Failed to load TED policy model. Path "
                f"'{os.path.abspath(path)}' doesn't exist."
            )

        model_path = Path(path)
        tf_model_file = model_path / f"{SAVE_MODEL_FILE_NAME}.tf_model"

        featurizer = TrackerFeaturizer.load(path)
        # setting path variable for featurizer so that we can load the nlu pipeline with
        # DIET from there;
        featurizer.path = path

        if not (model_path / f"{SAVE_MODEL_FILE_NAME}.data_example.pkl").is_file():
            return cls(featurizer=featurizer)
        # using pickle to be able to store and load both sparse and dense data_example
        loaded_data = io_utils.pickle_load(
            model_path / f"{SAVE_MODEL_FILE_NAME}.data_example.pkl"
        )
        label_data = io_utils.json_unpickle(
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
        self._set_optimizer(tf.keras.optimizers.Adam())

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
        if DIALOGUE_FEATURES not in self.data_signature:
            raise ValueError(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if LABEL_FEATURES not in self.data_signature:
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
                # if dense features are present
                # use the feature dimension of the dense features
                # dense_dim = shape[-1]
                dense_dim = dense_dim

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

        self._prepare_sparse_dense_layers(
            self.data_signature["dialogue_features"],
            "dialogue_features",
            self.config[REGULARIZATION_CONSTANT],
            100,
        )

        for is_sparse, shape in self.data_signature["label_features"]:
            if is_sparse:
                sparse_dim_label_features = shape[-1]
            else:
                sparse_dim_label_features = 100

        self._prepare_sparse_dense_layers(
            self.data_signature["label_features"],
            "label_features",
            self.config[REGULARIZATION_CONSTANT],
            100,
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

    def _create_all_labels_embed(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_labels = self.tf_label_data[LABEL_FEATURES]
        all_labels = self._combine_sparse_dense_features(all_labels, LABEL_FEATURES)
        all_labels = tf.squeeze(all_labels, axis=1)
        all_labels_embed = self._embed_label(all_labels)

        return all_labels, all_labels_embed

    @staticmethod
    def _compute_mask(sequence_lengths: tf.Tensor) -> tf.Tensor:
        mask = tf.sequence_mask(sequence_lengths, dtype=tf.float32)
        return mask

    def _emebed_dialogue(
        self, dialogue_in: tf.Tensor, sequence_lengths: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create dialogue level embedding and mask."""

        mask = self._compute_mask(sequence_lengths)
        if isinstance(dialogue_in, tf.SparseTensor):
            dialogue_in = self._tf_layers["sparse_to_dense.dialogue_features"](dialogue_in)

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

    @staticmethod
    def _last_token(x: tf.Tensor, sequence_lengths: tf.Tensor) -> tf.Tensor:
        last_sequence_index = tf.maximum(0, sequence_lengths - 1)
        batch_index = tf.range(tf.shape(last_sequence_index)[0])

        indices = tf.stack([batch_index, last_sequence_index], axis=1)
        return tf.expand_dims(tf.gather_nd(x, indices), 1)

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

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        batch = self.batch_to_model_data_format(batch_in, self.data_signature)
        sequence_lengths = tf.cast(
            tf.squeeze(batch["dialog_lengths"], axis=0), tf.int32
        )

        label_in = batch[LABEL_FEATURES]

        dialogue_in = self._combine_sparse_dense_features(
            batch[DIALOGUE_FEATURES], DIALOGUE_FEATURES
        )

        label_in = self._combine_sparse_dense_features(label_in, LABEL_FEATURES)
        label_in = tf.squeeze(label_in, axis=1)

        if self.max_history_tracker_featurizer_used:
            # add time dimension if max history featurizer is used
            label_in = label_in[:, tf.newaxis, :]

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

        dialogue_in = self._combine_sparse_dense_features(
            batch[DIALOGUE_FEATURES], DIALOGUE_FEATURES
        )

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


class HierarchicalTEDPolicy(Policy):
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
        # Number of units in transformer
        TRANSFORMER_SIZE: 512,
        # Number of transformer layers
        NUM_TRANSFORMER_LAYERS: 1,
        # Number of attention heads in transformer
        NUM_HEADS: 4,
        # Number of units in utterance transformer
        UTTERANCE_TRANSFORMER_SIZE: 512,
        # Number of utterance transformer layers
        UTTERANCE_NUM_TRANSFORMER_LAYERS: 1,
        # Number of attention heads in utterance transformer
        UTTERANCE_NUM_HEADS: 4,
        # If 'True' use key relative embeddings in attention
        KEY_RELATIVE_ATTENTION: False,
        # If 'True' use value relative embeddings in attention
        VALUE_RELATIVE_ATTENTION: False,
        # Max position for relative embeddings
        MAX_RELATIVE_POSITION: None,
        # ## Training parameters
        # Initial and final batch sizes:
        # Batch size will be linearly increased for each epoch.
        BATCH_SIZES: [64, 128],
        # Strategy used whenc creating batches.
        # Can be either 'sequence' or 'balanced'.
        BATCH_STRATEGY: BALANCED,
        # Number of epochs to train
        EPOCHS: 1,
        # Set random seed to any 'int' to get reproducible results
        RANDOM_SEED: None,
        # ## Parameters for embeddings
        # Dimension size of embedding vectors
        EMBEDDING_DIMENSION: 100,
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
        ENTITY_RECOGNITION: True
    }

    @staticmethod
    def _standard_featurizer(max_history: Optional[int] = None) -> TrackerFeaturizer:
        return MaxHistoryTrackerFeaturizer(
            E2ESingleStateFeaturizer(), max_history=max_history
        )

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = DEFAULT_POLICY_PRIORITY,
        max_history: Optional[int] = None,
        model: Optional[RasaModel] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        """Declare instance variables with default values."""

        if not featurizer:
            featurizer = self._standard_featurizer(max_history)

        super().__init__(featurizer, priority)

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
    def _label_ids_for_Y(data_Y: np.ndarray) -> np.ndarray:
        """Prepare Y data for training: extract label_ids.

        label_ids are indices of labels, while `data_Y` contains one-hot encodings.
        """

        return data_Y.argmax(axis=-1)

    # noinspection PyPep8Naming
    def _label_features_for_Y(self, label_ids: np.ndarray) -> np.ndarray:
        """Prepare Y data for training: features for label_ids."""

        all_label_features = self._label_data.get(LABEL_FEATURES)

        is_full_dialogue_featurizer_used = len(label_ids.shape) == 2
        if is_full_dialogue_featurizer_used:
            return np.stack(
                [
                    np.stack(
                        [all_label_features[label_idx] for label_idx in seq_label_ids]
                    )
                    for seq_label_ids in label_ids
                ]
            )

        if len(all_label_features) == 2:
            Y_sparse = np.array(
                [all_label_features[0][label_idx] for label_idx in label_ids]
            )
            Y_dense = np.stack(
                [all_label_features[1][label_idx] for label_idx in label_ids]
            )
        elif len(all_label_features) == 1:
            if isinstance(all_label_features[0][0], scipy.sparse.spmatrix):
                Y_sparse = np.array(
                    [all_label_features[0][label_idx] for label_idx in label_ids]
                )
                Y_dense = np.array([])
            elif isinstance(all_label_features[0][0], np.ndarray):
                Y_sparse = np.array([])
                Y_dense = np.array(
                    [all_label_features[1][label_idx] for label_idx in label_ids]
                )

        # max history featurizer is used
        return Y_sparse, Y_dense

    # noinspection PyPep8Naming
    def _create_model_data(
        self,
        data_X: Optional[np.ndarray],
        dialog_lengths: Optional[np.ndarray] = None,
        data_Y: Optional[np.ndarray] = None,
    ) -> RasaModelData:
        """Combine all model related data into RasaModelData."""

        label_ids = np.array([])
        Y_sparse, Y_dense = np.array([]), np.array([])

        if data_Y is not None:
            # label_ids = self._label_ids_for_Y(data_Y)
            label_ids = np.squeeze(data_Y, axis=-1)
            Y_sparse, Y_dense = self._label_features_for_Y(label_ids)
            # explicitly add last dimension to label_ids
            # to track correctly dynamic sequences
            label_ids = np.expand_dims(label_ids, -1)

        model_data = RasaModelData(label_key=LABEL_IDS)

        X_sparse_user = []
        X_sparse_bot = []
        X_dense_user = []
        X_dense_bot = []
        X_entities = []
        X_user_lens = []
        X_bot_lens = []

        for dial in data_X:
            sparse_state_user = []
            dense_state_user = []
            sparse_state_bot = []
            dense_state_bot = []
            entities = []
            user_lens = []
            bot_lens = []
            for state in dial:
                if state[0] is not None and state[1] is not None:
                    sparse_state_user.append(state[0].astype(np.float32))
                    sparse_state_bot.append(state[1].astype(np.float32))
                if state[2] is not None and state[3] is not None:
                    dense_state_user.append(state[2])
                    dense_state_bot.append(state[3])
                if state[0] is not None:
                    user_lens.append(state[0].shape[0])
                elif state[2] is not None:
                    user_lens.append(state[2].shape[0])
                if state[1] is not None:
                    bot_lens.append(state[1].shape[0])
                elif state[3] is not None:
                    bot_lens.append(state[3].shape[0])                        

                if state[4] is not None:
                    entities.append(state[4])
            if not sparse_state_user == []:
                sparse_state_user = np.array(sparse_state_user)
                sparse_state_bot = np.array(sparse_state_bot)
            if not dense_state_user == []:
                dense_state_user = np.array(dense_state_user)
                dense_state_bot = np.array(dense_state_bot)
            if not entities == []:
                entities = np.vstack(entities)
            if not bot_lens == []:
                bot_lens = np.expand_dims(np.array(bot_lens), 1)
            if not user_lens == []:
                user_lens = np.expand_dims(np.array(user_lens), 1)
            X_sparse_user.append(sparse_state_user)
            X_sparse_bot.append(sparse_state_bot)
            X_dense_user.append(dense_state_user)
            X_dense_bot.append(dense_state_bot)
            X_entities.append(entities)
            X_user_lens.append(user_lens)
            X_bot_lens.append(bot_lens)

        model_data.add_features(
            DIALOGUE_FEATURES,
            [np.array(X_sparse_user), np.array(X_sparse_bot), np.array(X_dense_user), np.array(X_dense_bot), np.array(X_entities)]
        )
        model_data.add_features(LABEL_FEATURES, [Y_sparse, Y_dense])
        model_data.add_features(LABEL_IDS, [label_ids])
        if dialog_lengths is not None:
            model_data.add_features("dialog_lengths", [np.array(dialog_lengths)])
        model_data.add_features(f"{DIALOGUE}_{USER}_lengths", [np.array(X_user_lens)])
        model_data.add_features(f"{DIALOGUE}_{BOT}_lengths", [np.array(X_bot_lens)])
        return model_data


    def collect_label_features(self, labels_example):
        from rasa.utils import train_utils
        sparse_features = []
        dense_features = []
        for feats in labels_example:
            if not feats[0] is None:
                sparse_feature = train_utils.sequence_to_sentence_features(feats[0])
                sparse_features.append(sparse_feature.tocsr().astype(np.float32))
            if not feats[1] is None:
                dense_features.append([feats[1]])

        sparse_features = np.array(sparse_features)
        dense_features = np.array(dense_features)
        return [sparse_features, dense_features]

    def _create_label_data(self, domain, kwargs) -> RasaModelData:
        # encode all label_ids with policies' featurizer
        # sparse_features, dense_features =
        labels_idx_examples = self.featurizer.state_featurizer.create_encoded_all_actions(
            domain, kwargs
        )
        labels_idx_examples = sorted(labels_idx_examples, key=lambda x: x[0])
        labels_example = [example for (_, example) in labels_idx_examples]
        label_features = self.collect_label_features(labels_example)

        label_data = RasaModelData()
        label_data.add_features(LABEL_FEATURES, label_features)

        label_ids = np.array([idx for (idx, _) in labels_idx_examples])
        # explicitly add last dimension to label_ids
        # to track correctly dynamic sequences
        label_data.add_features(LABEL_IDS, [np.expand_dims(label_ids, -1)])
        return label_data

    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any,
    ) -> None:
        """Train the policy on given training trackers."""
        kwargs['hierarchical'] = True

        # dealing with training data
        training_data = self.featurize_for_training(training_trackers, domain, **kwargs)

        self._label_data = self._create_label_data(domain, kwargs)

        # extract actual training data to feed to model
        model_data = self._create_model_data(
            training_data.X, training_data.true_length, training_data.y
        )
        if model_data.is_empty():
            logger.error(
                f"Can not train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the policy."
            )
            return

        # keep one example for persisting and loading
        self.data_example = model_data.first_data_example()

        self.model = HierarchicalTED(
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
        self, tracker: DialogueStateTracker, domain: Domain,
    ) -> List[float]:
        """Predict the next action the bot should take.

        Return the list of probabilities for the next actions.
        """

        if self.model is None:
            return self._default_predictions(domain)
        kwargs = {}
        kwargs['hierarchical'] = True
        # create model data from tracker
        data_X = self.featurizer.create_X([tracker], domain, **kwargs)
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
        # using pickle to be able to store and load both sparse and dense data_example
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
        # setting path variable for featurizer so that we can load the nlu pipeline with
        # DIET from there;
        featurizer.path = path

        if not (model_path / f"{SAVE_MODEL_FILE_NAME}.data_example.pkl").is_file():
            return cls(featurizer=featurizer)
        # using pickle to be able to store and load both sparse and dense data_example
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

        model = HierarchicalTED.load(
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



class HierarchicalTED(RasaModel):
    def __init__(
        self,
        data_signature: Dict[Text, List[FeatureSignature]],
        config: Dict[Text, Any],
        max_history_tracker_featurizer_used: bool,
        label_data: RasaModelData,
    ) -> None:
        super().__init__(name="TED", random_seed=config[RANDOM_SEED])

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
        self._set_optimizer(tf.keras.optimizers.Adam())

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
        if DIALOGUE_FEATURES not in self.data_signature:
            raise ValueError(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if LABEL_FEATURES not in self.data_signature:
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
                # if dense features are present
                # use the feature dimension of the dense features
                # dense_dim = shape[-1]
                dense_dim = dense_dim

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

        self._prepare_sparse_dense_layers(
            self.data_signature["dialogue_features"],
            "dialogue_features",
            self.config[REGULARIZATION_CONSTANT],
            100,
        )

        for is_sparse, shape in self.data_signature["label_features"]:
            if is_sparse:
                sparse_dim_label_features = shape[-1]
            else:
                sparse_dim_label_features = 100

        self._prepare_sparse_dense_layers(
            self.data_signature["label_features"],
            "label_features",
            self.config[REGULARIZATION_CONSTANT],
            100,
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
        self._tf_layers[f"{LABEL}_transformer"] = TransformerEncoder(
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
            name=LABEL + "_encoder",
        )
        self._tf_layers[f"transformer"] = TransformerEncoder(
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
        self._tf_layers[f"transformer_{BOT}_utts"] = TransformerEncoder(
            self.config[UTTERANCE_NUM_TRANSFORMER_LAYERS],
            self.config[UTTERANCE_TRANSFORMER_SIZE],
            self.config[UTTERANCE_NUM_HEADS],
            self.config[UTTERANCE_TRANSFORMER_SIZE] * 4,
            self.config[REGULARIZATION_CONSTANT],
            dropout_rate=self.config[DROP_RATE_DIALOGUE],
            attention_dropout_rate=self.config[DROP_RATE_ATTENTION],
            sparsity=self.config[WEIGHT_SPARSITY],
            unidirectional=True,
            use_key_relative_position=self.config[KEY_RELATIVE_ATTENTION],
            use_value_relative_position=self.config[VALUE_RELATIVE_ATTENTION],
            max_relative_position=self.config[MAX_RELATIVE_POSITION],
            name=BOT + "_utterance_encoder",
        )
        self._tf_layers[f"transformer_{USER}_utts"] = TransformerEncoder(
            self.config[UTTERANCE_NUM_TRANSFORMER_LAYERS],
            self.config[UTTERANCE_TRANSFORMER_SIZE],
            self.config[UTTERANCE_NUM_HEADS],
            self.config[UTTERANCE_TRANSFORMER_SIZE] * 4,
            self.config[REGULARIZATION_CONSTANT],
            dropout_rate=self.config[DROP_RATE_DIALOGUE],
            attention_dropout_rate=self.config[DROP_RATE_ATTENTION],
            sparsity=self.config[WEIGHT_SPARSITY],
            unidirectional=True,
            use_key_relative_position=self.config[KEY_RELATIVE_ATTENTION],
            use_value_relative_position=self.config[VALUE_RELATIVE_ATTENTION],
            max_relative_position=self.config[MAX_RELATIVE_POSITION],
            name=USER + "_utterance_encoder",
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
        
        if self.config[ENTITY_RECOGNITION]:
            print('LAYER PREPARE')

    def _create_all_labels_embed(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_labels = self.tf_label_data[LABEL_FEATURES]
        all_labels = self._combine_sparse_dense_features(all_labels, LABEL_FEATURES)
        all_labels = self._tf_layers[f"{LABEL}_transformer"](all_labels)
        all_labels = tfa.activations.gelu(all_labels)
        all_labels = all_labels[:,-1,:]
        all_labels_embed = self._embed_label(all_labels)
        return all_labels, all_labels_embed

    def _embed_utterances(self, utterances_in: tf.Tensor, name, utterance_lengths) -> tf.Tensor:
        batch_size, max_dialog_length, max_sentence_length, num_features = utterances_in.shape.as_list() 
        utterances = tf.reshape(utterances_in, [-1, tf.shape(utterances_in)[2], num_features])
        # sequence_lengths = tf.squeeze(tf.reshape(sequence_lengths, [-1, 1]), -1)
        utterances_embedded = self._tf_layers[f"transformer_"+name+"_utts"](utterances)
        utterances_embedded = tfa.activations.gelu(utterances_embedded)
        utterances_embedded = self._last_token(utterances_embedded, utterance_lengths)
        utterances_embedded = tf.reshape(utterances_embedded, [tf.shape(utterances_in)[0], tf.shape(utterances_in)[1], utterances_embedded.shape[-1]])
        return utterances_embedded



    @staticmethod
    def _compute_mask(sequence_lengths: tf.Tensor) -> tf.Tensor:
        mask = tf.sequence_mask(sequence_lengths, dtype=tf.float32)
        return mask

    def _emebed_dialogue(
        self, dialogue_in: [tf.Tensor], sequence_lengths: tf.Tensor, user_lengths, bot_lengths
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create dialogue level embedding and mask."""

        mask = self._compute_mask(sequence_lengths)

        dialogue = self._tf_layers[f"ffnn.{DIALOGUE}"](dialogue_in, self._training)
        utterance_features_bot = []
        utterance_features_user = []
        state_level_dialog_features = []
        i=0
        for feats in dialogue:
            if len(feats.shape)==4:
                if i%2 == 0:
                    utterance_features_user.append(feats)
                    i+=1
                else:
                    utterance_features_bot.append(feats)
                    i+=1
            else:
                state_level_dialog_features.append(feats)
        user_utterance_features = tf.concat(utterance_features_user, -1)
        bot_utterance_features = tf.concat(utterance_features_bot, -1)
        user_utterances_embedded = self._embed_utterances(user_utterance_features, USER, user_lengths)
        bot_utterance_embedded = self._embed_utterances(bot_utterance_features, BOT, bot_lengths)
        dialogue = tf.concat([user_utterances_embedded, bot_utterance_embedded] +state_level_dialog_features, -1)
        
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

    @staticmethod
    def _last_token(x: tf.Tensor, sequence_lengths: tf.Tensor) -> tf.Tensor:
        last_sequence_index = tf.maximum(0, sequence_lengths - 1)
        batch_index = tf.range(tf.shape(last_sequence_index)[0])

        indices = tf.stack([batch_index, last_sequence_index], axis=1)
        return tf.expand_dims(tf.gather_nd(x, indices), 1)

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
        # this is because dialog features will have some which are utterance 
        # level and some which are turn level, e.g., entities
        if name == LABEL_FEATURES:
            return tf.concat(dense_features, axis=-1)
        else:
            return dense_features

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        batch = self.batch_to_model_data_format(batch_in, self.data_signature)
        sequence_lengths = tf.cast(
            tf.squeeze(batch["dialog_lengths"], axis=0), tf.int32
        )
        user_lengths = batch[f"{DIALOGUE}_{USER}_lengths"]
        bot_lengths = batch[f"{DIALOGUE}_{BOT}_lengths"]
        user_lengths = tf.cast(tf.reshape(user_lengths, [-1]), tf.int32)
        bot_lengths = tf.cast(tf.reshape(bot_lengths, [-1]), tf.int32)

        label_in = batch[LABEL_FEATURES]

        label_in = self._combine_sparse_dense_features(label_in, LABEL_FEATURES)
        # label_in = tf.squeeze(label_in, axis=1)
        batch_dialogue_features = batch[DIALOGUE_FEATURES]

        dialogue_in = self._combine_sparse_dense_features(
            batch_dialogue_features, DIALOGUE_FEATURES
        )

        all_labels, all_labels_embed = self._create_all_labels_embed()

        dialogue_embed, mask = self._emebed_dialogue(dialogue_in, sequence_lengths, user_lengths, bot_lengths)
        label_in = self._tf_layers[f"{LABEL}_transformer"](label_in)
        label_in = tfa.activations.gelu(label_in)
        label_in = label_in[:,-1,:]
        if self.max_history_tracker_featurizer_used:
            # add time dimension if max history featurizer is used
            label_in = label_in[:, tf.newaxis, :]
        
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

        dialogue_in = self._combine_sparse_dense_features(
            batch[DIALOGUE_FEATURES], DIALOGUE_FEATURES
        )

        sequence_lengths = tf.expand_dims(tf.shape(dialogue_in[0])[1], axis=0)
        user_lengths = batch[f"{DIALOGUE}_{USER}_lengths"]
        bot_lengths = batch[f"{DIALOGUE}_{BOT}_lengths"]
        user_lengths = tf.cast(tf.reshape(user_lengths, [-1]), tf.int32)
        bot_lengths = tf.cast(tf.reshape(bot_lengths, [-1]), tf.int32)

        if self.all_labels_embed is None:
            _, self.all_labels_embed = self._create_all_labels_embed()

        dialogue_embed, mask = self._emebed_dialogue(dialogue_in, sequence_lengths, user_lengths, bot_lengths)

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
