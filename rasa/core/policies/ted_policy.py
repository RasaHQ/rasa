import copy
import logging
import os
import pickle

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from typing import Any, List, Optional, Text, Dict, Tuple, Union

import rasa.utils.io
from rasa.core.domain import Domain
from rasa.core.featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    LabelTokenizerSingleStateFeaturizer,
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
    NEG_MARGIN_SCALE,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    DROPRATE_DIALOGUE,
    DROPRATE_LABEL,
    DROPRATE_ATTENTION,
    WEIGHT_SPARSITY,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
)


logger = logging.getLogger(__name__)


class TEDPolicy(Policy):
    """Transformer Embedding Dialogue Policy (TEDP)

    The policy used in our paper https://arxiv.org/abs/1910.00486
    """

    SUPPORTS_ONLINE_TRAINING = True

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # nn architecture
        # a list of hidden layers sizes before dialogue and action embed layers
        # number of hidden layers is equal to the length of this list
        HIDDEN_LAYERS_SIZES: {DIALOGUE: [], LABEL: []},
        # number of units in transformer
        TRANSFORMER_SIZE: 128,
        # number of transformer layers
        NUM_TRANSFORMER_LAYERS: 1,
        # number of attention heads in transformer
        NUM_HEADS: 4,
        # if true use key relative embeddings in attention
        KEY_RELATIVE_ATTENTION: False,
        # if true use key relative embeddings in attention
        VALUE_RELATIVE_ATTENTION: False,
        # max position for relative embeddings
        MAX_RELATIVE_POSITION: None,
        # training parameters
        # initial and final batch sizes:
        # batch size will be linearly increased for each epoch
        BATCH_SIZES: [8, 32],
        # how to create batches
        BATCH_STRATEGY: "balanced",  # string 'sequence' or 'balanced'
        # number of epochs
        EPOCHS: 1,
        # set random seed to any int to get reproducible results
        RANDOM_SEED: None,
        # embedding parameters
        # dimension size of embedding vectors
        EMBEDDING_DIMENSION: 20,
        # the type of the similarity
        NUM_NEG: 20,
        # flag if minimize only maximum similarity over incorrect labels
        SIMILARITY_TYPE: "auto",  # string 'auto' or 'cosine' or 'inner'
        # the type of the loss function
        LOSS_TYPE: "softmax",  # string 'softmax' or 'margin'
        # number of top actions to normalize scores for softmax loss_type
        # set to 0 to turn off normalization
        RANKING_LENGTH: 10,
        # how similar the algorithm should try
        # to make embedding vectors for correct labels
        MAX_POS_SIM: 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect labels
        MAX_NEG_SIM: -0.2,  # should be -1.0 < ... < 1.0 for 'cosine'
        # the number of incorrect labels, the algorithm will minimize
        # their similarity to the user input during training
        USE_MAX_NEG_SIM: True,  # flag which loss function to use
        # scale loss inverse proportionally to confidence of correct prediction
        SCALE_LOSS: True,
        # regularization
        # the scale of regularization
        REGULARIZATION_CONSTANT: 0.001,
        # the scale of how important is to minimize the maximum similarity
        # between embeddings of different labels
        NEG_MARGIN_SCALE: 0.8,
        # dropout rate for dial nn
        DROPRATE_DIALOGUE: 0.1,
        # dropout rate for bot nn
        DROPRATE_LABEL: 0.0,
        # dropout rate for attention
        DROPRATE_ATTENTION: 0,
        # sparsity of the weights in dense layers
        WEIGHT_SPARSITY: 0.8,
        # visualization of accuracy
        # how often calculate validation accuracy
        EVAL_NUM_EPOCHS: 20,  # small values may hurt performance
        # how many examples to use for hold out validation set
        EVAL_NUM_EXAMPLES: 0,  # large values may hurt performance
    }
    # end default properties (DOC MARKER - don't remove)

    @staticmethod
    def _standard_featurizer(max_history: Optional[int] = None) -> TrackerFeaturizer:
        if max_history is None:
            return FullDialogueTrackerFeaturizer(LabelTokenizerSingleStateFeaturizer())
        else:
            return MaxHistoryTrackerFeaturizer(
                LabelTokenizerSingleStateFeaturizer(), max_history=max_history
            )

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = DEFAULT_POLICY_PRIORITY,
        max_history: Optional[int] = None,
        model: Optional[RasaModel] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:
        """Declare instance variables with default values"""

        if not featurizer:
            featurizer = self._standard_featurizer(max_history)

        super().__init__(featurizer, priority)

        self._load_params(**kwargs)

        self.model = model

        self._label_data = None
        self.data_example = None

    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        self.config = copy.deepcopy(self.defaults)
        self.config.update(kwargs)

        self.config = train_utils.check_deprecated_options(self.config)

        self.config = train_utils.update_similarity_type(self.config)

        if self.config[EVAL_NUM_EPOCHS] == -1:
            # magic value -1 is used to set evaluation to number of epochs
            self.config[EVAL_NUM_EPOCHS] = self.config[EPOCHS]
        elif self.config[EVAL_NUM_EPOCHS] < 1:
            raise ValueError(
                f"'{EVAL_NUM_EXAMPLES}' is set to '{self.config[EVAL_NUM_EPOCHS]}'. "
                f"Only values > 1 are allowed for this configuration value."
            )

    # data helpers
    # noinspection PyPep8Naming
    @staticmethod
    def _label_ids_for_Y(data_Y: np.ndarray) -> np.ndarray:
        """Prepare Y data for training: extract label_ids."""

        return data_Y.argmax(axis=-1)

    # noinspection PyPep8Naming
    def _label_features_for_Y(self, label_ids: np.ndarray) -> np.ndarray:
        """Prepare Y data for training: features for label_ids."""

        # full dialogue featurizer is used
        if len(label_ids.shape) == 2:
            return np.stack(
                [
                    np.stack(
                        [
                            self._label_data.get("label_features")[0][label_idx]
                            for label_idx in seq_label_ids
                        ]
                    )
                    for seq_label_ids in label_ids
                ]
            )

        # max history featurizer is used
        return np.stack(
            [
                self._label_data.get("label_features")[0][label_idx]
                for label_idx in label_ids
            ]
        )

    # noinspection PyPep8Naming
    def _create_model_data(
        self, data_X: np.ndarray, data_Y: Optional[np.ndarray] = None
    ) -> RasaModelData:
        """Combine all model related data into RasaModelData."""

        label_ids = np.array([])
        Y = np.array([])

        if data_Y is not None:
            label_ids = self._label_ids_for_Y(data_Y)
            Y = self._label_features_for_Y(label_ids)
            # explicitly add last dimension to label_ids
            # to track correctly dynamic sequences
            label_ids = np.expand_dims(label_ids, -1)

        model_data = RasaModelData(label_key="label_ids")
        model_data.add_features("dialogue_features", [data_X])
        model_data.add_features("label_features", [Y])
        model_data.add_features("label_ids", [label_ids])

        return model_data

    def _create_label_data(self, domain: Domain) -> RasaModelData:
        # encode all label_ids with policies' featurizer
        state_featurizer = self.featurizer.state_featurizer
        all_labels = state_featurizer.create_encoded_all_actions(domain)
        all_labels = all_labels.astype(np.float32)

        label_data = RasaModelData()
        label_data.add_features("label_features", [all_labels])
        return label_data

    # training methods
    def train(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any,
    ) -> None:
        """Train the policy on given training trackers."""

        # set numpy random seed
        np.random.seed(self.config[RANDOM_SEED])

        # dealing with training data
        training_data = self.featurize_for_training(training_trackers, domain, **kwargs)

        self._label_data = self._create_label_data(domain)

        # check if number of negatives is less than number of label_ids
        logger.debug(
            f"Check if num_neg {self.config[NUM_NEG]} is smaller "
            f"than number of label_ids {domain.num_actions}, "
            f"else set num_neg to the number of label_ids - 1."
        )
        self.config[NUM_NEG] = min(self.config[NUM_NEG], domain.num_actions - 1)

        # extract actual training data to feed to model
        model_data = self._create_model_data(training_data.X, training_data.y)
        if model_data.is_empty():
            logger.error(
                f"Can not train '{self.__class__.__name__}'. No data was provided. "
                f"Skipping training of the policy."
            )
            return

        # keep one example for persisting and loading
        self.data_example = {k: [v[:1] for v in vs] for k, vs in model_data.items()}

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

    def continue_training(
        self,
        training_trackers: List[DialogueStateTracker],
        domain: Domain,
        **kwargs: Any,
    ) -> None:
        """Continue training an already trained policy."""

        batch_size = kwargs.get("batch_size", 5)
        epochs = kwargs.get("epochs", 50)

        training_data = self._training_data_for_continue_training(
            batch_size, training_trackers, domain
        )

        model_data = self._create_model_data(training_data.X, training_data.y)

        self.model.fit(
            model_data,
            epochs,
            [batch_size],
            self.config[EVAL_NUM_EXAMPLES],
            self.config[EVAL_NUM_EPOCHS],
            batch_strategy=self.config[BATCH_STRATEGY],
        )

    def predict_action_probabilities(
        self, tracker: DialogueStateTracker, domain: Domain
    ) -> List[float]:
        """Predict the next action the bot should take.

        Return the list of probabilities for the next actions.
        """
        if self.model is None:
            return [0.0] * domain.num_actions

        # create model data from tracker
        data_X = self.featurizer.create_X([tracker], domain)
        model_data = self._create_model_data(data_X)

        output = self.model.predict(model_data)

        confidence = output["action_scores"].numpy()
        confidence = confidence[0, -1, :]

        if self.config[LOSS_TYPE] == "softmax" and self.config[RANKING_LENGTH] > 0:
            confidence = train_utils.normalize(confidence, self.config[RANKING_LENGTH])

        return confidence.tolist()

    def persist(self, path: Text):
        """Persists the policy to a storage."""

        if self.model is None:
            logger.debug(
                "Method `persist(...)` was called "
                "without a trained model present. "
                "Nothing to persist then!"
            )
            return

        file_name = "ted_policy"
        tf_model_file = os.path.join(path, f"{file_name}.tf_model")

        rasa.utils.io.create_directory_for_file(tf_model_file)

        self.featurizer.persist(path)

        self.model.save(tf_model_file)

        with open(os.path.join(path, file_name + ".priority.pkl"), "wb") as f:
            pickle.dump(self.priority, f)

        with open(os.path.join(path, file_name + ".meta.pkl"), "wb") as f:
            pickle.dump(self.config, f)

        with open(os.path.join(path, file_name + ".data_example.pkl"), "wb") as f:
            pickle.dump(self.data_example, f)

        with open(os.path.join(path, file_name + ".label_data.pkl"), "wb") as f:
            pickle.dump(self._label_data, f)

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

        file_name = "TED_policy"
        tf_model_file = os.path.join(path, f"{file_name}.tf_model")

        featurizer = TrackerFeaturizer.load(path)

        if not os.path.exists(os.path.join(path, file_name + ".data_example.pkl")):
            return cls(featurizer=featurizer)

        with open(os.path.join(path, file_name + ".data_example.pkl"), "rb") as f:
            model_data_example = RasaModelData(
                label_key="label_ids", data=pickle.load(f)
            )

        with open(os.path.join(path, file_name + ".label_data.pkl"), "rb") as f:
            label_data = pickle.load(f)

        with open(os.path.join(path, file_name + ".meta.pkl"), "rb") as f:
            meta = pickle.load(f)

        with open(os.path.join(path, file_name + ".priority.pkl"), "rb") as f:
            priority = pickle.load(f)

        meta = train_utils.update_similarity_type(meta)

        model = TED.load(
            tf_model_file,
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
            label_key="label_ids",
            data={k: vs for k, vs in model_data_example.items() if "dialogue" in k},
        )
        model.build_for_predict(predict_data_example)

        return cls(featurizer=featurizer, priority=priority, model=model, **meta)


# pytype: disable=key-error


class TED(RasaModel):
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
            k: vs for k, vs in data_signature.items() if "dialogue" in k
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
        self._tf_layers = {}
        self._prepare_layers()

    def _check_data(self) -> None:
        if "dialogue_features" not in self.data_signature:
            raise ValueError(
                f"No text features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )
        if "label_features" not in self.data_signature:
            raise ValueError(
                f"No label features specified. "
                f"Cannot train '{self.__class__.__name__}' model."
            )

    def _prepare_layers(self) -> None:
        self._tf_layers["loss.label"] = layers.DotProductLoss(
            self.config[NUM_NEG],
            self.config[LOSS_TYPE],
            self.config[MAX_POS_SIM],
            self.config[MAX_NEG_SIM],
            self.config[USE_MAX_NEG_SIM],
            self.config[NEG_MARGIN_SCALE],
            self.config[SCALE_LOSS],
            # set to 1 to get deterministic behaviour
            parallel_iterations=1 if self.random_seed is not None else 1000,
        )
        self._tf_layers["ffnn.dialogue"] = layers.Ffnn(
            self.config[HIDDEN_LAYERS_SIZES][DIALOGUE],
            self.config[DROPRATE_DIALOGUE],
            self.config[REGULARIZATION_CONSTANT],
            self.config[WEIGHT_SPARSITY],
            layer_name_suffix=DIALOGUE,
        )
        self._tf_layers["ffnn.label"] = layers.Ffnn(
            self.config[HIDDEN_LAYERS_SIZES][LABEL],
            self.config[DROPRATE_LABEL],
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
            dropout_rate=self.config[DROPRATE_DIALOGUE],
            attention_dropout_rate=self.config[DROPRATE_ATTENTION],
            sparsity=self.config[WEIGHT_SPARSITY],
            unidirectional=True,
            use_key_relative_position=self.config[KEY_RELATIVE_ATTENTION],
            use_value_relative_position=self.config[VALUE_RELATIVE_ATTENTION],
            max_relative_position=self.config[MAX_RELATIVE_POSITION],
            name=DIALOGUE + "_encoder",
        )
        self._tf_layers["embed.dialogue"] = layers.Embed(
            self.config[EMBEDDING_DIMENSION],
            self.config[REGULARIZATION_CONSTANT],
            DIALOGUE,
            self.config[SIMILARITY_TYPE],
        )
        self._tf_layers["embed.label"] = layers.Embed(
            self.config[EMBEDDING_DIMENSION],
            self.config[REGULARIZATION_CONSTANT],
            LABEL,
            self.config[SIMILARITY_TYPE],
        )

    def _create_all_labels_embed(self) -> Tuple[tf.Tensor, tf.Tensor]:
        all_labels = self.tf_label_data["label_features"][0]
        all_labels_embed = self._embed_label(all_labels)

        return all_labels, all_labels_embed

    def _emebed_dialogue(self, dialogue_in: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Create dialogue level embedding and mask."""

        # mask different length sequences
        # if there is at least one `-1` it should be masked
        mask = tf.sign(tf.reduce_max(dialogue_in, -1) + 1)

        dialogue = self._tf_layers["ffnn.dialogue"](dialogue_in, self._training)
        dialogue_transformed = self._tf_layers["transformer"](
            dialogue, 1 - tf.expand_dims(mask, axis=-1), self._training
        )
        dialogue_transformed = tfa.activations.gelu(dialogue_transformed)

        if self.max_history_tracker_featurizer_used:
            # pick last label if max history featurizer is used
            dialogue_transformed = dialogue_transformed[:, -1:, :]
            mask = mask[:, -1:]

        dialogue_embed = self._tf_layers["embed.dialogue"](dialogue_transformed)

        return dialogue_embed, mask

    def _embed_label(self, label_in: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        label = self._tf_layers["ffnn.label"](label_in, self._training)
        return self._tf_layers["embed.label"](label)

    def batch_loss(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> tf.Tensor:
        batch = self.batch_to_model_data_format(batch_in, self.data_signature)

        dialogue_in = batch["dialogue_features"][0]
        label_in = batch["label_features"][0]

        if self.max_history_tracker_featurizer_used:
            # add time dimension if max history featurizer is used
            label_in = label_in[:, tf.newaxis, :]

        all_labels, all_labels_embed = self._create_all_labels_embed()

        dialogue_embed, mask = self._emebed_dialogue(dialogue_in)
        label_embed = self._embed_label(label_in)

        loss, acc = self._tf_layers["loss.label"](
            dialogue_embed, label_embed, label_in, all_labels_embed, all_labels, mask
        )

        self.action_loss.update_state(loss)
        self.action_acc.update_state(acc)

        return loss

    def batch_predict(
        self, batch_in: Union[Tuple[tf.Tensor], Tuple[np.ndarray]]
    ) -> Dict[Text, tf.Tensor]:
        batch = self.batch_to_model_data_format(batch_in, self.predict_data_signature)

        dialogue_in = batch["dialogue_features"][0]

        if self.all_labels_embed is None:
            _, self.all_labels_embed = self._create_all_labels_embed()

        dialogue_embed, mask = self._emebed_dialogue(dialogue_in)

        sim_all = self._tf_layers["loss.label"].sim(
            dialogue_embed[:, :, tf.newaxis, :],
            self.all_labels_embed[tf.newaxis, tf.newaxis, :, :],
            mask,
        )

        scores = self._tf_layers["loss.label"].confidence_from_sim(
            sim_all, self.config[SIMILARITY_TYPE]
        )

        return {"action_scores": scores}


# pytype: enable=key-error
