import copy
import logging
import os
import pickle

import numpy as np
from typing import Any, List, Optional, Text, Dict, Tuple, Union, Callable

import rasa.utils.io
from rasa.core.domain import Domain
from rasa.core.featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    LabelTokenizerSingleStateFeaturizer,
    MaxHistoryTrackerFeaturizer,
)
from rasa.core.policies.policy import Policy
from rasa.core.constants import DEFAULT_POLICY_PRIORITY
from rasa.core.trackers import DialogueStateTracker
from rasa.utils import train_utils

import tensorflow as tf

# avoid warning println on contrib import - remove for tf 2
from rasa.utils.tensorflow import tf_models, tf_layers
from rasa.utils.tensorflow.tf_model_data import RasaModelData

logger = logging.getLogger(__name__)


# constants - configuration parameters
HIDDEN_LAYERS_SIZES_PRE_DIAL = "hidden_layers_sizes_pre_dial"
HIDDEN_LAYERS_SIZES_BOT = "hidden_layers_sizes_bot"
TRANSFORMER_SIZE = "transformer_size"
NUM_TRANSFORMER_LAYERS = "number_of_transformer_layers"
NUM_HEADS = "number_of_attention_heads"
POS_ENCODING = "positional_encoding"
MAX_SEQ_LENGTH = "maximum_sequence_length"
BATCH_SIZES = "batch_sizes"
BATCH_STRATEGY = "batch_strategy"
EPOCHS = "epochs"
RANDOM_SEED = "random_seed"
EMBED_DIM = "embedding_dimension"
NUM_NEG = "number_of_negative_examples"
SIMILARITY_TYPE = "similarity_type"
LOSS_TYPE = "loss_type"
MU_POS = "maximum_positive_similarity"
MU_NEG = "maximum_negative_similarity"
USE_MAX_SIM_NEG = "use_maximum_negative_similarity"
SCALE_LOSS = "scale_loss"
C2 = "l2_regularization"
C_EMB = "c_emb"
DROPRATE_DIAL = "droprate_dial"
DROPRATE_BOT = "droprate_bot"
EVAL_NUM_EPOCHS = "evaluate_every_number_of_epochs"
EVAL_NUM_EXAMPLES = "evaluate_on_number_of_examples"


class EmbeddingPolicy(Policy):
    """Transformer Embedding Dialogue Policy (TEDP)

    Transformer version of the REDP used in our paper https://arxiv.org/abs/1811.11707
    """

    SUPPORTS_ONLINE_TRAINING = True

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # nn architecture
        # a list of hidden layers sizes before user embed layer
        # number of hidden layers is equal to the length of this list
        HIDDEN_LAYERS_SIZES_PRE_DIAL: [],
        # a list of hidden layers sizes before bot embed layer
        # number of hidden layers is equal to the length of this list
        HIDDEN_LAYERS_SIZES_BOT: [],
        # number of units in transformer
        TRANSFORMER_SIZE: 128,
        # number of transformer layers
        NUM_TRANSFORMER_LAYERS: 1,
        # type of positional encoding in transformer
        POS_ENCODING: "timing",  # string 'timing' or 'emb'
        # max sequence length if pos_encoding='emb'
        MAX_SEQ_LENGTH: 256,
        # number of attention heads in transformer
        NUM_HEADS: 4,
        # training parameters
        # initial and final batch sizes:
        # batch size will be linearly increased for each epoch
        BATCH_SIZES: [8, 32],
        # how to create batches
        BATCH_STRATEGY: "balanced",  # string 'sequence' or 'balanced'
        # number of epochs
        EPOCHS: 10,
        # set random seed to any int to get reproducible results
        RANDOM_SEED: None,
        # embedding parameters
        # dimension size of embedding vectors
        EMBED_DIM: 20,
        # the type of the similarity
        NUM_NEG: 20,
        # flag if minimize only maximum similarity over incorrect labels
        SIMILARITY_TYPE: "auto",  # string 'auto' or 'cosine' or 'inner'
        # the type of the loss function
        LOSS_TYPE: "softmax",  # string 'softmax' or 'margin'
        # how similar the algorithm should try
        # to make embedding vectors for correct labels
        MU_POS: 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect labels
        MU_NEG: -0.2,  # should be -1.0 < ... < 1.0 for 'cosine'
        # the number of incorrect labels, the algorithm will minimize
        # their similarity to the user input during training
        USE_MAX_SIM_NEG: True,  # flag which loss function to use
        # scale loss inverse proportionally to confidence of correct prediction
        SCALE_LOSS: True,
        # regularization
        # the scale of L2 regularization
        C2: 0.001,
        # the scale of how important is to minimize the maximum similarity
        # between embeddings of different labels
        C_EMB: 0.8,
        # dropout rate for dial nn
        DROPRATE_DIAL: 0.1,
        # dropout rate for bot nn
        DROPRATE_BOT: 0.0,
        # visualization of accuracy
        # how often calculate validation accuracy
        EVAL_NUM_EPOCHS: 20,  # small values may hurt performance
        # how many examples to use for hold out validation set
        EVAL_NUM_EXAMPLES: 0,  # large values may hurt performance
    }
    # end default properties (DOC MARKER - don't remove)

    @staticmethod
    def _standard_featurizer(max_history: Optional[int] = None) -> "TrackerFeaturizer":
        if max_history is None:
            return FullDialogueTrackerFeaturizer(LabelTokenizerSingleStateFeaturizer())
        else:
            return MaxHistoryTrackerFeaturizer(
                LabelTokenizerSingleStateFeaturizer(), max_history=max_history
            )

    def __init__(
        self,
        featurizer: Optional["TrackerFeaturizer"] = None,
        priority: int = DEFAULT_POLICY_PRIORITY,
        max_history: Optional[int] = None,
        model: Optional[tf_models.RasaModel] = None,
        predict_func: Optional[Callable] = None,
        **kwargs: Any,
    ) -> None:
        """Declare instant variables with default values"""

        if not featurizer:
            featurizer = self._standard_featurizer(max_history)
        super().__init__(featurizer, priority)

        self._load_params(**kwargs)

        self.model = model
        self.predict_func = predict_func

        # encode all label_ids with numbers
        self._encoded_all_label_ids = None

        self.data_example = None

        self._tf_config = train_utils.load_tf_config(self.config)

    # init helpers
    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        self.config = copy.deepcopy(self.defaults)
        self.config.update(kwargs)

        if self.config[SIMILARITY_TYPE] == "auto":
            if self.config[LOSS_TYPE] == "softmax":
                self.config[SIMILARITY_TYPE] = "inner"
            elif self.config[LOSS_TYPE] == "margin":
                self.config[SIMILARITY_TYPE] = "cosine"

        if self.config[EVAL_NUM_EPOCHS] < 1:
            self.config[EVAL_NUM_EPOCHS] = self.config[EPOCHS]

    # data helpers
    # noinspection PyPep8Naming
    @staticmethod
    def _label_ids_for_Y(data_Y: np.ndarray) -> np.ndarray:
        """Prepare Y data for training: extract label_ids."""

        return data_Y.argmax(axis=-1)

    # noinspection PyPep8Naming
    def _label_features_for_Y(self, label_ids: np.ndarray) -> np.ndarray:
        """Prepare Y data for training: features for label_ids."""

        if len(label_ids.shape) == 2:  # full dialogue featurizer is used
            return np.stack(
                [
                    np.stack(
                        [
                            self._encoded_all_label_ids[label_idx]
                            for label_idx in seq_label_ids
                        ]
                    )
                    for seq_label_ids in label_ids
                ]
            )
        else:  # max history featurizer is used
            return np.stack(
                [self._encoded_all_label_ids[label_idx] for label_idx in label_ids]
            )

    # noinspection PyPep8Naming
    def _create_model_data(
        self, data_X: np.ndarray, data_Y: Optional[np.ndarray] = None
    ) -> RasaModelData:
        """Combine all tf session related data into dict."""
        if data_Y is not None:
            # training time
            label_ids = self._label_ids_for_Y(data_Y)
            Y = self._label_features_for_Y(label_ids)
            # explicitly add last dimension to label_ids
            # to track correctly dynamic sequences
            label_ids = np.expand_dims(label_ids, -1)
        else:
            # prediction time
            label_ids = None
            Y = None

        return RasaModelData(
            label_key="action_ids",
            data={
                "dialogue_features": [data_X],
                "bot_features": [Y],
                "action_ids": [label_ids],
            },
        )

    # training methods
    def train(
        self,
        training_trackers: List["DialogueStateTracker"],
        domain: "Domain",
        **kwargs: Any,
    ) -> None:
        """Train the policy on given training trackers."""

        logger.debug("Started training embedding policy.")

        # set numpy random seed
        np.random.seed(self.config[RANDOM_SEED])

        # dealing with training data
        training_data = self.featurize_for_training(training_trackers, domain, **kwargs)

        # encode all label_ids with policies' featurizer
        state_featurizer = self.featurizer.state_featurizer
        self._encoded_all_label_ids = state_featurizer.create_encoded_all_actions(
            domain
        )

        # check if number of negatives is less than number of label_ids
        logger.debug(
            "Check if num_neg {} is smaller "
            "than number of label_ids {}, "
            "else set num_neg to the number of label_ids - 1"
            "".format(self.config[NUM_NEG], domain.num_actions)
        )
        # noinspection PyAttributeOutsideInit
        self.config[NUM_NEG] = min(self.config[NUM_NEG], domain.num_actions - 1)

        # extract actual training data to feed to tf session
        model_data = self._create_model_data(training_data.X, training_data.y)

        # keep one example for persisting and loading
        self.data_example = {k: [v[:1] for v in vs] for k, vs in model_data.items()}

        self.model = TED(
            self.config,
            isinstance(self.featurizer, MaxHistoryTrackerFeaturizer),
            self._encoded_all_label_ids,
        )

        self.model.fit(
            model_data,
            self.config[EPOCHS],
            self.config[BATCH_SIZES],
            self.config[EVAL_NUM_EXAMPLES],
            self.config[EVAL_NUM_EPOCHS],
            batch_strategy=self.config[BATCH_STRATEGY],
            random_seed=self.config[RANDOM_SEED],
        )

    def continue_training(
        self,
        training_trackers: List["DialogueStateTracker"],
        domain: "Domain",
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
            random_seed=self.config[RANDOM_SEED],
        )

    def predict_action_probabilities(
        self, tracker: "DialogueStateTracker", domain: "Domain"
    ) -> List[float]:
        """Predict the next action the bot should take.

        Return the list of probabilities for the next actions.
        """
        if self.model is None or self.predict_func is None:
            return [0.0] * domain.num_actions

        # create model data from message and convert it into a batch of 1
        data_X = self.featurizer.create_X([tracker], domain)
        model_data = self._create_model_data(data_X)
        predict_dataset = model_data.as_tf_dataset(1)
        batch_in = next(iter(predict_dataset))

        confidence = self.predict_func(batch_in)

        return confidence[0, -1, :].tolist()

    def persist(self, path: Text):
        """Persists the policy to a storage."""

        if self.model is None:
            return

        file_name = "embedding_policy"
        tf_model_file = os.path.join(path, f"{file_name}.tf_model")

        rasa.utils.io.create_directory_for_file(tf_model_file)

        self.featurizer.persist(path)

        self.model.save_weights(tf_model_file, save_format="tf")

        with open(os.path.join(path, file_name + ".tf_config.pkl"), "wb") as f:
            pickle.dump(self._tf_config, f)

        self.config["priority"] = self.priority

        with open(os.path.join(path, file_name + ".meta.pkl"), "wb") as f:
            pickle.dump(self.config, f)

        with open(os.path.join(path, file_name + ".data_example.pkl"), "wb") as f:
            pickle.dump(self.data_example, f)

        with open(
            os.path.join(path, file_name + ".encoded_all_label_ids.pkl"), "wb"
        ) as f:
            pickle.dump(self._encoded_all_label_ids, f)

        return {"file": file_name}

    @classmethod
    def load(cls, path: Text) -> "EmbeddingPolicy":
        """Loads a policy from the storage.

        **Needs to load its featurizer**
        """

        if not os.path.exists(path):
            raise Exception(
                "Failed to load dialogue model. Path '{}' "
                "doesn't exist".format(os.path.abspath(path))
            )

        file_name = "embedding_policy"
        tf_model_file = os.path.join(path, f"{file_name}.tf_model")

        featurizer = TrackerFeaturizer.load(path)

        if not os.path.exists(tf_model_file + ".meta"):
            return cls(featurizer=featurizer)

        with open(os.path.join(path, file_name + ".data_example.pkl"), "rb") as f:
            model_data_example = RasaModelData(
                label_key="actions_ids", data=pickle.load(f)
            )

        with open(
            os.path.join(path, file_name + ".encoded_all_label_ids.pkl"), "rb"
        ) as f:
            encoded_all_label_ids = pickle.load(f)

        with open(os.path.join(path, file_name + ".meta.pkl"), "rb") as f:
            meta = pickle.load(f)

        if meta[SIMILARITY_TYPE] == "auto":
            if meta[LOSS_TYPE] == "softmax":
                meta[SIMILARITY_TYPE] = "inner"
            elif meta[LOSS_TYPE] == "margin":
                meta[SIMILARITY_TYPE] = "cosine"

        model = TED(
            meta,
            isinstance(featurizer, MaxHistoryTrackerFeaturizer),
            encoded_all_label_ids,
        )

        logger.debug("Loading the model ...")
        model.fit(
            model_data_example,
            1,
            1,
            0,
            0,
            batch_strategy=meta[BATCH_STRATEGY],
            silent=True,  # don't confuse users with training output
            eager=True,  # no need to build tf graph, eager is faster here
        )
        model.load_weights(tf_model_file)

        # build the graph for prediction
        model.set_training_phase(False)
        model_data = RasaModelData(
            label_key="actions_ids",
            data={k: vs for k, vs in model_data_example.items() if "text" in k},
        )
        model.data_signature = model_data.get_signature()
        model.build_for_predict(model_data)
        predict_dataset = model_data.as_tf_dataset(
            1, batch_strategy="sequence", shuffle=False
        )
        predict_func = tf.function(
            func=model.predict, input_signature=[predict_dataset.element_spec]
        )
        batch_in = next(iter(predict_dataset))
        predict_func(batch_in)
        logger.debug("Finished loading the model.")

        return cls(
            component_config=meta,
            priority=meta["priority"],
            model=model,
            predict_func=predict_func,
        )


class TED(tf_models.RasaModel):
    def __init__(
        self,
        config: Dict[Text, Any],
        max_history_tracker_featurizer_used: bool,
        encoded_all_label_ids: np.ndarray,
    ):
        super().__init__()

        self.config = config
        self.max_history_tracker_featurizer_used = max_history_tracker_featurizer_used

        self._encoded_all_label_ids = encoded_all_label_ids

        self._optimizer = tf.keras.optimizers.Adam()

        # tf tensors
        self.training = tf.ones((), tf.bool)

        # persist
        self.all_bot_embed = None

        self.metric_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_acc = tf.keras.metrics.Mean(name="acc")
        self.metrics_to_log = ["loss", "acc"]

        self._loss_label = tf_layers.DotProductLoss(
            self.config[NUM_NEG],
            self.config[LOSS_TYPE],
            self.config[MU_POS],
            self.config[MU_NEG],
            self.config[USE_MAX_SIM_NEG],
            self.config[C_EMB],
            self.config[SCALE_LOSS],
        )
        self._ffnn_pre_dial = tf_layers.ReluFfn(
            self.config[HIDDEN_LAYERS_SIZES_PRE_DIAL],
            self.config[DROPRATE_DIAL],
            self.config[C2],
            layer_name_suffix="pre_dial",
        )
        self._ffnn_bot = tf_layers.ReluFfn(
            self.config[HIDDEN_LAYERS_SIZES_BOT],
            self.config[DROPRATE_BOT],
            self.config[C2],
            layer_name_suffix="bot",
        )
        self._transformer = tf_layers.TransformerEncoder(
            self.config[NUM_TRANSFORMER_LAYERS],
            self.config[TRANSFORMER_SIZE],
            self.config[NUM_HEADS],
            self.config[TRANSFORMER_SIZE] * 4,
            self.config[MAX_SEQ_LENGTH],
            self.config[C2],
            self.config[DROPRATE_DIAL],
            name="dial_encoder",
        )
        self._embed_dial = tf_layers.Embed(
            self.config[EMBED_DIM],
            self.config[C2],
            "dial",
            self.config[SIMILARITY_TYPE],
        )
        self._embed_bot = tf_layers.Embed(
            self.config[EMBED_DIM], self.config[C2], "bot", self.config[SIMILARITY_TYPE]
        )

    def set_training_phase(self, training: bool) -> None:
        if training:
            self.training = tf.ones((), tf.bool)
        else:
            self.training = tf.zeros((), tf.bool)

    def _create_tf_dial(self, a_in: tf.Tensor):
        """Create dialogue level embedding and mask."""

        # mask different length sequences
        # if there is at least one `-1` it should be masked
        mask = tf.sign(tf.reduce_max(a_in, -1) + 1)

        a = self._ffnn_pre_dial(a_in, self.training)
        a = self._transformer(a, tf.expand_dims(mask, axis=-1), self.training)

        if self.max_history_tracker_featurizer_used:
            # pick last label if max history featurizer is used
            a = a[:, -1:, :]
            mask = mask[:, -1:]

        dial_embed = self._embed_dial(a)

        return dial_embed, mask

    def _create_tf_bot_embed(self, b_in: tf.Tensor):
        b = self._ffnn_bot(b_in, self.training)
        return self._embed_bot(b)

    def batch_loss(
        self, batch_in: Union[Tuple[np.ndarray], Tuple[tf.Tensor]]
    ) -> tf.Tensor:
        a_in, b_in, _ = batch_in

        if self.max_history_tracker_featurizer_used:
            # add time dimension if max history featurizer is used
            b_in = b_in[:, tf.newaxis, :]

        all_bot_raw = tf.constant(
            self._encoded_all_label_ids, dtype=tf.float32, name="all_bot_raw"
        )

        dial_embed, mask = self._create_tf_dial(a_in)

        bot_embed = self._create_tf_bot_embed(b_in)
        self.all_bot_embed = self._create_tf_bot_embed(all_bot_raw)

        loss, acc = self._loss_label(
            dial_embed, bot_embed, b_in, self.all_bot_embed, all_bot_raw, mask
        )

        self.metric_loss.update_state(loss)
        self.metric_acc.update_state(acc)

        return loss

    def build_for_predict(self) -> None:
        all_bot_raw = tf.constant(
            self._encoded_all_label_ids, dtype=tf.float32, name="all_bot_raw"
        )
        self.all_bot_embed = self._create_tf_bot_embed(all_bot_raw)

    def predict(
        self, batch_in: Union[Tuple[np.ndarray], Tuple[tf.Tensor]], **kwargs
    ) -> tf.Tensor:
        a_in, b_in, _ = batch_in

        dial_embed, mask = self._create_tf_dial(a_in)

        sim_all = self._loss_label.sim(
            dial_embed[:, :, tf.newaxis, :],
            self.all_bot_embed[tf.newaxis, tf.newaxis, :, :],
            mask,
        )

        return train_utils.confidence_from_sim(sim_all, self.config[SIMILARITY_TYPE])
