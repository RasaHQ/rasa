from collections import namedtuple
import copy
import json
import logging
import os
import pickle
import warnings

import numpy as np
from typing import Any, List, Optional, Text, Dict, Tuple

import rasa.utils.io
from rasa.core import utils
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
tf.contrib._warning = None
logger = logging.getLogger(__name__)


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
        "hidden_layers_sizes_pre_dial": [],
        # a list of hidden layers sizes before bot embed layer
        # number of hidden layers is equal to the length of this list
        "hidden_layers_sizes_bot": [],
        # number of units in transformer
        "transformer_size": 128,
        # number of transformer layers
        "num_transformer_layers": 1,
        # type of positional encoding in transformer
        "pos_encoding": "timing",  # string 'timing' or 'emb'
        # max sequence length if pos_encoding='emb'
        "max_seq_length": 256,
        # number of attention heads in transformer
        "num_heads": 4,
        # training parameters
        # initial and final batch sizes:
        # batch size will be linearly increased for each epoch
        "batch_size": [8, 32],
        # how to create batches
        "batch_strategy": "balanced",  # string 'sequence' or 'balanced'
        # number of epochs
        "epochs": 1,
        # set random seed to any int to get reproducible results
        "random_seed": None,
        # embedding parameters
        # dimension size of embedding vectors
        "embed_dim": 20,
        # the type of the similarity
        "num_neg": 20,
        # flag if minimize only maximum similarity over incorrect labels
        "similarity_type": "auto",  # string 'auto' or 'cosine' or 'inner'
        # the type of the loss function
        "loss_type": "softmax",  # string 'softmax' or 'margin'
        # how similar the algorithm should try
        # to make embedding vectors for correct labels
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect labels
        "mu_neg": -0.2,  # should be -1.0 < ... < 1.0 for 'cosine'
        # the number of incorrect labels, the algorithm will minimize
        # their similarity to the user input during training
        "use_max_sim_neg": True,  # flag which loss function to use
        # scale loss inverse proportionally to confidence of correct prediction
        "scale_loss": True,
        # regularization
        # the scale of L2 regularization
        "C2": 0.001,
        # the scale of how important is to minimize the maximum similarity
        # between embeddings of different labels
        "C_emb": 0.8,
        # dropout rate for dial nn
        "droprate_a": 0.1,
        # dropout rate for bot nn
        "droprate_b": 0.0,
        # visualization of accuracy
        # how often calculate validation accuracy
        "evaluate_every_num_epochs": 20,  # small values may hurt performance
        # how many examples to use for hold out validation set
        "evaluate_on_num_examples": 0,  # large values may hurt performance
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
        graph: Optional["tf.Graph"] = None,
        session: Optional["tf.Session"] = None,
        user_placeholder: Optional["tf.Tensor"] = None,
        bot_placeholder: Optional["tf.Tensor"] = None,
        similarity_all: Optional["tf.Tensor"] = None,
        pred_confidence: Optional["tf.Tensor"] = None,
        similarity: Optional["tf.Tensor"] = None,
        dial_embed: Optional["tf.Tensor"] = None,
        bot_embed: Optional["tf.Tensor"] = None,
        all_bot_embed: Optional["tf.Tensor"] = None,
        attention_weights: Optional["tf.Tensor"] = None,
        max_history: Optional[int] = None,
        **kwargs: Any
    ) -> None:
        """Declare instant variables with default values"""

        if not featurizer:
            featurizer = self._standard_featurizer(max_history)
        super(EmbeddingPolicy, self).__init__(featurizer, priority)

        self._load_params(**kwargs)

        # encode all label_ids with numbers
        self._encoded_all_label_ids = None

        # tf related instances
        self.graph = graph
        self.session = session
        self.a_in = user_placeholder
        self.b_in = bot_placeholder
        self.sim_all = similarity_all
        self.pred_confidence = pred_confidence
        self.sim = similarity

        # persisted embeddings
        self.dial_embed = dial_embed
        self.bot_embed = bot_embed
        self.all_bot_embed = all_bot_embed

        self.attention_weights = attention_weights
        # internal tf instances
        self._iterator = None
        self._train_op = None
        self._is_training = None

    # init helpers
    def _load_nn_architecture_params(self, config: Dict[Text, Any]) -> None:
        self.hidden_layers_sizes = {
            "pre_dial": config["hidden_layers_sizes_pre_dial"],
            "bot": config["hidden_layers_sizes_bot"],
        }

        self.pos_encoding = config["pos_encoding"]
        self.max_seq_length = config["max_seq_length"]
        self.num_heads = config["num_heads"]

        self.transformer_size = config["transformer_size"]
        self.num_transformer_layers = config["num_transformer_layers"]

        self.batch_size = config["batch_size"]
        self.batch_strategy = config["batch_strategy"]

        self.epochs = config["epochs"]

        self.random_seed = config["random_seed"]

    def _load_embedding_params(self, config: Dict[Text, Any]) -> None:
        self.embed_dim = config["embed_dim"]
        self.num_neg = config["num_neg"]

        self.similarity_type = config["similarity_type"]
        self.loss_type = config["loss_type"]
        if self.similarity_type == "auto":
            if self.loss_type == "softmax":
                self.similarity_type = "inner"
            elif self.loss_type == "margin":
                self.similarity_type = "cosine"

        self.mu_pos = config["mu_pos"]
        self.mu_neg = config["mu_neg"]
        self.use_max_sim_neg = config["use_max_sim_neg"]

        self.scale_loss = config["scale_loss"]

    def _load_regularization_params(self, config: Dict[Text, Any]) -> None:
        self.C2 = config["C2"]
        self.C_emb = config["C_emb"]
        self.droprate = {"bot": config["droprate_b"], "dial": config["droprate_a"]}

    def _load_visual_params(self, config: Dict[Text, Any]) -> None:
        self.evaluate_every_num_epochs = config["evaluate_every_num_epochs"]
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs
        self.evaluate_on_num_examples = config["evaluate_on_num_examples"]

    def _load_params(self, **kwargs: Dict[Text, Any]) -> None:
        config = copy.deepcopy(self.defaults)
        config.update(kwargs)

        self._tf_config = train_utils.load_tf_config(config)
        self._load_nn_architecture_params(config)
        self._load_embedding_params(config)
        self._load_regularization_params(config)
        self._load_visual_params(config)

    # data helpers
    # noinspection PyPep8Naming
    @staticmethod
    def _label_ids_for_Y(data_Y: "np.ndarray") -> "np.ndarray":
        """Prepare Y data for training: extract label_ids."""

        return data_Y.argmax(axis=-1)

    # noinspection PyPep8Naming
    def _label_features_for_Y(self, label_ids: "np.ndarray") -> "np.ndarray":
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
    def _create_session_data(
        self, data_X: "np.ndarray", data_Y: Optional["np.ndarray"] = None
    ) -> "train_utils.SessionData":
        """Combine all tf session related data into a named tuple"""

        if data_Y is not None:
            # training time
            label_ids = self._label_ids_for_Y(data_Y)
            Y = self._label_features_for_Y(label_ids)

            # idea taken from sklearn's stratify split
            if label_ids.ndim == 2:
                # for multi-label y, map each distinct row to a string repr
                # using join because str(row) uses an ellipsis if len(row) > 1000
                label_ids = np.array([" ".join(row.astype("str")) for row in label_ids])
        else:
            # prediction time
            label_ids = None
            Y = None

        return train_utils.SessionData(X=data_X, Y=Y, label_ids=label_ids)

    def _create_tf_bot_embed(self, b_in: "tf.Tensor") -> "tf.Tensor":
        """Create embedding bot vector."""

        b = train_utils.create_tf_fnn(
            b_in,
            self.hidden_layers_sizes["bot"],
            self.droprate["bot"],
            self.C2,
            self._is_training,
            layer_name_suffix="bot",
        )
        return train_utils.create_tf_embed(
            b, self.embed_dim, self.C2, self.similarity_type, layer_name_suffix="bot"
        )

    def _create_tf_dial(self, a_in) -> Tuple["tf.Tensor", "tf.Tensor"]:
        """Create dialogue level embedding and mask."""

        # mask different length sequences
        # if there is at least one `-1` it should be masked
        mask = tf.sign(tf.reduce_max(self.a_in, -1) + 1)

        a = train_utils.create_tf_fnn(
            a_in,
            self.hidden_layers_sizes["pre_dial"],
            self.droprate["dial"],
            self.C2,
            self._is_training,
            layer_name_suffix="pre_dial",
        )

        self.attention_weights = {}
        hparams = train_utils.create_t2t_hparams(
            self.num_transformer_layers,
            self.transformer_size,
            self.num_heads,
            self.droprate["dial"],
            self.pos_encoding,
            self.max_seq_length,
            self._is_training,
        )

        a = train_utils.create_t2t_transformer_encoder(
            a, mask, self.attention_weights, hparams, self.C2, self._is_training
        )

        if isinstance(self.featurizer, MaxHistoryTrackerFeaturizer):
            # pick last label if max history featurizer is used
            a = a[:, -1:, :]
            mask = mask[:, -1:]

        dial_embed = train_utils.create_tf_embed(
            a, self.embed_dim, self.C2, self.similarity_type, layer_name_suffix="dial"
        )

        return dial_embed, mask

    def _build_tf_train_graph(self) -> Tuple["tf.Tensor", "tf.Tensor"]:
        """Bulid train graph using iterator."""

        # session data are int counts but we need a float tensors
        self.a_in, self.b_in = self._iterator.get_next()
        if isinstance(self.featurizer, MaxHistoryTrackerFeaturizer):
            # add time dimension if max history featurizer is used
            self.b_in = self.b_in[:, tf.newaxis, :]

        all_bot_raw = tf.constant(
            self._encoded_all_label_ids, dtype=tf.float32, name="all_bot_raw"
        )

        self.dial_embed, mask = self._create_tf_dial(self.a_in)

        self.bot_embed = self._create_tf_bot_embed(self.b_in)
        self.all_bot_embed = self._create_tf_bot_embed(all_bot_raw)

        return train_utils.calculate_loss_acc(
            self.dial_embed,
            self.bot_embed,
            self.b_in,
            self.all_bot_embed,
            all_bot_raw,
            self.num_neg,
            mask,
            self.loss_type,
            self.mu_pos,
            self.mu_neg,
            self.use_max_sim_neg,
            self.C_emb,
            self.scale_loss,
        )

    # prepare for prediction
    def _create_tf_placeholders(self, session_data: "train_utils.SessionData") -> None:
        """Create placeholders for prediction."""

        dialogue_len = None  # use dynamic time
        self.a_in = tf.placeholder(
            dtype=tf.float32,
            shape=(None, dialogue_len, session_data.X.shape[-1]),
            name="a",
        )
        self.b_in = tf.placeholder(
            dtype=tf.float32,
            shape=(None, dialogue_len, None, session_data.Y.shape[-1]),
            name="b",
        )

    def _build_tf_pred_graph(
        self, session_data: "train_utils.SessionData"
    ) -> "tf.Tensor":
        """Rebuild tf graph for prediction."""

        self._create_tf_placeholders(session_data)

        self.dial_embed, mask = self._create_tf_dial(self.a_in)

        self.sim_all = train_utils.tf_raw_sim(
            self.dial_embed[:, :, tf.newaxis, :],
            self.all_bot_embed[tf.newaxis, tf.newaxis, :, :],
            mask,
        )

        self.bot_embed = self._create_tf_bot_embed(self.b_in)

        self.sim = train_utils.tf_raw_sim(
            self.dial_embed[:, :, tf.newaxis, :], self.bot_embed, mask
        )

        return train_utils.confidence_from_sim(self.sim_all, self.similarity_type)

    # training methods
    def train(
        self,
        training_trackers: List["DialogueStateTracker"],
        domain: "Domain",
        **kwargs: Any
    ) -> None:
        """Train the policy on given training trackers."""

        logger.debug("Started training embedding policy.")

        # set numpy random seed
        np.random.seed(self.random_seed)

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
            "".format(self.num_neg, domain.num_actions)
        )
        # noinspection PyAttributeOutsideInit
        self.num_neg = min(self.num_neg, domain.num_actions - 1)

        # extract actual training data to feed to tf session
        session_data = self._create_session_data(training_data.X, training_data.y)

        if self.evaluate_on_num_examples:
            session_data, eval_session_data = train_utils.train_val_split(
                session_data, self.evaluate_on_num_examples, self.random_seed
            )
        else:
            eval_session_data = None

        self.graph = tf.Graph()
        with self.graph.as_default():
            # set random seed in tf
            tf.set_random_seed(self.random_seed)

            # allows increasing batch size
            batch_size_in = tf.placeholder(tf.int64)

            (
                self._iterator,
                train_init_op,
                eval_init_op,
            ) = train_utils.create_iterator_init_datasets(
                session_data, eval_session_data, batch_size_in, self.batch_strategy
            )

            self._is_training = tf.placeholder_with_default(False, shape=())

            loss, acc = self._build_tf_train_graph()

            # define which optimizer to use
            self._train_op = tf.train.AdamOptimizer().minimize(loss)

            # train tensorflow graph
            self.session = tf.Session(config=self._tf_config)
            train_utils.train_tf_dataset(
                train_init_op,
                eval_init_op,
                batch_size_in,
                loss,
                acc,
                self._train_op,
                self.session,
                self._is_training,
                self.epochs,
                self.batch_size,
                self.evaluate_on_num_examples,
                self.evaluate_every_num_epochs,
            )

            # rebuild the graph for prediction
            self.pred_confidence = self._build_tf_pred_graph(session_data)

            self.attention_weights = train_utils.extract_attention(
                self.attention_weights
            )

    def continue_training(
        self,
        training_trackers: List["DialogueStateTracker"],
        domain: "Domain",
        **kwargs: Any
    ) -> None:
        """Continue training an already trained policy."""

        batch_size = kwargs.get("batch_size", 5)
        epochs = kwargs.get("epochs", 50)

        with self.graph.as_default():
            for _ in range(epochs):
                training_data = self._training_data_for_continue_training(
                    batch_size, training_trackers, domain
                )

                session_data = self._create_session_data(
                    training_data.X, training_data.y
                )
                train_dataset = train_utils.create_tf_dataset(session_data, batch_size)
                train_init_op = self._iterator.make_initializer(train_dataset)
                self.session.run(train_init_op)

                # fit to one extra example using updated trackers
                while True:
                    try:
                        self.session.run(
                            self._train_op, feed_dict={self._is_training: True}
                        )

                    except tf.errors.OutOfRangeError:
                        break

    def tf_feed_dict_for_prediction(
        self, tracker: "DialogueStateTracker", domain: "Domain"
    ) -> Dict["tf.Tensor", "np.ndarray"]:
        """Create feed dictionary for tf session."""

        # noinspection PyPep8Naming
        data_X = self.featurizer.create_X([tracker], domain)
        session_data = self._create_session_data(data_X)

        return {self.a_in: session_data.X}

    def predict_action_probabilities(
        self, tracker: "DialogueStateTracker", domain: "Domain"
    ) -> List[float]:
        """Predict the next action the bot should take.

        Return the list of probabilities for the next actions.
        """

        if self.session is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data"
            )
            return [0.0] * domain.num_actions

        tf_feed_dict = self.tf_feed_dict_for_prediction(tracker, domain)

        confidence = self.session.run(self.pred_confidence, feed_dict=tf_feed_dict)

        return confidence[0, -1, :].tolist()

    def persist(self, path: Text) -> None:
        """Persists the policy to a storage."""

        if self.session is None:
            warnings.warn(
                "Method `persist(...)` was called "
                "without a trained model present. "
                "Nothing to persist then!"
            )
            return

        self.featurizer.persist(path)

        meta = {"priority": self.priority}

        meta_file = os.path.join(path, "embedding_policy.json")
        utils.dump_obj_as_json_to_file(meta_file, meta)

        file_name = "tensorflow_embedding.ckpt"
        checkpoint = os.path.join(path, file_name)
        rasa.utils.io.create_directory_for_file(checkpoint)

        with self.graph.as_default():
            train_utils.persist_tensor("user_placeholder", self.a_in, self.graph)
            train_utils.persist_tensor("bot_placeholder", self.b_in, self.graph)

            train_utils.persist_tensor("similarity_all", self.sim_all, self.graph)
            train_utils.persist_tensor(
                "pred_confidence", self.pred_confidence, self.graph
            )
            train_utils.persist_tensor("similarity", self.sim, self.graph)

            train_utils.persist_tensor("dial_embed", self.dial_embed, self.graph)
            train_utils.persist_tensor("bot_embed", self.bot_embed, self.graph)
            train_utils.persist_tensor("all_bot_embed", self.all_bot_embed, self.graph)

            train_utils.persist_tensor(
                "attention_weights", self.attention_weights, self.graph
            )

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with open(os.path.join(path, file_name + ".tf_config.pkl"), "wb") as f:
            pickle.dump(self._tf_config, f)

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

        featurizer = TrackerFeaturizer.load(path)

        file_name = "tensorflow_embedding.ckpt"
        checkpoint = os.path.join(path, file_name)

        if not os.path.exists(checkpoint + ".meta"):
            return cls(featurizer=featurizer)

        meta_file = os.path.join(path, "embedding_policy.json")
        meta = json.loads(rasa.utils.io.read_file(meta_file))

        with open(os.path.join(path, file_name + ".tf_config.pkl"), "rb") as f:
            _tf_config = pickle.load(f)

        graph = tf.Graph()
        with graph.as_default():
            session = tf.Session(config=_tf_config)
            saver = tf.train.import_meta_graph(checkpoint + ".meta")

            saver.restore(session, checkpoint)

            a_in = train_utils.load_tensor("user_placeholder")
            b_in = train_utils.load_tensor("bot_placeholder")

            sim_all = train_utils.load_tensor("similarity_all")
            pred_confidence = train_utils.load_tensor("pred_confidence")
            sim = train_utils.load_tensor("similarity")

            dial_embed = train_utils.load_tensor("dial_embed")
            bot_embed = train_utils.load_tensor("bot_embed")
            all_bot_embed = train_utils.load_tensor("all_bot_embed")

            attention_weights = train_utils.load_tensor("attention_weights")

        return cls(
            featurizer=featurizer,
            priority=meta["priority"],
            graph=graph,
            session=session,
            user_placeholder=a_in,
            bot_placeholder=b_in,
            similarity_all=sim_all,
            pred_confidence=pred_confidence,
            similarity=sim,
            dial_embed=dial_embed,
            bot_embed=bot_embed,
            all_bot_embed=all_bot_embed,
            attention_weights=attention_weights,
        )
