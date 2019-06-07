import io
import logging
import numpy as np
import os
import pickle
import typing
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Text, Tuple

from rasa.nlu.classifiers import INTENT_RANKING_LENGTH
from rasa.nlu.components import Component
from rasa.utils.common import is_logging_disabled

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    import tensorflow as tf
    from rasa.nlu.config import RasaNLUModelConfig
    from rasa.nlu.training_data import TrainingData
    from rasa.nlu.model import Metadata
    from rasa.nlu.training_data import Message

try:
    import tensorflow as tf

    # avoid warning println on contrib import - remove for tf 2
    tf.contrib._warning = None
except ImportError:
    tf = None


class EmbeddingIntentClassifier(Component):
    """Intent classifier using supervised embeddings.

    The embedding intent classifier embeds user inputs
    and intent labels into the same space.
    Supervised embeddings are trained by maximizing similarity between them.
    It also provides rankings of the labels that did not "win".

    The embedding intent classifier needs to be preceded by
    a featurizer in the pipeline.
    This featurizer creates the features used for the embeddings.
    It is recommended to use ``CountVectorsFeaturizer`` that
    can be optionally preceded by ``SpacyNLP`` and ``SpacyTokenizer``.

    Based on the starspace idea from: https://arxiv.org/abs/1709.03856.
    However, in this implementation the `mu` parameter is treated differently
    and additional hidden layers are added together with dropout.
    """

    provides = ["intent", "intent_ranking"]

    requires = ["text_features"]

    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer for input words
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_a": [256, 128],
        # sizes of hidden layers before the embedding layer for intent labels
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_b": [],
        # training parameters
        # initial and final batch sizes - batch size will be
        # linearly increased for each epoch
        "batch_size": [64, 256],
        # number of epochs
        "epochs": 300,
        # embedding parameters
        # dimension size of embedding vectors
        "embed_dim": 20,
        # how similar the algorithm should try
        # to make embedding vectors for correct intent labels
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect intent labels
        "mu_neg": -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
        # the type of the similarity
        "similarity_type": "cosine",  # string 'cosine' or 'inner'
        # the number of incorrect intents, the algorithm will minimize
        # their similarity to the input words during training
        "num_neg": 20,
        # flag: if true, only minimize the maximum similarity for
        # incorrect intent labels
        "use_max_sim_neg": True,
        # set random seed to any int to get reproducible results
        # try to change to another int if you are not getting good results
        "random_seed": None,
        # regularization parameters
        # the scale of L2 regularization
        "C2": 0.002,
        # the scale of how critical the algorithm should be of minimizing the
        # maximum similarity between embeddings of different intent labels
        "C_emb": 0.8,
        # dropout rate for rnn
        "droprate": 0.2,
        # flag: if true, the algorithm will split the intent labels into tokens
        #       and use bag-of-words representations for them
        "intent_tokenization_flag": False,
        # delimiter string to split the intent labels
        "intent_split_symbol": "_",
        # visualization of accuracy
        # how often to calculate training accuracy
        "evaluate_every_num_epochs": 10,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        "evaluate_on_num_examples": 1000,  # large values may hurt performance
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        inv_intent_dict: Optional[Dict[int, Text]] = None,
        encoded_all_intents: Optional[np.ndarray] = None,
        session: Optional["tf.Session"] = None,
        graph: Optional["tf.Graph"] = None,
        message_placeholder: Optional["tf.Tensor"] = None,
        intent_placeholder: Optional["tf.Tensor"] = None,
        similarity_op: Optional["tf.Tensor"] = None,
        word_embed: Optional["tf.Tensor"] = None,
        intent_embed: Optional["tf.Tensor"] = None,
    ) -> None:
        """Declare instant variables with default values"""

        self._check_tensorflow()
        super(EmbeddingIntentClassifier, self).__init__(component_config)

        self._load_params()

        # transform numbers to intents
        self.inv_intent_dict = inv_intent_dict
        # encode all intents with numbers
        self.encoded_all_intents = encoded_all_intents

        # tf related instances
        self.session = session
        self.graph = graph
        self.a_in = message_placeholder
        self.b_in = intent_placeholder
        self.sim_op = similarity_op

        # persisted embeddings
        self.word_embed = word_embed
        self.intent_embed = intent_embed

    # init helpers
    def _load_nn_architecture_params(self, config: Dict[Text, Any]) -> None:
        self.hidden_layer_sizes = {
            "a": config["hidden_layers_sizes_a"],
            "b": config["hidden_layers_sizes_b"],
        }

        self.batch_size = config["batch_size"]
        self.epochs = config["epochs"]

    def _load_embedding_params(self, config: Dict[Text, Any]) -> None:
        self.embed_dim = config["embed_dim"]
        self.mu_pos = config["mu_pos"]
        self.mu_neg = config["mu_neg"]
        self.similarity_type = config["similarity_type"]
        self.num_neg = config["num_neg"]
        self.use_max_sim_neg = config["use_max_sim_neg"]
        self.random_seed = self.component_config["random_seed"]

    def _load_regularization_params(self, config: Dict[Text, Any]) -> None:
        self.C2 = config["C2"]
        self.C_emb = config["C_emb"]
        self.droprate = config["droprate"]

    def _load_flag_if_tokenize_intents(self, config: Dict[Text, Any]) -> None:
        self.intent_tokenization_flag = config["intent_tokenization_flag"]
        self.intent_split_symbol = config["intent_split_symbol"]
        if self.intent_tokenization_flag and not self.intent_split_symbol:
            logger.warning(
                "intent_split_symbol was not specified, "
                "so intent tokenization will be ignored"
            )
            self.intent_tokenization_flag = False

    def _load_visual_params(self, config: Dict[Text, Any]) -> None:
        self.evaluate_every_num_epochs = config["evaluate_every_num_epochs"]
        if self.evaluate_every_num_epochs < 1:
            self.evaluate_every_num_epochs = self.epochs

        self.evaluate_on_num_examples = config["evaluate_on_num_examples"]

    def _load_params(self) -> None:

        self._load_nn_architecture_params(self.component_config)
        self._load_embedding_params(self.component_config)
        self._load_regularization_params(self.component_config)
        self._load_flag_if_tokenize_intents(self.component_config)
        self._load_visual_params(self.component_config)

    # package safety checks
    @classmethod
    def required_packages(cls) -> List[Text]:
        return ["tensorflow"]

    @staticmethod
    def _check_tensorflow():
        if tf is None:
            raise ImportError(
                "Failed to import `tensorflow`. "
                "Please install `tensorflow`. "
                "For example with `pip install tensorflow`."
            )

    # training data helpers:
    @staticmethod
    def _create_intent_dict(training_data: "TrainingData") -> Dict[Text, int]:
        """Create intent dictionary"""

        distinct_intents = set(
            [example.get("intent") for example in training_data.intent_examples]
        )
        return {intent: idx for idx, intent in enumerate(sorted(distinct_intents))}

    @staticmethod
    def _create_intent_token_dict(
        intents: List[Text], intent_split_symbol: Text
    ) -> Dict[Text, int]:
        """Create intent token dictionary"""

        distinct_tokens = set(
            [token for intent in intents for token in intent.split(intent_split_symbol)]
        )
        return {token: idx for idx, token in enumerate(sorted(distinct_tokens))}

    def _create_encoded_intents(self, intent_dict: Dict[Text, int]) -> np.ndarray:
        """Create matrix with intents encoded in rows as bag of words.

        If intent_tokenization_flag is off, returns identity matrix.
        """

        if self.intent_tokenization_flag:
            intent_token_dict = self._create_intent_token_dict(
                list(intent_dict.keys()), self.intent_split_symbol
            )

            encoded_all_intents = np.zeros((len(intent_dict), len(intent_token_dict)))
            for key, idx in intent_dict.items():
                for t in key.split(self.intent_split_symbol):
                    encoded_all_intents[idx, intent_token_dict[t]] = 1

            return encoded_all_intents
        else:
            return np.eye(len(intent_dict))

    # noinspection PyPep8Naming
    def _create_all_Y(self, size: int) -> np.ndarray:
        """Stack encoded_all_intents on top of each other

        to create candidates for training examples and
        to calculate training accuracy
        """

        return np.stack([self.encoded_all_intents] * size)

    # noinspection PyPep8Naming
    def _prepare_data_for_training(
        self, training_data: "TrainingData", intent_dict: Dict[Text, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare data for training"""

        X = np.stack([e.get("text_features") for e in training_data.intent_examples])

        intents_for_X = np.array(
            [intent_dict[e.get("intent")] for e in training_data.intent_examples]
        )

        Y = np.stack(
            [self.encoded_all_intents[intent_idx] for intent_idx in intents_for_X]
        )

        return X, Y, intents_for_X

    # tf helpers:
    def _create_tf_embed_nn(
        self,
        x_in: "tf.Tensor",
        is_training: "tf.Tensor",
        layer_sizes: List[int],
        name: Text,
    ) -> "tf.Tensor":
        """Create nn with hidden layers and name"""

        reg = tf.contrib.layers.l2_regularizer(self.C2)
        x = x_in
        for i, layer_size in enumerate(layer_sizes):
            x = tf.layers.dense(
                inputs=x,
                units=layer_size,
                activation=tf.nn.relu,
                kernel_regularizer=reg,
                name="hidden_layer_{}_{}".format(name, i),
            )
            x = tf.layers.dropout(x, rate=self.droprate, training=is_training)

        x = tf.layers.dense(
            inputs=x,
            units=self.embed_dim,
            kernel_regularizer=reg,
            name="embed_layer_{}".format(name),
        )
        return x

    def _create_tf_embed(
        self, a_in: "tf.Tensor", b_in: "tf.Tensor", is_training: "tf.Tensor"
    ) -> Tuple["tf.Tensor", "tf.Tensor"]:
        """Create tf graph for training"""

        emb_a = self._create_tf_embed_nn(
            a_in, is_training, self.hidden_layer_sizes["a"], name="a"
        )
        emb_b = self._create_tf_embed_nn(
            b_in, is_training, self.hidden_layer_sizes["b"], name="b"
        )
        return emb_a, emb_b

    def _tf_sim(
        self, a: "tf.Tensor", b: "tf.Tensor"
    ) -> Tuple["tf.Tensor", "tf.Tensor"]:
        """Define similarity

        in two cases:
            sim: between embedded words and embedded intent labels
            sim_emb: between individual embedded intent labels only
        """

        if self.similarity_type == "cosine":
            # normalize embedding vectors for cosine similarity
            a = tf.nn.l2_normalize(a, -1)
            b = tf.nn.l2_normalize(b, -1)

        if self.similarity_type in {"cosine", "inner"}:
            sim = tf.reduce_sum(tf.expand_dims(a, 1) * b, -1)
            sim_emb = tf.reduce_sum(b[:, 0:1, :] * b[:, 1:, :], -1)

            return sim, sim_emb

        else:
            raise ValueError(
                "Wrong similarity type {}, "
                "should be 'cosine' or 'inner'"
                "".format(self.similarity_type)
            )

    def _tf_loss(self, sim: "tf.Tensor", sim_emb: "tf.Tensor") -> "tf.Tensor":
        """Define loss"""

        # loss for maximizing similarity with correct action
        loss = tf.maximum(0.0, self.mu_pos - sim[:, 0])

        if self.use_max_sim_neg:
            # minimize only maximum similarity over incorrect actions
            max_sim_neg = tf.reduce_max(sim[:, 1:], -1)
            loss += tf.maximum(0.0, self.mu_neg + max_sim_neg)
        else:
            # minimize all similarities with incorrect actions
            max_margin = tf.maximum(0.0, self.mu_neg + sim[:, 1:])
            loss += tf.reduce_sum(max_margin, -1)

        # penalize max similarity between intent embeddings
        max_sim_emb = tf.maximum(0.0, tf.reduce_max(sim_emb, -1))
        loss += max_sim_emb * self.C_emb

        # average the loss over the batch and add regularization losses
        loss = tf.reduce_mean(loss) + tf.losses.get_regularization_loss()
        return loss

    # training helpers:
    def _create_batch_b(
        self, batch_pos_b: np.ndarray, intent_ids: np.ndarray
    ) -> np.ndarray:
        """Create batch of intents.

        Where the first is correct intent
        and the rest are wrong intents sampled randomly
        """

        batch_pos_b = batch_pos_b[:, np.newaxis, :]

        # sample negatives
        batch_neg_b = np.zeros(
            (batch_pos_b.shape[0], self.num_neg, batch_pos_b.shape[-1])
        )
        for b in range(batch_pos_b.shape[0]):
            # create negative indexes out of possible ones
            # except for correct index of b
            negative_indexes = [
                i
                for i in range(self.encoded_all_intents.shape[0])
                if i != intent_ids[b]
            ]
            negs = np.random.choice(negative_indexes, size=self.num_neg)

            batch_neg_b[b] = self.encoded_all_intents[negs]

        return np.concatenate([batch_pos_b, batch_neg_b], 1)

    def _linearly_increasing_batch_size(self, epoch: int) -> int:
        """Linearly increase batch size with every epoch.

        The idea comes from https://arxiv.org/abs/1711.00489
        """

        if not isinstance(self.batch_size, list):
            return int(self.batch_size)

        if self.epochs > 1:
            return int(
                self.batch_size[0]
                + epoch * (self.batch_size[1] - self.batch_size[0]) / (self.epochs - 1)
            )
        else:
            return int(self.batch_size[0])

    # noinspection PyPep8Naming
    def _train_tf(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        intents_for_X: np.ndarray,
        loss: "tf.Tensor",
        is_training: "tf.Tensor",
        train_op: "tf.Tensor",
    ) -> None:
        """Train tf graph"""

        self.session.run(tf.global_variables_initializer())

        if self.evaluate_on_num_examples:
            logger.info(
                "Accuracy is updated every {} epochs"
                "".format(self.evaluate_every_num_epochs)
            )

        pbar = tqdm(range(self.epochs), desc="Epochs", disable=is_logging_disabled())
        train_acc = 0
        last_loss = 0
        for ep in pbar:
            indices = np.random.permutation(len(X))

            batch_size = self._linearly_increasing_batch_size(ep)
            batches_per_epoch = len(X) // batch_size + int(len(X) % batch_size > 0)

            ep_loss = 0
            for i in range(batches_per_epoch):
                end_idx = (i + 1) * batch_size
                start_idx = i * batch_size
                batch_a = X[indices[start_idx:end_idx]]
                batch_pos_b = Y[indices[start_idx:end_idx]]
                intents_for_b = intents_for_X[indices[start_idx:end_idx]]
                # add negatives
                batch_b = self._create_batch_b(batch_pos_b, intents_for_b)

                sess_out = self.session.run(
                    {"loss": loss, "train_op": train_op},
                    feed_dict={
                        self.a_in: batch_a,
                        self.b_in: batch_b,
                        is_training: True,
                    },
                )
                ep_loss += sess_out.get("loss") / batches_per_epoch

            if self.evaluate_on_num_examples:
                if (
                    ep == 0
                    or (ep + 1) % self.evaluate_every_num_epochs == 0
                    or (ep + 1) == self.epochs
                ):
                    train_acc = self._output_training_stat(
                        X, intents_for_X, is_training
                    )
                    last_loss = ep_loss

                pbar.set_postfix(
                    {
                        "loss": "{:.3f}".format(ep_loss),
                        "acc": "{:.3f}".format(train_acc),
                    }
                )
            else:
                pbar.set_postfix({"loss": "{:.3f}".format(ep_loss)})

        if self.evaluate_on_num_examples:
            logger.info(
                "Finished training embedding classifier, "
                "loss={:.3f}, train accuracy={:.3f}"
                "".format(last_loss, train_acc)
            )

    # noinspection PyPep8Naming
    def _output_training_stat(
        self, X: np.ndarray, intents_for_X: np.ndarray, is_training: "tf.Tensor"
    ) -> np.ndarray:
        """Output training statistics"""

        n = self.evaluate_on_num_examples
        ids = np.random.permutation(len(X))[:n]
        all_Y = self._create_all_Y(X[ids].shape[0])

        train_sim = self.session.run(
            self.sim_op,
            feed_dict={self.a_in: X[ids], self.b_in: all_Y, is_training: False},
        )

        train_acc = np.mean(np.argmax(train_sim, -1) == intents_for_X[ids])
        return train_acc

    def train(
        self,
        training_data: "TrainingData",
        cfg: Optional["RasaNLUModelConfig"] = None,
        **kwargs: Any
    ) -> None:
        """Train the embedding intent classifier on a data set."""

        intent_dict = self._create_intent_dict(training_data)
        if len(intent_dict) < 2:
            logger.error(
                "Can not train an intent classifier. "
                "Need at least 2 different classes. "
                "Skipping training of intent classifier."
            )
            return

        self.inv_intent_dict = {v: k for k, v in intent_dict.items()}
        self.encoded_all_intents = self._create_encoded_intents(intent_dict)

        # noinspection PyPep8Naming
        X, Y, intents_for_X = self._prepare_data_for_training(
            training_data, intent_dict
        )

        # check if number of negatives is less than number of intents
        logger.debug(
            "Check if num_neg {} is smaller than "
            "number of intents {}, "
            "else set num_neg to the number of intents - 1"
            "".format(self.num_neg, self.encoded_all_intents.shape[0])
        )
        self.num_neg = min(self.num_neg, self.encoded_all_intents.shape[0] - 1)

        self.graph = tf.Graph()
        with self.graph.as_default():
            # set random seed
            np.random.seed(self.random_seed)
            tf.set_random_seed(self.random_seed)

            self.a_in = tf.placeholder(tf.float32, (None, X.shape[-1]), name="a")
            self.b_in = tf.placeholder(tf.float32, (None, None, Y.shape[-1]), name="b")

            is_training = tf.placeholder_with_default(False, shape=())

            (self.word_embed, self.intent_embed) = self._create_tf_embed(
                self.a_in, self.b_in, is_training
            )

            self.sim_op, sim_emb = self._tf_sim(self.word_embed, self.intent_embed)
            loss = self._tf_loss(self.sim_op, sim_emb)

            train_op = tf.train.AdamOptimizer().minimize(loss)

            # train tensorflow graph
            self.session = tf.Session()

            self._train_tf(X, Y, intents_for_X, loss, is_training, train_op)

    # process helpers
    # noinspection PyPep8Naming
    def _calculate_message_sim(
        self, X: np.ndarray, all_Y: np.ndarray
    ) -> Tuple[np.ndarray, List[float]]:
        """Load tf graph and calculate message similarities"""

        message_sim = self.session.run(
            self.sim_op, feed_dict={self.a_in: X, self.b_in: all_Y}
        )
        message_sim = message_sim.flatten()  # sim is a matrix

        intent_ids = message_sim.argsort()[::-1]
        message_sim[::-1].sort()

        if self.similarity_type == "cosine":
            # clip negative values to zero
            message_sim[message_sim < 0] = 0
        elif self.similarity_type == "inner":
            # normalize result to [0, 1] with softmax
            message_sim = np.exp(message_sim)
            message_sim /= np.sum(message_sim)

        # transform sim to python list for JSON serializing
        return intent_ids, message_sim.tolist()

    def process(self, message: "Message", **kwargs: Any) -> None:
        """Return the most likely intent and its similarity to the input."""

        intent = {"name": None, "confidence": 0.0}
        intent_ranking = []

        if self.session is None:
            logger.error(
                "There is no trained tf.session: "
                "component is either not trained or "
                "didn't receive enough training data"
            )

        else:
            # get features (bag of words) for a message
            # noinspection PyPep8Naming
            X = message.get("text_features").reshape(1, -1)

            # stack encoded_all_intents on top of each other
            # to create candidates for test examples
            # noinspection PyPep8Naming
            all_Y = self._create_all_Y(X.shape[0])

            # load tf graph and session
            intent_ids, message_sim = self._calculate_message_sim(X, all_Y)

            # if X contains all zeros do not predict some label
            if X.any() and intent_ids.size > 0:
                intent = {
                    "name": self.inv_intent_dict[intent_ids[0]],
                    "confidence": message_sim[0],
                }

                ranking = list(zip(list(intent_ids), message_sim))
                ranking = ranking[:INTENT_RANKING_LENGTH]
                intent_ranking = [
                    {"name": self.inv_intent_dict[intent_idx], "confidence": score}
                    for intent_idx, score in ranking
                ]

        message.set("intent", intent, add_to_output=True)
        message.set("intent_ranking", intent_ranking, add_to_output=True)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:
        """Persist this model into the passed directory.

        Return the metadata necessary to load the model again.
        """

        if self.session is None:
            return {"file": None}

        checkpoint = os.path.join(model_dir, file_name + ".ckpt")

        try:
            os.makedirs(os.path.dirname(checkpoint))
        except OSError as e:
            # be happy if someone already created the path
            import errno

            if e.errno != errno.EEXIST:
                raise
        with self.graph.as_default():
            self.graph.clear_collection("message_placeholder")
            self.graph.add_to_collection("message_placeholder", self.a_in)

            self.graph.clear_collection("intent_placeholder")
            self.graph.add_to_collection("intent_placeholder", self.b_in)

            self.graph.clear_collection("similarity_op")
            self.graph.add_to_collection("similarity_op", self.sim_op)

            self.graph.clear_collection("word_embed")
            self.graph.add_to_collection("word_embed", self.word_embed)
            self.graph.clear_collection("intent_embed")
            self.graph.add_to_collection("intent_embed", self.intent_embed)

            saver = tf.train.Saver()
            saver.save(self.session, checkpoint)

        with io.open(
            os.path.join(model_dir, file_name + "_inv_intent_dict.pkl"), "wb"
        ) as f:
            pickle.dump(self.inv_intent_dict, f)
        with io.open(
            os.path.join(model_dir, file_name + "_encoded_all_intents.pkl"), "wb"
        ) as f:
            pickle.dump(self.encoded_all_intents, f)

        return {"file": file_name}

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: "Metadata" = None,
        cached_component: Optional["EmbeddingIntentClassifier"] = None,
        **kwargs: Any
    ) -> "EmbeddingIntentClassifier":

        if model_dir and meta.get("file"):
            file_name = meta.get("file")
            checkpoint = os.path.join(model_dir, file_name + ".ckpt")
            graph = tf.Graph()
            with graph.as_default():
                sess = tf.Session()
                saver = tf.train.import_meta_graph(checkpoint + ".meta")

                saver.restore(sess, checkpoint)

                a_in = tf.get_collection("message_placeholder")[0]
                b_in = tf.get_collection("intent_placeholder")[0]

                sim_op = tf.get_collection("similarity_op")[0]

                word_embed = tf.get_collection("word_embed")[0]
                intent_embed = tf.get_collection("intent_embed")[0]

            with io.open(
                os.path.join(model_dir, file_name + "_inv_intent_dict.pkl"), "rb"
            ) as f:
                inv_intent_dict = pickle.load(f)
            with io.open(
                os.path.join(model_dir, file_name + "_encoded_all_intents.pkl"), "rb"
            ) as f:
                encoded_all_intents = pickle.load(f)

            return cls(
                component_config=meta,
                inv_intent_dict=inv_intent_dict,
                encoded_all_intents=encoded_all_intents,
                session=sess,
                graph=graph,
                message_placeholder=a_in,
                intent_placeholder=b_in,
                similarity_op=sim_op,
                word_embed=word_embed,
                intent_embed=intent_embed,
            )

        else:
            logger.warning(
                "Failed to load nlu model. Maybe path {} "
                "doesn't exist"
                "".format(os.path.abspath(model_dir))
            )
            return cls(component_config=meta)
