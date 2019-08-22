import io
import logging
import numpy as np
import os
import pickle
import typing
from typing import Any, Dict, Optional, Text

from rasa.nlu.classifiers import LABEL_RANKING_LENGTH
from rasa.utils import train_utils

from rasa.nlu.classifiers.embedding_intent_classifier import EmbeddingIntentClassifier
from rasa.constants import (
    DEFAULT_OPEN_UTTERANCE_TYPE_KEY,
    DEFAULT_OPEN_UTTERANCE_TYPE_KEY_RANKING,
)
from rasa.core.actions.action import RESPOND_PREFIX
from rasa.nlu.constants import MESSAGE_TEXT_ATTRIBUTE, MESSAGE_RESPONSE_ATTRIBUTE

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from tensorflow import Graph, Session, Tensor
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


class ResponseSelector(EmbeddingIntentClassifier):
    """Response selector using supervised embeddings.

        The response selector embeds user inputs
        and candidate response into the same space.
        Supervised embeddings are trained by maximizing similarity between them.
        It also provides rankings of the response that did not "win".

        The supervised response selector needs to be preceded by
        a featurizer in the pipeline.
        This featurizer creates the features used for the embeddings.
        It is recommended to use ``CountVectorsFeaturizer`` that
        can be optionally preceded by ``SpacyNLP`` and ``SpacyTokenizer``.

        Based on the starspace idea from: https://arxiv.org/abs/1709.03856.
        However, in this implementation the `mu` parameter is treated differently
        and additional hidden layers are added together with dropout.
        """

    provides = ["response", "response_ranking"]

    requires = ["text_features"]

    name = "ResponseSelector"

    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer for input words
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_a": [256, 128],
        # sizes of hidden layers before the embedding layer for intent labels
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_b": [],
        # Whether to share the embedding between input words and intent labels
        "share_embedding": False,
        # training parameters
        # initial and final batch sizes - batch size will be
        # linearly increased for each epoch
        "batch_size": [64, 256],
        # how to create batches
        "batch_strategy": "balanced",  # string 'sequence' or 'balanced'
        # number of epochs
        "epochs": 300,
        # set random seed to any int to get reproducible results
        "random_seed": None,
        # embedding parameters
        # dimension size of embedding vectors
        "embed_dim": 20,
        # the type of the similarity
        "num_neg": 20,
        # flag if minimize only maximum similarity over incorrect actions
        "similarity_type": "auto",  # string 'auto' or 'cosine' or 'inner'
        # the type of the loss function
        "loss_type": "softmax",  # string 'softmax' or 'margin'
        # how similar the algorithm should try
        # to make embedding vectors for correct intent labels
        "mu_pos": 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect intent labels
        "mu_neg": -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
        # flag: if true, only minimize the maximum similarity for
        # incorrect intent labels
        "use_max_sim_neg": True,
        # scale loss inverse proportionally to confidence of correct prediction
        "scale_loss": True,
        # regularization parameters
        # the scale of L2 regularization
        "C2": 0.002,
        # the scale of how critical the algorithm should be of minimizing the
        # maximum similarity between embeddings of different intent labels
        "C_emb": 0.8,
        # dropout rate for rnn
        "droprate": 0.2,
        "label_tokenization_flag": False,
        # delimiter string to split the labels
        "label_split_symbol": "_",
        # visualization of accuracy
        # how often to calculate training accuracy
        "evaluate_every_num_epochs": 20,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        "evaluate_on_num_examples": 0,  # large values may hurt performance,
        # selector config
        "response_type": None,
    }
    # end default properties (DOC MARKER - don't remove)

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        inv_label_dict: Optional[Dict[int, Text]] = None,
        session: Optional["tf.Session"] = None,
        graph: Optional["tf.Graph"] = None,
        message_placeholder: Optional["tf.Tensor"] = None,
        label_placeholder: Optional["tf.Tensor"] = None,
        similarity_all: Optional["tf.Tensor"] = None,
        pred_confidence: Optional["tf.Tensor"] = None,
        similarity: Optional["tf.Tensor"] = None,
        message_embed: Optional["tf.Tensor"] = None,
        label_embed: Optional["tf.Tensor"] = None,
        all_labels_embed: Optional["tf.Tensor"] = None,
    ) -> None:
        super(ResponseSelector, self).__init__(
            component_config,
            inv_label_dict,
            session,
            graph,
            message_placeholder,
            label_placeholder,
            similarity_all,
            pred_confidence,
            similarity,
            message_embed,
            label_embed,
            all_labels_embed,
        )

    def _load_selector_params(self, config: Dict[Text, Any]):
        self.response_type = config["response_type"]

    def _load_params(self) -> None:
        super(ResponseSelector, self)._load_params()
        self._load_selector_params(self.component_config)

    def process(self, message: "Message", **kwargs: Any) -> None:
        """Return the most likely intent and its similarity to the input."""

        label = {"name": None, "confidence": 0.0}
        label_ranking = []

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

            # load tf graph and session
            label_ids, message_sim = self._calculate_message_sim(X)

            # if X contains all zeros do not predict some label
            if X.any() and label_ids.size > 0:
                label = {
                    "name": self.inverted_label_dict[label_ids[0]],
                    "confidence": message_sim[0],
                }

                ranking = list(zip(list(label_ids), message_sim))
                ranking = ranking[:LABEL_RANKING_LENGTH]
                label_ranking = [
                    {"name": self.inverted_label_dict[label_idx], "confidence": score}
                    for label_idx, score in ranking
                ]

        if self.response_type:
            response_key_for_tracker = "{0}{1}_response".format(
                RESPOND_PREFIX, self.response_type
            )
            response_ranking_key_for_tracker = "{0}{1}_response_ranking".format(
                RESPOND_PREFIX, self.response_type
            )
        else:
            response_key_for_tracker = DEFAULT_OPEN_UTTERANCE_TYPE_KEY
            response_ranking_key_for_tracker = DEFAULT_OPEN_UTTERANCE_TYPE_KEY_RANKING

        message.set(response_key_for_tracker, label, add_to_output=True)
        message.set(response_ranking_key_for_tracker, label_ranking, add_to_output=True)

    def preprocess_data(self, training_data):
        """Performs sanity checks on training data, extracts encodings for labels and prepares data for training"""

        if self.response_type:
            training_data = training_data.filter_by_intent(self.response_type)

        label_id_dict = self._create_label_id_dict(
            training_data, attribute=MESSAGE_RESPONSE_ATTRIBUTE
        )

        self.inverted_label_dict = {v: k for k, v in label_id_dict.items()}
        self._encoded_all_label_ids = self._create_encoded_label_ids(
            training_data,
            label_id_dict,
            attribute=MESSAGE_RESPONSE_ATTRIBUTE,
            attribute_feature_name="response_features",
        )

        # check if number of negatives is less than number of label_ids
        logger.debug(
            "Check if num_neg {} is smaller than "
            "number of label_ids {}, "
            "else set num_neg to the number of label_ids - 1"
            "".format(self.num_neg, self._encoded_all_label_ids.shape[0])
        )
        # noinspection PyAttributeOutsideInit
        self.num_neg = min(self.num_neg, self._encoded_all_label_ids.shape[0] - 1)

        session_data = self._create_session_data(
            training_data, label_id_dict, attribute=MESSAGE_RESPONSE_ATTRIBUTE
        )

        self.check_input_dimension_consistency(session_data)

        return session_data, label_id_dict
