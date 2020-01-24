import logging
import typing
from typing import Any, Dict, Text

from rasa.nlu.components import any_of
from rasa.nlu.classifiers.embedding_intent_classifier import EmbeddingIntentClassifier
from rasa.nlu.constants import (
    RESPONSE_ATTRIBUTE,
    RESPONSE_SELECTOR_PROPERTY_NAME,
    DEFAULT_OPEN_UTTERANCE_TYPE,
    DENSE_FEATURE_NAMES,
    TEXT_ATTRIBUTE,
    SPARSE_FEATURE_NAMES,
)

logger = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    from rasa.nlu.training_data import Message

import tensorflow as tf

# avoid warning println on contrib import - remove for tf 2
tf.contrib._warning = None


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

    provides = [RESPONSE_ATTRIBUTE, "response_ranking"]

    requires = [
        any_of(
            DENSE_FEATURE_NAMES[TEXT_ATTRIBUTE], SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE]
        ),
        any_of(
            DENSE_FEATURE_NAMES[RESPONSE_ATTRIBUTE],
            SPARSE_FEATURE_NAMES[RESPONSE_ATTRIBUTE],
        ),
    ]

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer for input words
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_a": [256, 128],
        # sizes of hidden layers before the embedding layer for intent labels
        # the number of hidden layers is thus equal to the length of this list
        "hidden_layers_sizes_b": [256, 128],
        # Whether to share the hidden layer weights between input words and intent labels
        "share_hidden_layers": False,
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
        # default dense dimension used if no dense features are present
        "dense_dim": {"text": 512, "label": 20},
        # dimension size of embedding vectors
        "embed_dim": 20,
        # the type of the similarity
        "num_neg": 20,
        # flag if minimize only maximum similarity over incorrect actions
        "similarity_type": "auto",  # string 'auto' or 'cosine' or 'inner'
        # the type of the loss function
        "loss_type": "softmax",  # string 'softmax' or 'margin'
        # number of top responses to normalize scores for softmax loss_type
        # set to 0 to turn off normalization
        "ranking_length": 10,
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
        # visualization of accuracy
        # how often to calculate training accuracy
        "evaluate_every_num_epochs": 20,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        "evaluate_on_num_examples": 0,  # large values may hurt performance,
        # selector config
        # name of the intent for which this response selector is to be trained
        "retrieval_intent": None,
    }
    # end default properties (DOC MARKER - don't remove)

    def _load_selector_params(self, config: Dict[Text, Any]):
        self.retrieval_intent = config["retrieval_intent"]
        if not self.retrieval_intent:
            # retrieval intent was left to its default value
            logger.info(
                "Retrieval intent parameter was left to its default value. This response selector will be trained"
                "on training examples combining all retrieval intents."
            )

    def _load_params(self) -> None:
        super()._load_params()
        self._load_selector_params(self.component_config)

    @staticmethod
    def _set_message_property(
        message: "Message", prediction_dict: Dict[Text, Any], selector_key: Text
    ):

        message_selector_properties = message.get(RESPONSE_SELECTOR_PROPERTY_NAME, {})
        message_selector_properties[selector_key] = prediction_dict
        message.set(
            RESPONSE_SELECTOR_PROPERTY_NAME,
            message_selector_properties,
            add_to_output=True,
        )

    def preprocess_train_data(self, training_data):
        """Performs sanity checks on training data, extracts encodings for labels
        and prepares data for training"""
        if self.retrieval_intent:
            training_data = training_data.filter_by_intent(self.retrieval_intent)

        label_id_dict = self._create_label_id_dict(
            training_data, attribute=RESPONSE_ATTRIBUTE
        )

        self.inverted_label_dict = {v: k for k, v in label_id_dict.items()}
        self._label_data = self._create_label_data(
            training_data, label_id_dict, attribute=RESPONSE_ATTRIBUTE
        )

        session_data = self._create_session_data(
            training_data.intent_examples,
            label_id_dict,
            label_attribute=RESPONSE_ATTRIBUTE,
        )

        self.check_input_dimension_consistency(session_data)

        return session_data

    def process(self, message: "Message", **kwargs: Any) -> None:
        """Return the most likely response and its similarity to the input."""

        label, label_ranking = self.predict_label(message)

        selector_key = (
            self.retrieval_intent
            if self.retrieval_intent
            else DEFAULT_OPEN_UTTERANCE_TYPE
        )

        logger.debug(
            f"Adding following selector key to message property: {selector_key}"
        )

        prediction_dict = {"response": label, "ranking": label_ranking}

        self._set_message_property(message, prediction_dict, selector_key)
