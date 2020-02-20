import logging

from typing import Any, Dict, Optional, Text

from rasa.nlu.training_data import TrainingData, Message
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.components import any_of
from rasa.utils.tensorflow.constants import (
    LABEL,
    HIDDEN_LAYERS_SIZES,
    SHARE_HIDDEN_LAYERS,
    NUM_TRANSFORMER_LAYERS,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
    LEARNING_RATE,
    DENSE_DIMENSION,
    RANKING_LENGTH,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    SPARSE_INPUT_DROPOUT,
    MASKED_LM,
    ENTITY_RECOGNITION,
    INTENT_CLASSIFICATION,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    DROPRATE,
    NEG_MARGIN_SCALE,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    BILOU_FLAG,
)
from rasa.nlu.constants import (
    RESPONSE,
    RESPONSE_SELECTOR_PROPERTY_NAME,
    DEFAULT_OPEN_UTTERANCE_TYPE,
    DENSE_FEATURE_NAMES,
    TEXT,
    SPARSE_FEATURE_NAMES,
)
from rasa.utils.tensorflow.model_data import RasaModelData
from rasa.utils.tensorflow.models import RasaModel
from rasa.utils.common import raise_warning
from rasa.constants import DOCS_URL_COMPONENTS

logger = logging.getLogger(__name__)


class ResponseSelector(DIETClassifier):
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

    provides = [RESPONSE, "response_ranking"]

    requires = [
        any_of(DENSE_FEATURE_NAMES[TEXT], SPARSE_FEATURE_NAMES[TEXT]),
        any_of(DENSE_FEATURE_NAMES[RESPONSE], SPARSE_FEATURE_NAMES[RESPONSE]),
    ]

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer
        # for input words and responses
        # the number of hidden layers is thus equal to the length of this list
        HIDDEN_LAYERS_SIZES: {TEXT: [256, 128], LABEL: [256, 128]},
        # Whether to share the hidden layer weights between input words and intent
        # labels
        SHARE_HIDDEN_LAYERS: False,
        # training parameters
        # initial and final batch sizes - batch size will be
        # linearly increased for each epoch
        BATCH_SIZES: [64, 256],
        # how to create batches
        BATCH_STRATEGY: "balanced",  # string 'sequence' or 'balanced'
        # number of epochs
        EPOCHS: 300,
        # set random seed to any int to get reproducible results
        RANDOM_SEED: None,
        # optimizer
        LEARNING_RATE: 0.001,
        # embedding parameters
        # default dense dimension used if no dense features are present
        DENSE_DIMENSION: {TEXT: 512, LABEL: 512},
        # dimension size of embedding vectors
        EMBEDDING_DIMENSION: 20,
        # the type of the similarity
        NUM_NEG: 20,
        # flag if minimize only maximum similarity over incorrect actions
        SIMILARITY_TYPE: "auto",  # string 'auto' or 'cosine' or 'inner'
        # the type of the loss function
        LOSS_TYPE: "softmax",  # string 'softmax' or 'margin'
        # number of top responses to normalize scores for softmax loss_type
        # set to 0 to turn off normalization
        RANKING_LENGTH: 10,
        # how similar the algorithm should try
        # to make embedding vectors for correct intent labels
        MAX_POS_SIM: 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect intent labels
        MAX_NEG_SIM: -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
        # flag: if true, only minimize the maximum similarity for
        # incorrect intent labels
        USE_MAX_NEG_SIM: True,
        # scale loss inverse proportionally to confidence of correct prediction
        SCALE_LOSS: True,
        # regularization parameters
        # the scale of L2 regularization
        REGULARIZATION_CONSTANT: 0.002,
        # the scale of how critical the algorithm should be of minimizing the
        # maximum similarity between embeddings of different intent labels
        NEG_MARGIN_SCALE: 0.8,
        # dropout rate for rnn
        DROPRATE: 0.2,
        # if true apply dropout to sparse tensors
        SPARSE_INPUT_DROPOUT: False,
        # visualization of accuracy
        # how often to calculate training accuracy
        EVAL_NUM_EPOCHS: 20,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        EVAL_NUM_EXAMPLES: 0,  # large values may hurt performance,
        # selector config
        # name of the intent for which this response selector is to be trained
        "retrieval_intent": None,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        inverted_label_dict: Optional[Dict[int, Text]] = None,
        inverted_tag_dict: Optional[Dict[int, Text]] = None,
        model: Optional[RasaModel] = None,
        batch_tuple_sizes: Optional[Dict] = None,
    ) -> None:

        component_config = component_config or {}

        # the following properties cannot be adapted for the ResponseSelector
        component_config[INTENT_CLASSIFICATION] = True
        component_config[ENTITY_RECOGNITION] = False
        component_config[BILOU_FLAG] = False
        component_config[MASKED_LM] = False
        component_config[NUM_TRANSFORMER_LAYERS] = 0

        super().__init__(
            component_config,
            inverted_label_dict,
            inverted_tag_dict,
            model,
            batch_tuple_sizes,
        )

        raise_warning(
            f"'ResponseSelector' is deprecated and will be removed in version 2.0. "
            f"Use 'DIETSelector' instead.",
            category=FutureWarning,
            docs=DOCS_URL_COMPONENTS,
        )

    @property
    def label_key(self) -> Text:
        return "label_ids"

    def _load_selector_params(self, config: Dict[Text, Any]) -> None:
        self.retrieval_intent = config["retrieval_intent"]
        if not self.retrieval_intent:
            # retrieval intent was left to its default value
            logger.info(
                "Retrieval intent parameter was left to its default value. This "
                "response selector will be trained on training examples combining "
                "all retrieval intents."
            )

    def _check_config_parameters(self) -> None:
        super()._check_config_parameters()
        self._load_selector_params(self.component_config)

    @staticmethod
    def _set_message_property(
        message: Message, prediction_dict: Dict[Text, Any], selector_key: Text
    ) -> None:

        message_selector_properties = message.get(RESPONSE_SELECTOR_PROPERTY_NAME, {})
        message_selector_properties[selector_key] = prediction_dict
        message.set(
            RESPONSE_SELECTOR_PROPERTY_NAME,
            message_selector_properties,
            add_to_output=True,
        )

    def preprocess_train_data(self, training_data: TrainingData) -> RasaModelData:
        """Performs sanity checks on training data, extracts encodings for labels
        and prepares data for training"""
        if self.retrieval_intent:
            training_data = training_data.filter_by_intent(self.retrieval_intent)

        label_id_dict = self._create_label_id_dict(training_data, attribute=RESPONSE)
        self.inverted_label_dict = {v: k for k, v in label_id_dict.items()}

        self._label_data = self._create_label_data(
            training_data, label_id_dict, attribute=RESPONSE
        )

        model_data = self._create_model_data(
            training_data.intent_examples, label_id_dict, label_attribute=RESPONSE
        )

        self.check_input_dimension_consistency(model_data)

        return model_data

    def process(self, message: Message, **kwargs: Any) -> None:
        """Return the most likely response and its similarity to the input."""

        out = self._predict(message)
        label, label_ranking = self._predict_label(out)

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
