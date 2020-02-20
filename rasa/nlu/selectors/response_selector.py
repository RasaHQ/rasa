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
    DROP_RATE,
    NEGATIVE_MARGIN_SCALE,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    BILOU_FLAG,
    RETRIEVAL_INTENT,
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
import rasa.utils.common as common_utils
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
        # ## Architecture of the used neural network
        # Hidden layer sizes for layers before the embedding layers for user message
        # and labels.
        # The number of hidden layers is equal to the length of the corresponding
        # list.
        HIDDEN_LAYERS_SIZES: {TEXT: [256, 128], LABEL: [256, 128]},
        # Whether to share the hidden layer weights between user message and labels.
        SHARE_HIDDEN_LAYERS: False,
        # ## Training parameters
        # Initial and final batch sizes:
        # Batch size will be linearly increased for each epoch.
        BATCH_SIZES: [64, 256],
        # Strategy used when creating batches.
        # Can be either 'sequence' or 'balanced'.
        BATCH_STRATEGY: "balanced",
        # Number of epochs to train
        EPOCHS: 300,
        # Set random seed to any 'int' to get reproducible results
        RANDOM_SEED: None,
        # Initial learning rate for the optimizer
        LEARNING_RATE: 0.001,
        # ## Parameters for embeddings
        # Dimension size of embedding vectors
        EMBEDDING_DIMENSION: 20,
        # Default dense dimension to use if no dense features are present.
        DENSE_DIMENSION: {TEXT: 512, LABEL: 512},
        # The number of incorrect labels. The algorithm will minimize
        # their similarity to the user input during training.
        NUM_NEG: 20,
        # Type of similarity measure to use, either 'auto' or 'cosine' or 'inner'.
        SIMILARITY_TYPE: "auto",
        # The type of the loss function, either 'softmax' or 'margin'.
        LOSS_TYPE: "softmax",
        # Number of top actions to normalize scores for loss type 'softmax'.
        # Set to 0 to turn off normalization.
        RANKING_LENGTH: 10,
        # Indicates how similar the algorithm should try to make embedding vectors
        # for correct labels.
        # Should be 0.0 < ... < 1.0 for 'cosine' similarity type.
        MAX_POS_SIM: 0.8,
        # Maximum negative similarity for incorrect labels.
        # Should be -1.0 < ... < 1.0 for 'cosine' similarity type.
        MAX_NEG_SIM: -0.4,
        # If 'True' the algorithm only minimizes maximum similarity over
        # incorrect intent labels, used only if 'loss_type' is set to 'margin'.
        USE_MAX_NEG_SIM: True,
        # Scale loss inverse proportionally to confidence of correct prediction
        SCALE_LOSS: True,
        # ## Regularization parameters
        # The scale of regularization
        REGULARIZATION_CONSTANT: 0.002,
        # The scale of how important is to minimize the maximum similarity
        # between embeddings of different labels.
        NEGATIVE_MARGIN_SCALE: 0.8,
        # Dropout rate for encoder
        DROP_RATE: 0.2,
        # If 'True' apply dropout to sparse tensors
        SPARSE_INPUT_DROPOUT: False,
        # ## Evaluation parameters
        # How often calculate validation accuracy.
        # Small values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EPOCHS: 20,
        # How many examples to use for hold out validation set
        # Large values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EXAMPLES: 0,
        # ## Selector config
        # Name of the intent for which this response selector is to be trained
        RETRIEVAL_INTENT: None,
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

        common_utils.raise_warning(
            f"'ResponseSelector' is deprecated and will be removed in version 2.0. "
            f"Use 'DIETSelector' instead.",
            category=FutureWarning,
            docs=DOCS_URL_COMPONENTS,
        )

    @property
    def label_key(self) -> Text:
        return "label_ids"

    def _load_selector_params(self, config: Dict[Text, Any]) -> None:
        self.retrieval_intent = config[RETRIEVAL_INTENT]
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
