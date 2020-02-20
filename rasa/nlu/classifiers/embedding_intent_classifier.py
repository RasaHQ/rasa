import logging
from typing import Any, Dict, Optional, Text

from rasa.constants import DOCS_URL_COMPONENTS
from rasa.nlu.components import any_of
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.constants import TEXT, DENSE_FEATURE_NAMES, SPARSE_FEATURE_NAMES
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
    WEIGHT_SPARSITY,
    NEGATIVE_MARGIN_SCALE,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    BILOU_FLAG,
)
from rasa.utils.common import raise_warning
from rasa.utils.tensorflow.models import RasaModel

logger = logging.getLogger(__name__)


class EmbeddingIntentClassifier(DIETClassifier):

    provides = ["intent", "intent_ranking"]

    requires = [any_of(DENSE_FEATURE_NAMES[TEXT], SPARSE_FEATURE_NAMES[TEXT])]

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer
        # for input words and intent labels,
        # the number of hidden layers is thus equal to the length of this list
        HIDDEN_LAYERS_SIZES: {TEXT: [256, 128], LABEL: []},
        # Whether to share the hidden layer weights between input words and labels
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
        DENSE_DIMENSION: {TEXT: 512, LABEL: 20},
        # dimension size of embedding vectors
        EMBEDDING_DIMENSION: 20,
        # the type of the similarity
        NUM_NEG: 20,
        # flag if minimize only maximum similarity over incorrect actions
        SIMILARITY_TYPE: "auto",  # string 'auto' or 'cosine' or 'inner'
        # the type of the loss function
        LOSS_TYPE: "softmax",  # string 'softmax' or 'margin'
        # number of top intents to normalize scores for softmax loss_type
        # set to 0 to turn off normalization
        RANKING_LENGTH: 10,
        # how similar the algorithm should try
        # to make embedding vectors for correct labels
        MAX_POS_SIM: 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect labels
        MAX_NEG_SIM: -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
        # flag: if true, only minimize the maximum similarity for incorrect labels
        USE_MAX_NEG_SIM: True,
        # scale loss inverse proportionally to confidence of correct prediction
        SCALE_LOSS: True,
        # regularization parameters
        # the scale of regularization
        REGULARIZATION_CONSTANT: 0.002,
        # the scale of how critical the algorithm should be of minimizing the
        # maximum similarity between embeddings of different labels
        NEGATIVE_MARGIN_SCALE: 0.8,
        # dropout rate for rnn
        DROP_RATE: 0.2,
        # sparsity of the weights in dense layers
        WEIGHT_SPARSITY: 0.8,
        # if true apply dropout to sparse tensors
        SPARSE_INPUT_DROPOUT: False,
        # visualization of accuracy
        # how often to calculate training accuracy
        EVAL_NUM_EPOCHS: 20,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        EVAL_NUM_EXAMPLES: 0,  # large values may hurt performance
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

        # the following properties cannot be adapted for the EmbeddingIntentClassifier
        component_config[INTENT_CLASSIFICATION] = True
        component_config[ENTITY_RECOGNITION] = False
        component_config[MASKED_LM] = False
        component_config[BILOU_FLAG] = False
        component_config[NUM_TRANSFORMER_LAYERS] = 0

        super().__init__(
            component_config,
            inverted_label_dict,
            inverted_tag_dict,
            model,
            batch_tuple_sizes,
        )

        raise_warning(
            "'EmbeddingIntentClassifier' is deprecated and will be removed in version "
            "2.0. Use 'DIETClassifier' instead.",
            category=FutureWarning,
            docs="https://rasa.com/docs/rasa/migration-guide/",
        )
