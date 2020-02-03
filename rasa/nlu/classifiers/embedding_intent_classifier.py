import logging
from typing import Any, Dict, Optional, Text

from rasa.constants import DOCS_BASE_URL
from rasa.nlu.components import any_of
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.constants import (
    TOKENS_NAMES,
    TEXT_ATTRIBUTE,
    ENTITIES_ATTRIBUTE,
    DENSE_FEATURE_NAMES,
    SPARSE_FEATURE_NAMES,
)
from rasa.utils.tensorflow.constants import (
    HIDDEN_LAYERS_SIZES_TEXT,
    HIDDEN_LAYERS_SIZES_LABEL,
    SHARE_HIDDEN_LAYERS,
    TRANSFORMER_SIZE,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    MAX_SEQ_LENGTH,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
    LEARNING_RATE,
    DENSE_DIM,
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
    UNIDIRECTIONAL_ENCODER,
    DROPRATE,
    C_EMB,
    C2,
    SCALE_LOSS,
    USE_MAX_SIM_NEG,
    MU_NEG,
    MU_POS,
    EMBED_DIM,
    BILOU_FLAG,
)
from rasa.utils.common import raise_warning

logger = logging.getLogger(__name__)


class EmbeddingIntentClassifier(DIETClassifier):

    provides = [ENTITIES_ATTRIBUTE]

    requires = [
        TOKENS_NAMES[TEXT_ATTRIBUTE],
        any_of(
            DENSE_FEATURE_NAMES[TEXT_ATTRIBUTE], SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE]
        ),
    ]

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer for input words
        # the number of hidden layers is thus equal to the length of this list
        HIDDEN_LAYERS_SIZES_TEXT: [],
        # sizes of hidden layers before the embedding layer for intent labels
        # the number of hidden layers is thus equal to the length of this list
        HIDDEN_LAYERS_SIZES_LABEL: [],
        # Whether to share the hidden layer weights between input words and labels
        SHARE_HIDDEN_LAYERS: False,
        # number of units in transformer
        TRANSFORMER_SIZE: 256,
        # number of transformer layers
        NUM_TRANSFORMER_LAYERS: 2,
        # number of attention heads in transformer
        NUM_HEADS: 4,
        # max sequence length if pos_encoding='emb'
        MAX_SEQ_LENGTH: 256,
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
        DENSE_DIM: {"text": 512, "label": 20},
        # dimension size of embedding vectors
        EMBED_DIM: 20,
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
        MU_POS: 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect labels
        MU_NEG: -0.4,  # should be -1.0 < ... < 1.0 for 'cosine'
        # flag: if true, only minimize the maximum similarity for incorrect labels
        USE_MAX_SIM_NEG: True,
        # scale loss inverse proportionally to confidence of correct prediction
        SCALE_LOSS: True,
        # regularization parameters
        # the scale of L2 regularization
        C2: 0.002,
        # the scale of how critical the algorithm should be of minimizing the
        # maximum similarity between embeddings of different labels
        C_EMB: 0.8,
        # dropout rate for rnn
        DROPRATE: 0.2,
        # use a unidirectional or bidirectional encoder
        UNIDIRECTIONAL_ENCODER: True,
        # if true apply dropout to sparse tensors
        SPARSE_INPUT_DROPOUT: True,
        # visualization of accuracy
        # how often to calculate training accuracy
        EVAL_NUM_EPOCHS: 20,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        EVAL_NUM_EXAMPLES: 0,  # large values may hurt performance
    }
    # end default properties (DOC MARKER - don't remove)

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:

        component_config[INTENT_CLASSIFICATION] = True
        component_config[ENTITY_RECOGNITION] = False
        component_config[MASKED_LM] = False
        component_config[BILOU_FLAG] = False

        if "hidden_layers_sizes_a" in component_config:
            component_config[HIDDEN_LAYERS_SIZES_TEXT] = component_config[
                "hidden_layers_sizes_a"
            ]
        if "hidden_layers_sizes_b" in component_config:
            component_config[HIDDEN_LAYERS_SIZES_LABEL] = component_config[
                "hidden_layers_sizes_b"
            ]

        super().__init__(component_config)

        raise_warning(
            f"'EmbeddingIntentClassifier' is deprecated. Use 'DIETClassifier' instead .",
            category=DeprecationWarning,
            docs=f"{DOCS_BASE_URL}/nlu/components/",
        )
