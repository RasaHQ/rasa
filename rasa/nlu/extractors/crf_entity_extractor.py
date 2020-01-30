import logging
from typing import Any, Dict, Optional, Text

from rasa.constants import DOCS_BASE_URL
from rasa.nlu.components import any_of
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.constants import (
    TEXT_ATTRIBUTE,
    ENTITIES_ATTRIBUTE,
    DENSE_FEATURE_NAMES,
    SPARSE_FEATURE_NAMES,
)
from rasa.utils.tensorflow.constants import (
    HIDDEN_LAYERS_SIZES_TEXT,
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
    SPARSE_INPUT_DROPOUT,
    MASKED_LM,
    ENTITY_RECOGNITION,
    INTENT_CLASSIFICATION,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    UNIDIRECTIONAL_ENCODER,
    DROPRATE,
    C2,
    BILOU_FLAG,
)
from rasa.utils.common import raise_warning

logger = logging.getLogger(__name__)


class CRFEntityExtractor(DIETClassifier):

    provides = [ENTITIES_ATTRIBUTE]

    requires = [
        any_of(
            DENSE_FEATURE_NAMES[TEXT_ATTRIBUTE], SPARSE_FEATURE_NAMES[TEXT_ATTRIBUTE]
        )
    ]

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # nn architecture
        # sizes of hidden layers before the embedding layer for input words
        # the number of hidden layers is thus equal to the length of this list
        HIDDEN_LAYERS_SIZES_TEXT: [256, 128],
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
        # regularization parameters
        # the scale of L2 regularization
        C2: 0.002,
        # dropout rate for rnn
        DROPRATE: 0.2,
        # if true apply dropout to sparse tensors
        SPARSE_INPUT_DROPOUT: True,
        # visualization of accuracy
        # how often to calculate training accuracy
        EVAL_NUM_EPOCHS: 20,  # small values may hurt performance
        # how many examples to use for calculation of training accuracy
        EVAL_NUM_EXAMPLES: 0,  # large values may hurt performance,
        # BILOU_flag determines whether to use BILOU tagging or not.
        # More rigorous however requires more examples per entity
        # rule of thumb: use only if more than 100 egs. per entity
        BILOU_FLAG: False,
    }
    # end default properties (DOC MARKER - don't remove)

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None) -> None:

        component_config[INTENT_CLASSIFICATION] = False
        component_config[ENTITY_RECOGNITION] = True
        component_config[MASKED_LM] = False
        component_config[TRANSFORMER_SIZE] = 128
        component_config[NUM_TRANSFORMER_LAYERS] = 0
        component_config[NUM_HEADS] = 4
        component_config[SHARE_HIDDEN_LAYERS] = False
        component_config[MAX_SEQ_LENGTH] = 256
        component_config[UNIDIRECTIONAL_ENCODER] = True

        super().__init__(component_config)

        raise_warning(
            f"'CRFEntityExtractor' is deprecated. Use 'DIETClassifier' in"
            f"combination with the 'LexicalSyntacticFeaturizer'. Check "
            f"Check '{DOCS_BASE_URL}/nlu/components/' for more details.",
            DeprecationWarning,
        )
