import logging
from typing import Any, Dict, Optional, Text

from rasa.core.constants import DEFAULT_POLICY_PRIORITY, DIALOGUE
from rasa.core.featurizers import TrackerFeaturizer
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.constants import DOCS_URL_POLICIES
from rasa.utils.tensorflow.constants import (
    LABEL,
    HIDDEN_LAYERS_SIZES,
    TRANSFORMER_SIZE,
    NUM_TRANSFORMER_LAYERS,
    NUM_HEADS,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
    RANKING_LENGTH,
    LOSS_TYPE,
    SIMILARITY_TYPE,
    NUM_NEG,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    NEGATIVE_MARGIN_SCALE,
    REGULARIZATION_CONSTANT,
    SCALE_LOSS,
    USE_MAX_NEG_SIM,
    MAX_NEG_SIM,
    MAX_POS_SIM,
    EMBEDDING_DIMENSION,
    DROP_RATE_DIALOGUE,
    DROP_RATE_LABEL,
    DROP_RATE_ATTENTION,
    WEIGHT_SPARSITY,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
)
from rasa.utils.common import raise_warning
from rasa.utils.tensorflow.models import RasaModel

logger = logging.getLogger(__name__)


class EmbeddingPolicy(TEDPolicy):
    """Transformer Embedding Dialogue (TED) Policy.

    The policy used in our paper https://arxiv.org/abs/1910.00486.
    """

    defaults = {
        # nn architecture
        # a list of hidden layers sizes before dialogue and action embedding layers
        # number of hidden layers is equal to the length of this list
        HIDDEN_LAYERS_SIZES: {DIALOGUE: [], LABEL: []},
        # number of units in transformer
        TRANSFORMER_SIZE: 128,
        # number of transformer layers
        NUM_TRANSFORMER_LAYERS: 1,
        # number of attention heads in transformer
        NUM_HEADS: 4,
        # training parameters
        # initial and final batch sizes:
        # batch size will be linearly increased for each epoch
        BATCH_SIZES: [8, 32],
        # how to create batches
        BATCH_STRATEGY: "balanced",  # 'sequence' or 'balanced'
        # number of epochs
        EPOCHS: 1,
        # set random seed to any int to get reproducible results
        RANDOM_SEED: None,
        # embedding parameters
        # dimension size of embedding vectors
        EMBEDDING_DIMENSION: 20,
        # the type of the similarity
        NUM_NEG: 20,
        # flag if minimize only maximum similarity over incorrect labels
        SIMILARITY_TYPE: "auto",  # 'auto' or 'cosine' or 'inner'
        # the type of the loss function
        LOSS_TYPE: "softmax",  # 'softmax' or 'margin'
        # number of top actions to normalize scores for softmax loss_type
        # set to 0 to turn off normalization
        RANKING_LENGTH: 10,
        # how similar the algorithm should try
        # to make embedding vectors for correct labels
        MAX_POS_SIM: 0.8,  # should be 0.0 < ... < 1.0 for 'cosine'
        # maximum negative similarity for incorrect labels
        MAX_NEG_SIM: -0.2,  # should be -1.0 < ... < 1.0 for 'cosine'
        # the number of incorrect labels, the algorithm will minimize
        # their similarity to the user input during training
        USE_MAX_NEG_SIM: True,  # flag which loss function to use
        # scale loss inverse proportionally to confidence of correct prediction
        SCALE_LOSS: True,
        # regularization
        # the scale of regularization
        REGULARIZATION_CONSTANT: 0.001,
        # the scale of how important is to minimize the maximum similarity
        # between embeddings of different labels
        NEGATIVE_MARGIN_SCALE: 0.8,
        # dropout rate for dial nn
        DROP_RATE_DIALOGUE: 0.1,
        # dropout rate for bot nn
        DROP_RATE_LABEL: 0.0,
        # dropout rate for attention
        DROP_RATE_ATTENTION: 0,
        # sparsity of the weights in dense layers
        WEIGHT_SPARSITY: 0.8,
        # visualization of accuracy
        # how often calculate validation accuracy
        EVAL_NUM_EPOCHS: 20,  # small values may hurt performance
        # how many examples to use for hold out validation set
        EVAL_NUM_EXAMPLES: 0,  # large values may hurt performance
        # if true use key relative embeddings in attention
        KEY_RELATIVE_ATTENTION: False,
        # if true use key relative embeddings in attention
        VALUE_RELATIVE_ATTENTION: False,
        # max position for relative embeddings
        MAX_RELATIVE_POSITION: None,
    }

    def __init__(
        self,
        featurizer: Optional[TrackerFeaturizer] = None,
        priority: int = DEFAULT_POLICY_PRIORITY,
        max_history: Optional[int] = None,
        model: Optional[RasaModel] = None,
        **kwargs: Dict[Text, Any],
    ) -> None:

        super().__init__(featurizer, priority, max_history, model, **kwargs)

        raise_warning(
            f"'EmbeddingPolicy' is deprecated and will be removed in version 2.0. "
            f"Use 'TEDPolicy' instead.",
            category=FutureWarning,
            docs="https://rasa.com/docs/rasa/migration-guide/",
        )
