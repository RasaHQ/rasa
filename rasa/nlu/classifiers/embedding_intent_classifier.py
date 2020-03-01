import logging
from typing import Any, Dict, Optional, Text, List, Type

from rasa.constants import DOCS_URL_MIGRATION_GUIDE
from rasa.nlu.featurizers.featurizer import Featurizer
from rasa.nlu.components import Component
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.constants import TEXT
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
    SOFTMAX,
    AUTO,
    BALANCED,
)
import rasa.utils.common as common_utils
from rasa.utils.tensorflow.models import RasaModel

logger = logging.getLogger(__name__)


class EmbeddingIntentClassifier(DIETClassifier):
    """Dual Intent Entity Transformer used for intent classification.

    The ``EmbeddingIntentClassifier`` embeds user inputs and intent labels into the
    same space.
    Supervised embeddings are trained by maximizing similarity between them.
    This algorithm is based on `StarSpace <https://arxiv.org/abs/1709.03856>`_.
    However, in this implementation the loss function is slightly different and
    additional hidden layers are added together with dropout.
    This algorithm also provides similarity rankings of the labels that did not "win".
    """

    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [Featurizer]

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # ## Architecture of the used neural network
        # Hidden layer sizes for layers before the embedding layers for user message
        # and labels.
        # The number of hidden layers is equal to the length of the corresponding
        # list.
        HIDDEN_LAYERS_SIZES: {TEXT: [256, 128], LABEL: []},
        # Whether to share the hidden layer weights between user message and labels.
        SHARE_HIDDEN_LAYERS: False,
        # ## Training parameters
        # Initial and final batch sizes:
        # Batch size will be linearly increased for each epoch.
        BATCH_SIZES: [64, 256],
        # Strategy used when creating batches.
        # Can be either 'sequence' or 'balanced'.
        BATCH_STRATEGY: BALANCED,
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
        DENSE_DIMENSION: {TEXT: 512, LABEL: 20},
        # The number of incorrect labels. The algorithm will minimize
        # their similarity to the user input during training.
        NUM_NEG: 20,
        # Type of similarity measure to use, either 'auto' or 'cosine' or 'inner'.
        SIMILARITY_TYPE: AUTO,
        # The type of the loss function, either 'softmax' or 'margin'.
        LOSS_TYPE: SOFTMAX,
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
        # Sparsity of the weights in dense layers
        WEIGHT_SPARSITY: 0.8,
        # If 'True' apply dropout to sparse tensors
        SPARSE_INPUT_DROPOUT: False,
        # ## Evaluation parameters
        # How often calculate validation accuracy.
        # Small values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EPOCHS: 20,
        # How many examples to use for hold out validation set
        # Large values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EXAMPLES: 0,
    }

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        index_label_id_mapping: Optional[Dict[int, Text]] = None,
        index_tag_id_mapping: Optional[Dict[int, Text]] = None,
        model: Optional[RasaModel] = None,
    ) -> None:

        component_config = component_config or {}

        # the following properties cannot be adapted for the EmbeddingIntentClassifier
        component_config[INTENT_CLASSIFICATION] = True
        component_config[ENTITY_RECOGNITION] = False
        component_config[MASKED_LM] = False
        component_config[BILOU_FLAG] = False
        component_config[NUM_TRANSFORMER_LAYERS] = 0

        super().__init__(
            component_config, index_label_id_mapping, index_tag_id_mapping, model
        )

        common_utils.raise_warning(
            "'EmbeddingIntentClassifier' is deprecated and will be removed in version "
            "2.0. Use 'DIETClassifier' instead.",
            category=FutureWarning,
            docs=DOCS_URL_MIGRATION_GUIDE,
        )
