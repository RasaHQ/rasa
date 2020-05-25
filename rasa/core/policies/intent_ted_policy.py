import copy
import logging
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from typing import Any, List, Optional, Text, Dict, Tuple, Union

import rasa.utils.io as io_utils
from rasa.core.domain import Domain
from rasa.core.featurizers import (
    TrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
    LabelTokenizerSingleStateFeaturizer,
    MaxHistoryTrackerFeaturizer,
    IntentMaxHistoryFeaturizer,
    IntentTokenizerSingleStateFeaturizer,
)
from rasa.core.policies.policy import Policy
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.core.constants import DEFAULT_POLICY_PRIORITY, DIALOGUE
from rasa.core.trackers import DialogueStateTracker
from rasa.utils import train_utils
from rasa.utils.tensorflow import layers
from rasa.utils.tensorflow.transformer import TransformerEncoder
from rasa.utils.tensorflow.models import RasaModel
from rasa.utils.tensorflow.model_data import RasaModelData, FeatureSignature
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
    SOFTMAX,
    AUTO,
    BALANCED,
    TENSORBOARD_LOG_DIR,
    TENSORBOARD_LOG_LEVEL,
)


logger = logging.getLogger(__name__)

DIALOGUE_FEATURES = f"{DIALOGUE}_features"
LABEL_FEATURES = f"{LABEL}_features"
LABEL_IDS = f"{LABEL}_ids"

SAVE_MODEL_FILE_NAME = "ted_policy"


class IntentTEDPolicy(TEDPolicy):
    """Transformer Embedding Dialogue (TED) Policy is described in
    https://arxiv.org/abs/1910.00486.

    This policy has a pre-defined architecture, which comprises the
    following steps:
        - concatenate user input (user intent and entities), previous system actions,
          slots and active forms for each time step into an input vector to
          pre-transformer embedding layer;
        - feed it to transformer;
        - apply a dense layer to the output of the transformer to get embeddings of a
          dialogue for each time step;
        - apply a dense layer to create embeddings for system actions for each time
          step;
        - calculate the similarity between the dialogue embedding and embedded system
          actions. This step is based on the StarSpace
          (https://arxiv.org/abs/1709.03856) idea.
    """

    SUPPORTS_ONLINE_TRAINING = True

    # please make sure to update the docs when changing a default parameter
    defaults = {
        # ## Architecture of the used neural network
        # Hidden layer sizes for layers before the dialogue and label embedding layers.
        # The number of hidden layers is equal to the length of the corresponding
        # list.
        HIDDEN_LAYERS_SIZES: {DIALOGUE: [], LABEL: []},
        # Number of units in transformer
        TRANSFORMER_SIZE: 128,
        # Number of transformer layers
        NUM_TRANSFORMER_LAYERS: 1,
        # Number of attention heads in transformer
        NUM_HEADS: 4,
        # If 'True' use key relative embeddings in attention
        KEY_RELATIVE_ATTENTION: False,
        # If 'True' use value relative embeddings in attention
        VALUE_RELATIVE_ATTENTION: False,
        # Max position for relative embeddings
        MAX_RELATIVE_POSITION: None,
        # ## Training parameters
        # Initial and final batch sizes:
        # Batch size will be linearly increased for each epoch.
        BATCH_SIZES: [8, 32],
        # Strategy used whenc creating batches.
        # Can be either 'sequence' or 'balanced'.
        BATCH_STRATEGY: BALANCED,
        # Number of epochs to train
        EPOCHS: 1,
        # Set random seed to any 'int' to get reproducible results
        RANDOM_SEED: None,
        # ## Parameters for embeddings
        # Dimension size of embedding vectors
        EMBEDDING_DIMENSION: 20,
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
        MAX_NEG_SIM: -0.2,
        # If 'True' the algorithm only minimizes maximum similarity over
        # incorrect intent labels, used only if 'loss_type' is set to 'margin'.
        USE_MAX_NEG_SIM: True,
        # If 'True' scale loss inverse proportionally to the confidence
        # of the correct prediction
        SCALE_LOSS: True,
        # ## Regularization parameters
        # The scale of regularization
        REGULARIZATION_CONSTANT: 0.001,
        # The scale of how important is to minimize the maximum similarity
        # between embeddings of different labels,
        # used only if 'loss_type' is set to 'margin'.
        NEGATIVE_MARGIN_SCALE: 0.8,
        # Dropout rate for embedding layers of dialogue features.
        DROP_RATE_DIALOGUE: 0.1,
        # Dropout rate for embedding layers of label, e.g. action, features.
        DROP_RATE_LABEL: 0.0,
        # Dropout rate for attention.
        DROP_RATE_ATTENTION: 0,
        # Sparsity of the weights in dense layers
        WEIGHT_SPARSITY: 0.8,
        # ## Evaluation parameters
        # How often calculate validation accuracy.
        # Small values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EPOCHS: 20,
        # How many examples to use for hold out validation set
        # Large values may hurt performance, e.g. model accuracy.
        EVAL_NUM_EXAMPLES: 0,
        # If you want to use tensorboard to visualize training and validation metrics,
        # set this option to a valid output directory.
        TENSORBOARD_LOG_DIR: None,
        # Define when training metrics for tensorboard should be logged.
        # Either after every epoch or for every training step.
        # Valid values: 'epoch' and 'minibatch'
        TENSORBOARD_LOG_LEVEL: "epoch",
    }

    @staticmethod
    def _standard_featurizer(max_history: Optional[int] = None) -> TrackerFeaturizer:
        return IntentMaxHistoryFeaturizer(
            IntentTokenizerSingleStateFeaturizer(), max_history=max_history
        )
