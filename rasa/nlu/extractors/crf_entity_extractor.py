import logging
import os
import warnings
from typing import Any, Dict, Optional, Text

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.model import Metadata
from rasa.nlu.training_data import TrainingData, Message
from rasa.constants import DOCS_BASE_URL
from rasa.nlu.components import any_of
from rasa.nlu.classifiers.diet_classifier import DIETClassifier
from rasa.nlu.constants import (
    TEXT_ATTRIBUTE,
    ENTITIES_ATTRIBUTE,
    DENSE_FEATURE_NAMES,
    SPARSE_FEATURE_NAMES,
    TOKENS_NAMES,
)
from rasa.utils.tensorflow.constants import (
    HIDDEN_LAYERS_SIZES_TEXT,
    SHARE_HIDDEN_LAYERS,
    NUM_TRANSFORMER_LAYERS,
    BATCH_SIZES,
    BATCH_STRATEGY,
    EPOCHS,
    RANDOM_SEED,
    LEARNING_RATE,
    DENSE_DIM,
    SPARSE_INPUT_DROPOUT,
    MASKED_LM,
    ENTITY_RECOGNITION,
    LABEL_CLASSIFICATION,
    EVAL_NUM_EXAMPLES,
    EVAL_NUM_EPOCHS,
    UNIDIRECTIONAL_ENCODER,
    DROPRATE,
    C2,
    BILOU_FLAG,
)
from rasa.utils.common import raise_warning
from rasa.utils.tensorflow.tf_models import RasaModel

logger = logging.getLogger(__name__)


class CRFEntityExtractor(DIETClassifier):

    provides = [ENTITIES_ATTRIBUTE]

    requires = [TOKENS_NAMES[TEXT_ATTRIBUTE]]

    # default properties (DOC MARKER - don't remove)
    defaults = {
        # 'features' is [before, word, after] array with before, word,
        # after holding keys about which features to use for each word,
        # for example, 'title' in array before will have the feature
        # "is the preceding word in title case?"
        # POS features require 'SpacyTokenizer'.
        "features": [
            ["low", "title", "upper"],
            [
                "BOS",
                "EOS",
                "low",
                "prefix5",
                "prefix2",
                "suffix5",
                "suffix3",
                "suffix2",
                "upper",
                "title",
                "digit",
            ],
            ["low", "title", "upper"],
        ],
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

    def __init__(
        self,
        component_config: Optional[Dict[Text, Any]] = None,
        featurizer: Optional[LexicalSyntacticFeaturizer] = None,
        inverted_label_dict: Optional[Dict[int, Text]] = None,
        inverted_tag_dict: Optional[Dict[int, Text]] = None,
        model: Optional[RasaModel] = None,
        batch_tuple_sizes: Optional[Dict] = None,
    ) -> None:
        component_config = component_config or {}

        # the following properties are fixed for the CRFEntityExtractor
        component_config[LABEL_CLASSIFICATION] = False
        component_config[ENTITY_RECOGNITION] = True
        component_config[MASKED_LM] = False
        component_config[NUM_TRANSFORMER_LAYERS] = 0
        component_config[SHARE_HIDDEN_LAYERS] = False

        super().__init__(
            component_config,
            inverted_label_dict,
            inverted_tag_dict,
            model,
            batch_tuple_sizes,
        )

        self.featurizer = featurizer or LexicalSyntacticFeaturizer(
            self.component_config
        )

        raise_warning(
            f"'CRFEntityExtractor' is deprecated. Use 'DIETClassifier' in"
            f"combination with 'LexicalSyntacticFeaturizer' instead.",
            category=DeprecationWarning,
            docs=f"{DOCS_BASE_URL}/nlu/components/",
        )

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        if not training_data.entity_examples:
            return

        self.featurizer.train(training_data, **kwargs)

        super().train(training_data, config, **kwargs)

    def process(self, message: Message, **kwargs: Any) -> None:

        self.featurizer.process(message, **kwargs)

        super().process(message, **kwargs)

    def persist(self, file_name: Text, model_dir: Text) -> Dict[Text, Any]:

        self.featurizer.persist(file_name, model_dir)

        return super().persist(file_name, model_dir)

    @classmethod
    def load(
        cls,
        meta: Dict[Text, Any],
        model_dir: Text = None,
        model_metadata: Metadata = None,
        cached_component: Optional["CRFEntityExtractor"] = None,
        **kwargs: Any,
    ) -> "CRFEntityExtractor":

        if not model_dir or not meta.get("file"):
            warnings.warn(
                f"Failed to load 'CRFEntityExtractor'. "
                f"Maybe the path '{os.path.abspath(model_dir)}' doesn't exist?"
            )
            return cls(component_config=meta)

        featurizer = LexicalSyntacticFeaturizer.load(
            meta, model_dir, model_metadata, cached_component, **kwargs
        )

        (
            batch_tuple_sizes,
            inv_label_dict,
            inv_tag_dict,
            label_data,
            meta,
            data_example,
        ) = cls._load_from_files(meta, model_dir)

        model = cls._load_model(inv_tag_dict, label_data, meta, data_example, model_dir)

        return cls(
            component_config=meta,
            featurizer=featurizer,
            inverted_label_dict=inv_label_dict,
            inverted_tag_dict=inv_tag_dict,
            model=model,
            batch_tuple_sizes=batch_tuple_sizes,
        )
