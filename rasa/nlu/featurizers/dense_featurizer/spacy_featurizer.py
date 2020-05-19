import numpy as np
import typing
import logging
from typing import Any, Optional, Text, Dict, List, Type

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.nlu.utils.spacy_utils import SpacyNLP
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.training_data import Message, TrainingData
from rasa.nlu.constants import (
    TEXT,
    SPACY_DOCS,
    DENSE_FEATURE_NAMES,
    DENSE_FEATURIZABLE_ATTRIBUTES,
)
from rasa.utils.tensorflow.constants import POOLING, MEAN_POOLING

if typing.TYPE_CHECKING:
    from spacy.tokens import Doc


logger = logging.getLogger(__name__)


class SpacyFeaturizer(DenseFeaturizer):
    @classmethod
    def required_components(cls) -> List[Type[Component]]:
        return [SpacyNLP, SpacyTokenizer]

    defaults = {
        # Specify what pooling operation should be used to calculate the vector of
        # the CLS token. Available options: 'mean' and 'max'
        POOLING: MEAN_POOLING
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super().__init__(component_config)

        self.pooling_operation = self.component_config[POOLING]

    def _features_for_doc(self, doc: "Doc") -> np.ndarray:
        """Feature vector for a single document / sentence / tokens."""
        return np.array([t.vector for t in doc])

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:

        for example in training_data.intent_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self._set_spacy_features(example, attribute)

    def get_doc(self, message: Message, attribute: Text) -> Any:
        return message.get(SPACY_DOCS[attribute])

    def process(self, message: Message, **kwargs: Any) -> None:
        self._set_spacy_features(message)

    def _set_spacy_features(self, message: Message, attribute: Text = TEXT) -> None:
        """Adds the spacy word vectors to the messages features."""
        doc = self.get_doc(message, attribute)

        if doc is None:
            return

        # in case an empty spaCy model was used, no vectors are present
        if doc.vocab.vectors_length == 0:
            logger.debug("No features present. You are using an empty spaCy model.")
            return

        features = self._features_for_doc(doc)

        cls_token_vec = self._calculate_cls_vector(features, self.pooling_operation)
        features = np.concatenate([features, cls_token_vec])

        features = self._combine_with_existing_dense_features(
            message, features, DENSE_FEATURE_NAMES[attribute]
        )
        message.set(DENSE_FEATURE_NAMES[attribute], features)
