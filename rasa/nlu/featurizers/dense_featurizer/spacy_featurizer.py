import numpy as np
import typing
import logging
from typing import Any, Optional, Text, Dict, List, Type

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.components import Component
from rasa.nlu.featurizers.featurizer import DenseFeaturizer
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.utils.spacy_utils import SpacyNLP
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import (
    SPACY_DOCS,
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
)
from rasa.shared.nlu.constants import TEXT, FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
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
        # the complete utterance. Available options: 'mean' and 'max'
        POOLING: MEAN_POOLING
    }

    def __init__(self, component_config: Optional[Dict[Text, Any]] = None):
        super().__init__(component_config)

        self.pooling_operation = self.component_config[POOLING]

    def _features_for_doc(self, doc: "Doc") -> np.ndarray:
        """Feature vector for a single document / sentence / tokens."""
        return np.array([t.vector for t in doc if t.text and t.text.strip()])

    def train(
        self,
        training_data: TrainingData,
        config: Optional[RasaNLUModelConfig] = None,
        **kwargs: Any,
    ) -> None:
        """Process and featurize all messages in the training data."""
        msg_pipe = training_data.add_features(self.unique_name)
        try:
            msg = next(msg_pipe)
            while msg:
                features = self.message_to_features(msg)
                msg = msg_pipe.send(features)
        except StopIteration:
            pass

    def get_doc(self, message: Message, attribute: Text) -> Any:
        """Return the spacy doc for the given attribute."""
        return message.get(SPACY_DOCS[attribute])

    def process(self, message: Message, **kwargs: Any) -> None:
        """Calculate and add features in-place to the message object."""
        for feature in self.message_to_features(message):
            message.add_features(feature)

    def message_to_features(self, message: Message) -> List[Features]:
        """Calculates features for all relevant attributes of the message."""
        features = []
        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            features.extend(self.message_attribute_to_features(message, attribute))
        return features

    def message_attribute_to_features(
        self, message: Message, attribute: Text = TEXT
    ) -> List[Features]:
        """Calculates features for a specific attribute of the message."""
        doc = self.get_doc(message, attribute)

        if doc is None:
            return []

        # in case an empty spaCy model was used, no vectors are present
        if doc.vocab.vectors_length == 0:
            logger.debug("No features present. You are using an empty spaCy model.")
            return []

        sequence_features = self._features_for_doc(doc)
        sentence_features = self._calculate_sentence_features(
            sequence_features, self.pooling_operation
        )

        return [
            Features(
                sequence_features,
                FEATURE_TYPE_SEQUENCE,
                attribute,
                self.component_config[FEATURIZER_CLASS_ALIAS],
            ),
            Features(
                sentence_features,
                FEATURE_TYPE_SENTENCE,
                attribute,
                self.component_config[FEATURIZER_CLASS_ALIAS],
            ),
        ]
