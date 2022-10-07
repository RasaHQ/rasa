import numpy as np
import typing
import logging
from typing import Any, Text, Dict, List, Type

from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.features import Features
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


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER, is_trainable=False
)
class SpacyFeaturizer(DenseFeaturizer, GraphComponent):
    """Featurize messages using SpaCy."""

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [SpacyTokenizer]

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["spacy"]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """The component's default config (see parent class for full docstring)."""
        return {
            **DenseFeaturizer.get_default_config(),
            # Specify what pooling operation should be used to calculate the vector of
            # the complete utterance. Available options: 'mean' and 'max'
            POOLING: MEAN_POOLING,
        }

    def __init__(self, config: Dict[Text, Any], name: Text) -> None:
        """Initializes SpacyFeaturizer."""
        super().__init__(name, config)
        self.pooling_operation = self._config[POOLING]

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> GraphComponent:
        """Creates a new component (see parent class for full docstring)."""
        return cls(config, execution_context.node_name)

    def _features_for_doc(self, doc: "Doc") -> np.ndarray:
        """Feature vector for a single document / sentence / tokens."""
        return np.array([t.vector for t in doc if t.text and t.text.strip()])

    def _get_doc(self, message: Message, attribute: Text) -> Any:
        return message.get(SPACY_DOCS[attribute])

    def process(self, messages: List[Message]) -> List[Message]:
        """Processes incoming messages and computes and sets features."""
        for message in messages:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self._set_spacy_features(message, attribute)
        return messages

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Processes the training examples in the given training data in-place.

        Args:
          training_data: Training data.

        Returns:
          Same training data after processing.
        """
        self.process(training_data.training_examples)
        return training_data

    def _set_spacy_features(self, message: Message, attribute: Text = TEXT) -> None:
        """Adds the spacy word vectors to the messages features."""
        doc = self._get_doc(message, attribute)

        if doc is None:
            return

        # in case an empty spaCy model was used, no vectors are present
        if doc.vocab.vectors_length == 0:
            logger.debug("No features present. You are using an empty spaCy model.")
            return

        sequence_features = self._features_for_doc(doc)
        sentence_features = self.aggregate_sequence_features(
            sequence_features, self.pooling_operation
        )

        final_sequence_features = Features(
            sequence_features,
            FEATURE_TYPE_SEQUENCE,
            attribute,
            self._config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sequence_features)
        final_sentence_features = Features(
            sentence_features,
            FEATURE_TYPE_SENTENCE,
            attribute,
            self._config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sentence_features)

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass
