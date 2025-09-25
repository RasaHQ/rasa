from __future__ import annotations

import logging
import typing
from typing import Any, Dict, List, Text, Tuple, Type

# importing mitie at module level to ensure error is raised if mitie is not installed
import mitie
import numpy as np

from rasa.engine.graph import ExecutionContext, GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
    TOKENS_NAMES,
)
from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer
from rasa.nlu.tokenizers.tokenizer import Token, Tokenizer
from rasa.nlu.utils.mitie_utils import MitieModel, MitieNLP
from rasa.shared.nlu.constants import FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.tensorflow.constants import MEAN_POOLING, POOLING

if typing.TYPE_CHECKING:
    import mitie

logger = logging.getLogger(__name__)


@DefaultV1Recipe.register(
    DefaultV1Recipe.ComponentType.MESSAGE_FEATURIZER,
    is_trainable=False,
    model_from="MitieNLP",
)
class MitieFeaturizer(DenseFeaturizer, GraphComponent):
    """A class that featurizes using Mitie."""

    @classmethod
    def required_components(cls) -> List[Type]:
        """Components that should be included in the pipeline before this component."""
        return [MitieNLP, Tokenizer]

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            **DenseFeaturizer.get_default_config(),
            # Specify what pooling operation should be used to calculate the vector of
            # the complete utterance. Available options: 'mean' and 'max'
            POOLING: MEAN_POOLING,
        }

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["mitie", "numpy"]

    def __init__(
        self, config: Dict[Text, Any], execution_context: ExecutionContext
    ) -> None:
        """Instantiates a new `MitieFeaturizer` instance."""
        super().__init__(execution_context.node_name, config)
        self.pooling_operation = self._config[POOLING]

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> MitieFeaturizer:
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, execution_context)

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        pass

    def ndim(self, feature_extractor: "mitie.total_word_feature_extractor") -> int:
        """Returns the number of dimensions."""
        return feature_extractor.num_dimensions

    def process(self, messages: List[Message], model: MitieModel) -> List[Message]:
        """Featurizes all given messages in-place.

        Returns:
          The given list of messages which have been modified in-place.
        """
        for message in messages:
            self._process_message(message, model)
        return messages

    def process_training_data(
        self, training_data: TrainingData, model: MitieModel
    ) -> TrainingData:
        """Processes the training examples in the given training data in-place.

        Args:
          training_data: Training data.
          model: A Mitie model.

        Returns:
          Same training data after processing.
        """
        self.process(training_data.training_examples, model)
        return training_data

    def _process_message(self, message: Message, model: MitieModel) -> None:
        """Processes a message."""
        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            self._process_training_example(
                message, attribute, model.word_feature_extractor
            )

    def _process_training_example(
        self,
        example: Message,
        attribute: Text,
        mitie_feature_extractor: "mitie.total_word_feature_extractor",
    ) -> None:
        tokens = example.get(TOKENS_NAMES[attribute])

        if tokens:
            sequence_features, sentence_features = self.features_for_tokens(
                tokens, mitie_feature_extractor
            )

            self._set_features(example, sequence_features, sentence_features, attribute)

    def _set_features(
        self,
        message: Message,
        sequence_features: np.ndarray,
        sentence_features: np.ndarray,
        attribute: Text,
    ) -> None:
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

    def features_for_tokens(
        self,
        tokens: List[Token],
        feature_extractor: "mitie.total_word_feature_extractor",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Calculates features."""
        sequence_features = np.array(
            [feature_extractor.get_feature_vector(token.text) for token in tokens]
        )

        sentence_fetaures = self.aggregate_sequence_features(
            sequence_features, self.pooling_operation
        )

        return sequence_features, sentence_fetaures
