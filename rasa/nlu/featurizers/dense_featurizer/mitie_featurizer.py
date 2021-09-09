import numpy as np
import logging
import typing
from typing import Any, List, Text, Optional, Dict, Tuple

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.featurizers.dense_featurizer.dense_featurizer import DenseFeaturizer2
from rasa.shared.nlu.training_data.features import Features
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.nlu.training_data.message import Message
from rasa.nlu.constants import (
    DENSE_FEATURIZABLE_ATTRIBUTES,
    FEATURIZER_CLASS_ALIAS,
    TOKENS_NAMES,
)
from rasa.shared.nlu.constants import FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE
from rasa.utils.tensorflow.constants import MEAN_POOLING, POOLING
from rasa.nlu.featurizers.dense_featurizer._mitie_featurizer import MitieFeaturizer

if typing.TYPE_CHECKING:
    import mitie

logger = logging.getLogger(__name__)

# TODO: This is a workaround around until we have all components migrated to
# `GraphComponent`.
MitieFeaturizer = MitieFeaturizer


class MitieFeaturizerGraphComponent(DenseFeaturizer2, GraphComponent):
    """A class that featurizes using Mitie."""

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {
            **DenseFeaturizer2.get_default_config(),
            # Specify what pooling operation should be used to calculate the vector of
            # the complete utterance. Available options: 'mean' and 'max'
            POOLING: MEAN_POOLING,
        }

    def __init__(
        self,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> None:
        """Instantiates a new `MitieFeaturizerGraphComponent` instance."""
        super().__init__(config)
        self.config = config
        self.pooling_operation = self.config["pooling"]

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> "MitieFeaturizerGraphComponent":
        """Creates a new untrained component (see parent class for full docstring)."""
        return cls(config, model_storage, resource, execution_context)

    @staticmethod
    def required_packages() -> List[Text]:
        """Any extra python dependencies required for this component to run."""
        return ["mitie", "numpy"]

    def ndim(self, feature_extractor: "mitie.total_word_feature_extractor") -> int:
        """Returns the number of dimensions."""
        return feature_extractor.num_dimensions

    def train_process(self, training_data: TrainingData, **kwargs: Any,) -> None:
        """Trains the featurizer."""
        mitie_feature_extractor = self._mitie_feature_extractor(**kwargs)
        for example in training_data.training_examples:
            for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
                self.process_training_example(
                    example, attribute, mitie_feature_extractor
                )

    def process_training_example(
        self, example: Message, attribute: Text, mitie_feature_extractor: Any
    ) -> None:
        """Processes training examples."""
        tokens = example.get(TOKENS_NAMES[attribute])

        if tokens is not None:
            sequence_features, sentence_features = self.features_for_tokens(
                tokens, mitie_feature_extractor
            )

            self._set_features(example, sequence_features, sentence_features, attribute)

    def process(self, message: Message, **kwargs: Any) -> None:
        """Processes a message."""
        mitie_feature_extractor = self._mitie_feature_extractor(**kwargs)
        for attribute in DENSE_FEATURIZABLE_ATTRIBUTES:
            tokens = message.get(TOKENS_NAMES[attribute])
            if tokens:
                sequence_features, sentence_features = self.features_for_tokens(
                    tokens, mitie_feature_extractor
                )

                self._set_features(
                    message, sequence_features, sentence_features, attribute
                )

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
            self.config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sequence_features)

        final_sentence_features = Features(
            sentence_features,
            FEATURE_TYPE_SENTENCE,
            attribute,
            self.config[FEATURIZER_CLASS_ALIAS],
        )
        message.add_features(final_sentence_features)

    def _mitie_feature_extractor(self, **kwargs: Any) -> Any:
        mitie_feature_extractor = kwargs.get("mitie_feature_extractor")
        if not mitie_feature_extractor:
            raise Exception(
                "Failed to train 'MitieFeaturizer'. "
                "Missing a proper MITIE feature extractor. "
                "Make sure this component is preceded by "
                "the 'MitieNLP' component in the pipeline "
                "configuration."
            )
        return mitie_feature_extractor

    def features_for_tokens(
        self,
        tokens: List[Token],
        feature_extractor: "mitie.total_word_feature_extractor",
    ) -> Tuple[np.ndarray, np.ndarray]:
        # calculate features
        sequence_features = []
        for token in tokens:
            sequence_features.append(feature_extractor.get_feature_vector(token.text))
        sequence_features = np.array(sequence_features)

        sentence_fetaures = self.aggregate_sequence_features(
            sequence_features, self.pooling_operation
        )

        return sequence_features, sentence_fetaures
