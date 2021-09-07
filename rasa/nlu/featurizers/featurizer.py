from __future__ import annotations
from abc import abstractmethod
from collections import Counter
from rasa.nlu.tokenizers.tokenizer import Tokenizer
from typing import Generic, Iterable, List, Text, Optional, Dict, Any, TypeVar

from rasa.nlu.constants import FEATURIZER_CLASS_ALIAS
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.constants import FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE

# TODO: remove after all featurizers have been migrated
from rasa.nlu.featurizers._featurizer import (
    Featurizer,
    SparseFeaturizer,
    DenseFeaturizer,
)

Featurizer = Featurizer
SparseFeaturizer = SparseFeaturizer
DenseFeaturizer = DenseFeaturizer

FeatureType = TypeVar("FeatureType")


class Featurizer2(Generic[FeatureType]):
    """Base class for all featurizers."""

    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        """Returns the component's default config."""
        return {FEATURIZER_CLASS_ALIAS: None}

    def __init__(self, name: Text, config: Dict[Text, Any]) -> None:
        """Instantiates a new featurizer.

        Args:
          config: configuration
          name: a name that can be used as identifier, in case the configuration does
            not specify an `alias`
        """
        super().__init__()
        self.validate_config(config)
        self._config = {**self.get_default_config(), **config}
        self._identifier = self._config.get(FEATURIZER_CLASS_ALIAS, name)

    @property
    def identifier(self) -> Text:
        """Returns the name of this featurizer.

        Every feature created by this featurizer will contain this identifier as
        `origin` information.
        """
        return self._identifier

    @classmethod
    @abstractmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        ...

    @classmethod
    @abstractmethod
    def validate_compatibility_with_tokenizer(cls, tokenizer: Tokenizer) -> None:
        """Validate that the featurizer is compatible with the given tokenizer."""
        # TODO: add (something like) this to recipe validation
        # TODO: replace tokenizer by config of tokenizer to enable static check
        ...

    @abstractmethod
    def process(self, messages: List[Message]) -> List[Message]:
        """Featurizes all given messages in-place.

        Args:
          messages: messages to be featurized
        Returns:
          the same list with the same messages after featurization
        """
        ...

    def process_training_data(self, training_data: TrainingData) -> TrainingData:
        """Processes the training examples in the given training data in-place.

        Args:
          training_data: the training data
        Returns:
          same training data after processing
        """
        self.process(training_data.training_examples)
        return training_data

    def add_features_to_message(
        self,
        sequence: FeatureType,
        sentence: Optional[FeatureType],
        attribute: Text,
        message: Message,
    ) -> None:
        """Adds sequence and sentence features for the attribute to the given message.

        Args:
          sequence: sequence feature matrix
          sentence: sentence feature matrix
          attribute: the attribute which both features describe
          message: the message to which we want to add those features
        """
        for type, features in [
            (FEATURE_TYPE_SEQUENCE, sequence),
            (FEATURE_TYPE_SENTENCE, sentence),
        ]:
            if features is not None:
                wrapped_feature = Features(features, type, attribute, self.identifier,)
                message.add_features(wrapped_feature)

    @staticmethod
    def validate_configs_compatible(
        featurizer_configs: Iterable[Dict[Text, Any]]
    ) -> None:
        """Validates that the given configurations of featurizers can be used together.

        Raises:
          `InvalidConfigException` if the given featurizers should not be used in
            the same graph.
        """
        # NOTE: this assumes the names given via the execution context are unique
        alias_counter = Counter(
            config[FEATURIZER_CLASS_ALIAS]
            for config in featurizer_configs
            if FEATURIZER_CLASS_ALIAS in config
        )
        if not alias_counter:  # no alias found
            return
        if alias_counter.most_common(1)[0][1] > 1:
            raise InvalidConfigException(
                f"Expected the featurizers to have unique names but found "
                f" (name, count): {alias_counter.most_common()}. "
                f"Please update your recipe such that each featurizer has a unique "
                f"alias."
            )
