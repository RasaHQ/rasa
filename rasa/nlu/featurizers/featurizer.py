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

    def __init__(self, config: Optional[Dict[Text, Any]] = None) -> None:
        """Instantiates a new featurizer.

        Args:
          config: configuration
        """
        super().__init__()
        self.validate_config(config)
        self._config = config
        # TODO: we can't use COMPONENT_INDEX anymore to create a unique name
        # on the fly --> take care of that in graph validation
        self._identifier = self._config.get(FEATURIZER_CLASS_ALIAS, None)

    @property
    def identifier(self) -> Text:
        """Returns an identifier for this featurizer.

        This name should be unique among all featurizers in the graph where this
        component is used.
        """
        return self._identifier

    # TODO: validate config for this component in the context of a recipe

    @classmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the featurizer is configured properly."""
        identifier = config.get(FEATURIZER_CLASS_ALIAS, [])
        if not identifier:
            raise ValueError(
                f"Expected the config to map key `{FEATURIZER_CLASS_ALIAS}` "
                f"to some name only found {config}"
            )

    def validate_compatibility_with_tokenizer(self, tokenizer: Tokenizer) -> None:
        """Validate that the featurizer is compatible with the given tokenizer."""
        # NOTE: this wasn't done before, there was just a comment
        # TODO: add (something like) this to graph validation
        pass

    # @abstractmethod
    # def train(self, training_data: TrainingData,) -> None:
    #     """Trains the featurizer.
    # NOTE: do we want a common train? For count vectorizer we then need some
    # `setup(spacy_nlp)` - `train(spacy_nlp, training_data)` (and no common interface)

    @abstractmethod
    def process(self, messages: List[Message],) -> List[Message]:
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
    def validate_featurizers_compatible(featurizers: Iterable[Featurizer2]) -> None:
        """Validates that the given list of featurizers can be used together.

        Raises:
          `InvalidConfigException` if the given featurizers should not be used in
            the same graph.
        """
        id_counter = Counter(featurizer.identifier for featurizer in featurizers)
        if id_counter.most_common(1)[0][1] > 0:
            raise InvalidConfigException(
                f"Expected the featurizers to have unique names but found "
                f" (name, count): {id_counter.most_common()}. "
                f"Please update your recipe such that each featurizer has a unique "
                f"name."
            )
