from __future__ import annotations
from abc import abstractmethod, ABC
from collections import Counter
from typing import Generic, Iterable, Text, Optional, Dict, Any, TypeVar

from rasa.nlu.constants import FEATURIZER_CLASS_ALIAS
from rasa.shared.nlu.training_data.features import Features
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.exceptions import InvalidConfigException
from rasa.shared.nlu.constants import FEATURE_TYPE_SENTENCE, FEATURE_TYPE_SEQUENCE

FeatureType = TypeVar("FeatureType")


class Featurizer(Generic[FeatureType], ABC):
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
            not specify an `alias` (or this `alias` is None)
        """
        super().__init__()
        self.validate_config(config)
        self._config = config
        self._identifier = self._config[FEATURIZER_CLASS_ALIAS] or name

    @classmethod
    @abstractmethod
    def validate_config(cls, config: Dict[Text, Any]) -> None:
        """Validates that the component is configured properly."""
        ...

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
                wrapped_feature = Features(features, type, attribute, self._identifier)
                message.add_features(wrapped_feature)

    @staticmethod
    def raise_if_featurizer_configs_are_not_compatible(
        featurizer_configs: Iterable[Dict[Text, Any]],
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
                f"Please update your config such that each featurizer has a unique "
                f"alias."
            )
