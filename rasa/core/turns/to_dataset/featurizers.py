from __future__ import annotations
from abc import abstractmethod
from typing import (
    Any,
    Dict,
    List,
    Text,
    TypeVar,
    Generic,
    Tuple,
    Optional,
)
import logging
from abc import ABC

import numpy as np

from rasa.core.featurizers.precomputation import MessageContainerForCoreFeaturization
from rasa.core.turns.to_dataset.label_from_turn_sequence_extractor import (
    LabelFromTurnsExtractor,
)
from rasa.core.turns.to_dataset.turn_sub_sequence_generator import TurnType
from rasa.core.turns.utils.entity_tags_encoder import EntityTagsEncoder
from rasa.core.turns.utils.feature_lookup import FeatureLookup
from rasa.core.turns.utils.multi_label_encoder import MultiLabelEncoder
from rasa.core.turns.utils.trainable import Trainable
from rasa.nlu.constants import TOKENS_NAMES
from rasa.shared.core.domain import Domain
from rasa.shared.nlu.constants import TEXT
from rasa.shared.nlu.training_data.features import Features

logger = logging.getLogger(__name__)

RawLabelType = TypeVar("RawLabelType")


class TurnFeaturizer(Trainable, Generic[TurnType], ABC):
    """Featurize a single turn."""

    def train(
        self,
        domain: Domain,
    ) -> None:
        self._train(
            domain=domain,
        )
        self.mark_as_trained()

    @abstractmethod
    def _train(self, domain: Domain):
        raise NotImplementedError

    def featurize(
        self,
        turn: TurnType,
        precomputations: Optional[MessageContainerForCoreFeaturization],
        training: bool = True,
    ) -> Dict[str, List[Features]]:
        self.raise_if_not_trained()
        return self._featurize(
            turn=turn, precomputations=precomputations, training=training
        )

    @abstractmethod
    def _featurize(
        self,
        turn: TurnType,
        precomputations: Optional[MessageContainerForCoreFeaturization],
        training: bool = True,
    ) -> Dict[str, List[Features]]:
        raise NotImplementedError


class LabelFeaturizer(Trainable, Generic[RawLabelType], ABC):
    """Converts a label to `Features`."""

    def train(
        self,
        domain: Domain,
    ) -> None:
        self._train(domain=domain)
        self.mark_as_trained()

    @abstractmethod
    def _train(
        self,
        domain: Domain,
    ):
        raise NotImplementedError

    def featurize(
        self,
        raw_label: RawLabelType,
        precomputations: Optional[MessageContainerForCoreFeaturization],
    ) -> List[Features]:
        self.raise_if_not_trained()
        return self._featurize(raw_label=raw_label, precomputations=precomputations)

    @abstractmethod
    def _featurize(
        self,
        raw_label: RawLabelType,
        precomputations: Optional[MessageContainerForCoreFeaturization],
    ) -> List[Features]:
        raise NotImplementedError


class LabelIndexer(Trainable, Generic[TurnType, RawLabelType], ABC):
    """Converts a label to an index."""

    def train(
        self, domain: Domain, extractor: LabelFromTurnsExtractor[TurnType, RawLabelType]
    ) -> None:
        self._train(domain=domain, extractor=extractor)
        self.mark_as_trained()

    @abstractmethod
    def _train(
        self, domain: Domain, extractor: LabelFromTurnsExtractor[TurnType, RawLabelType]
    ) -> None:
        raise NotImplementedError

    def index(
        self,
        raw_label: Optional[RawLabelType],
    ) -> np.ndarray:
        self.raise_if_not_trained()
        return self._index(raw_label=raw_label)

    @abstractmethod
    def _index(self, raw_label: Optional[RawLabelType]) -> np.ndarray:
        raise NotImplementedError


class LabelFeaturizerViaLookup(LabelFeaturizer[Any]):
    """Featurizes a label via the lookup table."""

    def __init__(self, attribute: Text) -> None:
        super().__init__()
        self.attribute = attribute

    def _train(
        self,
        domain: Domain,
    ) -> None:
        pass

    def _featurize(
        self,
        raw_label: Any,
        precomputations: Optional[MessageContainerForCoreFeaturization],
    ) -> List[Features]:
        if raw_label is None:
            return []
        return FeatureLookup.lookup_features(
            message_data={self.attribute: raw_label}, precomputations=precomputations
        )[self.attribute]


class FeaturizeViaEntityTagsEncoder(LabelFeaturizer[Tuple[Text, Dict[Text, Any]]]):
    """Featurizes entities via an entity tags encoder."""

    def __init__(self, bilou_tagging: bool = False) -> None:
        super().__init__()
        self._bilou_tagging = bilou_tagging

    def _train(
        self,
        domain: Domain,
    ) -> None:
        self._entity_tags_encoder = EntityTagsEncoder(
            domain=domain,
            bilou_tagging=self._bilou_tagging,
        )

    def _featurize(
        self,
        raw_label: Tuple[Text, Dict[Text, Any]],
        precomputations: Optional[MessageContainerForCoreFeaturization],
    ) -> List[Features]:
        if precomputations is None:
            return []

        if not isinstance(raw_label, Tuple) or len(raw_label) != 2:
            # TODO: typing should help identify issues with this
            raise RuntimeError(
                "Expected text and entities. Use a label extractor that returns "
                "Text and entities"
            )

        text, entities = raw_label

        # Don't bother encoding anything if there are less than 2 entity tags,
        # because we won't train any entity extractor anyway.
        if (
            not text
            or not entities
            or self._entity_tags_encoder.entity_tag_spec.num_tags < 2
        ):
            return []

        message = precomputations.lookup_message(user_text=text)
        text_tokens = message.get(TOKENS_NAMES[TEXT])
        return self._entity_tags_encoder.encode(
            text_tokens=text_tokens, entities=entities
        )


class IndexerFromLabelExtractor(
    LabelIndexer[TurnType, RawLabelType],
    Generic[TurnType, RawLabelType],
):
    """Converts a label to an index.

    This indexer must be trained and uses a given `LabelFromTurnsExtractor` to
    retrieve all possible labels from the given domain. These labels are then used
    to create a label to index mapping.
    """

    def _train(
        self,
        domain: Domain,
        extractor: LabelFromTurnsExtractor[TurnType, RawLabelType],
    ) -> None:
        all_labels = extractor.from_domain(domain=domain)
        self._multi_label_encoder = MultiLabelEncoder(dimension_names=all_labels)

    def _index(
        self,
        raw_label: RawLabelType,
    ) -> np.ndarray:
        return self._multi_label_encoder.encode_as_index_array([raw_label])


class MultiIndexerFromLabelExtractor(
    IndexerFromLabelExtractor[TurnType, List[RawLabelType]],
    Generic[TurnType, RawLabelType],
):
    """Converts multi-labels to a list of indices.

    This indexer must be trained and uses a given `LabelFromTurnsExtractor` to
    retrieve all possible labels from the given domain. These labels are then used
    to create a label to index mapping.
    """

    def _index(
        self,
        raw_label: List[RawLabelType],
    ) -> np.ndarray:
        return self._multi_label_encoder.encode_as_index_array(raw_label)
