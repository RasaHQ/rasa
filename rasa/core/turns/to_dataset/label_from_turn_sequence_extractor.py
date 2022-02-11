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
from dataclasses import dataclass
import logging

from rasa.core.turns.to_dataset.turn_sub_sequence_generator import (
    TurnType,
    steps2str,
    TurnSequenceModifier,
)
from rasa.shared.core.domain import Domain

logger = logging.getLogger(__name__)

RawLabelType = TypeVar("RawLabelType")

TurnType = TypeVar("TurnType")


@dataclass
class LabelFromTurnsExtractor(Generic[TurnType, RawLabelType]):
    """Extracts label information from a sequence of turns."""

    name: Optional[str]
    on_training: bool = True
    on_inference: bool = False

    def __repr__(self):
        return self.__class__.__name__

    @abstractmethod
    def extract(
        self, turns: List[TurnType], training: bool, inplace_allowed: bool
    ) -> Tuple[List[TurnType], RawLabelType]:
        raise NotImplementedError

    def apply_to(
        self, turns: List[TurnType], training: bool, inplace_allowed: bool
    ) -> Tuple[List[TurnType], RawLabelType]:
        if (training and self.on_training) or (not training and self.on_inference):
            turns, raw_label = self.extract(
                turns, training=training, inplace_allowed=inplace_allowed
            )
            logger.debug(f"{self.__class__.__name__} extracted: {raw_label}")
            logger.debug(f"Remaining turns:\n{steps2str(turns)}")
            return turns, raw_label
        return turns, None

    def from_domain(self, domain: Domain) -> List[RawLabelType]:
        # TODO: do we need to be able to handle a new domain here?
        raise NotImplementedError

    @classmethod
    def apply_all(
        cls,
        label_extractors: List[Tuple[Text, LabelFromTurnsExtractor[TurnType, Any]]],
        turns: List[TurnType],
        training: bool,
        inplace_allowed: bool,
    ) -> Dict[Text, Any]:
        outputs = {}
        for extractor in label_extractors:
            turns, extracted = extractor.apply_to(
                turns=turns, training=training, inplace_allowed=inplace_allowed
            )
            if extracted:
                outputs[extractor.name] = extracted
        return turns, outputs


class FakeLabelFromTurnsExtractor(
    LabelFromTurnsExtractor[TurnType, Any], Generic[TurnType]
):
    """A fake extractor that does not extract anything ."""

    def __init__(self, turn_sequence_modifier: TurnSequenceModifier[TurnType]):
        super().__init__(
            name=None,
            on_training=turn_sequence_modifier.on_training,
            on_inference=turn_sequence_modifier.on_inference,
        )
        self._turn_sequence_modifier = turn_sequence_modifier

    def extract(
        self, turns: List[TurnType], training: bool, inplace_allowed: bool
    ) -> Tuple[List[TurnType], Optional]:
        modified_turns = self._turn_sequence_modifier.modify(
            turns=turns, training=training, inplace_allowed=inplace_allowed
        )
        return modified_turns, None
