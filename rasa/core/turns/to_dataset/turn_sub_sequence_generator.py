from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging
from typing import Generic, Optional, List, TypeVar, Iterator, Set, Tuple, Any, Dict

from rasa.core.turns.turn import Turn

logger = logging.getLogger(__name__)

TurnType = TypeVar("TurnType")


def steps2str(steps: List[Any]) -> str:
    indent = " " * 2
    steps = "\n".join(f"{indent}{idx:2}. {turn}" for idx, turn in enumerate(steps))
    return f"[\n{steps}\n]"


class TurnSubSequenceGenerator(Generic[TurnType], ABC):
    """Generates multiple sub-sequences from a single sequence of turns."""

    def __init__(
        self,
        preprocessing: Optional[List[TurnSequenceModifier[TurnType]]],
        filters: Optional[List[TurnSequenceValidation[TurnType]]],
        ignore_duplicates: bool,
        modifiers: Optional[List[TurnSequenceModifier[TurnType]]],
        result_filters: Optional[List[TurnSequenceValidation[TurnType]]],
    ) -> None:
        """
        Args:
            filters: only applied during training
            ...
        """
        self._preprocessing = preprocessing or []
        self._filters = filters or []
        self._ignore_duplicates = ignore_duplicates
        self._modifiers = modifiers or []
        self._result_filters = result_filters or []

    def apply_to(
        self,
        turns: List[TurnType],
        training: bool,
        limit: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Iterator[List[Turn]]:
        """

        During training, the whole sequence of turns is processed.

        During validation, we extract multiple (or no) sub-sequences of turns from the
        given sequence of turns: Each subsequence of turns from the 0-th to the i-th
        turn that passes the given sequence filters and is not a duplicate of any
        other subsequence created so far.

        In both cases, the same modifiers are applied.

        Args:
            turns: ...
            training: ...
            limit: ...
            context: ...

        """
        steps = [len(turns) + 1] if not training else range(2, len(turns) + 1)
        num_generated = 0

        identifier_cache: Set[int] = set()

        logger.debug(f"Start generating subsequences from:\n{steps2str(turns)}")

        preprocessed_turns = TurnSequenceModifier.apply_all(
            modifiers=self._preprocessing,
            turns=turns,
            training=True,
            inplace_allowed=(not training),
            context=context,
        )

        logger.debug(f"Applied pre-processing:\n{steps2str(preprocessed_turns)}")

        for idx in steps:

            if limit and num_generated >= limit:
                return

            # we'll make a copy of this subsequence, once we know we continue with it
            subsequence = preprocessed_turns[:idx]

            logger.debug(
                f"Attempt to generate from subsequence:\n{steps2str(subsequence)}"
            )

            # during training - skip if it does not pass filters
            if training and self._filters:
                failed = TurnSequenceValidation.apply_all(
                    validations=self._filters, turns=subsequence
                )
                if failed:
                    logger.debug(f"Continue (filter {failed})")
                    continue

            # apply modifiers
            subsequence = TurnSequenceModifier.apply_all(
                modifiers=self._modifiers,
                turns=subsequence,
                training=training,
                inplace_allowed=(not training),
                context=context,
            )

            if self._modifiers:
                logger.debug(f"Modified subsequence:\n{steps2str(subsequence)}")

            # during training - skip if it does not pass filters
            if training and self._result_filters:
                failed = TurnSequenceValidation.apply_all(
                    validations=self._result_filters, turns=subsequence
                )
                if failed:
                    logger.debug(f"Continue (filter {failed})")
                    continue

            # during training - skip if it is a duplicate
            if training and self._ignore_duplicates:
                identifier = hash(tuple(subsequence))
                logger.debug(f"Identifier: {identifier}")
                if identifier in identifier_cache:
                    logger.debug(f"Continue (duplicate of other subsequence)")
                    continue
                else:
                    identifier_cache.add(identifier)

            num_generated += 1
            yield subsequence


@dataclass
class TurnSequenceValidation(Generic[TurnType]):
    """Determines whether or not a given list of turns satisfies some criteria."""

    @abstractmethod
    def validate(self, turns: List[TurnType]) -> bool:
        raise NotImplementedError

    @staticmethod
    def apply_all(
        validations: List[TurnSequenceValidation[TurnType]],
        turns: List[TurnType],
    ) -> Optional[TurnSequenceValidation[Turn]]:
        return next(
            (
                validation
                for validation in validations
                if not validation.validate(turns)
            ),
            None,
        )


@dataclass
class EndsWith(TurnSequenceValidation[TurnType]):
    turn_type: str
    offset: int = -1

    def validate(self, turns: List[TurnType]) -> bool:
        return turns[self.offset].get_type() == self.turn_type


@dataclass
class HasMinLength(TurnSequenceValidation[TurnType]):
    min_length: int

    def validate(self, turns: List[TurnType]) -> bool:
        return len(turns) >= self.min_length


@dataclass
class TurnSequenceModifier(Generic[TurnType], ABC):
    """Returns a modified list of turns.

    Must not modify the given list of turns.
    """

    on_training: bool = True
    on_inference: bool = True

    @abstractmethod
    def modify(
        self,
        turns: List[TurnType],
        training: bool,
        inplace_allowed: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TurnType]:
        """Returns a modified turn sequence.

        Args:
            turns: a list of turns
            training: whether or not we're in training mode
            inplace_allowed: if this is set to `False` then the single turns of the given
               sequence of turns may be modified inplace_allowed; otherwise the given turns
               must not be modified but the returned turn sequence may contain new
               turn objects
            context: more context information that may be needed by modifiers to decide
               how to modify turns exactly (e.g. whether to use text for to_dataset
               is something that needs to be configurable dynamically)
        Returns:
            a modified turn sequence
        """
        raise NotImplementedError

    def apply_to(
        self,
        turns: List[TurnType],
        inplace_allowed: bool,
        training: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TurnType]:
        if (self.on_training and training) or (self.on_inference and not training):
            return self.modify(
                turns=turns,
                training=training,
                inplace_allowed=inplace_allowed,
                context=context,
            )
        return turns

    @staticmethod
    def apply_all(
        modifiers: List[TurnSequenceModifier[TurnType]],
        turns: List[TurnType],
        training: bool,
        inplace_allowed: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TurnType]:
        for modifier in modifiers:
            turns = modifier.apply_to(
                turns=turns,
                inplace_allowed=inplace_allowed,
                training=training,
                context=context,
            )
        return turns


class RemoveLastTurn(TurnSequenceModifier[TurnType], Generic[TurnType]):
    def modify(
        self,
        turns: List[TurnType],
        training: bool,
        inplace_allowed: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TurnType]:
        return turns[:-1]


@dataclass
class KeepMaxHistory(TurnSequenceModifier[TurnType], Generic[TurnType]):

    max_history: Optional[int] = None
    offset_for_training: int = 0

    def modify(
        self,
        turns: List[TurnType],
        training: bool,
        inplace_allowed: bool,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[TurnType]:
        """Keeps the last `max_history`(+1) turns during inference (training)."""
        if self.max_history is not None:
            keep = (
                (self.max_history + self.offset_for_training)
                if training
                else self.max_history
            )
            turns = turns[-keep:]
        return turns
