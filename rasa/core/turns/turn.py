from __future__ import annotations
from abc import abstractmethod
from dataclasses import dataclass
import logging
from typing import (
    Any,
    Optional,
    Text,
    List,
)

from rasa.shared.core.domain import Domain
from rasa.shared.core.trackers import DialogueStateTracker

logger = logging.getLogger(__name__)


@dataclass
class TurnParser:
    @abstractmethod
    def parse(
        self,
        tracker: DialogueStateTracker,
        domain: Domain,
    ) -> List[Turn]:
        ...


class Turn:
    @abstractmethod
    def get_type(self) -> str:
        ...

    def __repr__(self) -> Text:
        return f"{self.__class__.__name__}({self.get_type()})"

    @staticmethod
    def get_index_of_last(turns: List[Turn], turn_type: str) -> Optional[int]:
        """Returns the index of the last turn attributed to the given party.

        Returns:
            the index of the last user turn, or None if there is no such turn
        """
        return next(
            (
                idx
                for idx in range(len(turns) - 1, -1, -1)
                if turns[idx].get_type() == turn_type
            ),
            None,
        )
