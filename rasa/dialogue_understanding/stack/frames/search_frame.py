from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from rasa.dialogue_understanding.stack.frames import DialogueStackFrame


@dataclass
class SearchStackFrame(DialogueStackFrame):
    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return "search"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> SearchStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return SearchStackFrame(
            frame_id=data["frame_id"],
        )
