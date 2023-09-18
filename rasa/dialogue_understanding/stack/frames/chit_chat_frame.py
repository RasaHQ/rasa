from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
from rasa.dialogue_understanding.stack.frames import DialogueStackFrame


@dataclass
class ChitChatStackFrame(DialogueStackFrame):
    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return "chitchat"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ChitChatStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return ChitChatStackFrame(
            frame_id=data["frame_id"],
        )
