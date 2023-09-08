from __future__ import annotations

from dataclasses import dataclass, field
import dataclasses
from enum import Enum
from typing import Any, Dict, List, Tuple

import structlog

import rasa.shared.utils.common
from rasa.shared.utils.io import random_string


structlogger = structlog.get_logger()


def generate_stack_frame_id() -> str:
    """Generates a stack frame ID.

    Returns:
        The generated stack frame ID.
    """
    return random_string(8)


@dataclass
class DialogueStackFrame:
    """Represents the current flow step."""

    frame_id: str = field(default_factory=generate_stack_frame_id)
    """The ID of the current frame."""

    def as_dict(self) -> Dict[str, Any]:
        """Returns the `DialogueStackFrame` as a dictionary.

        Returns:
            The `DialogueStackFrame` as a dictionary.
        """

        def custom_asdict_factory(data: List[Tuple[str, Any]]) -> Dict[str, Any]:
            """Converts enum values to their value."""
            return {
                field: value.value if isinstance(value, Enum) else value
                for field, value in data
            }

        dump = dataclasses.asdict(self, dict_factory=custom_asdict_factory)
        dump["type"] = self.type()
        return dump

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        raise NotImplementedError

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> DialogueStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        raise NotImplementedError

    def context_as_dict(
        self, underlying_frames: List[DialogueStackFrame]
    ) -> Dict[str, Any]:
        """Returns the context of the frame."""
        return self.as_dict()

    @staticmethod
    def create_typed_frame(data: Dict[str, Any]) -> DialogueStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        for cls in rasa.shared.utils.common.all_subclasses(DialogueStackFrame):
            try:
                if data.get("type") == cls.type():
                    return cls.from_dict(data)
            except NotImplementedError:
                # we don't want to raise an error if the frame type is not
                # implemented, as this is ok to be raised by an abstract class
                pass
        else:
            structlogger.warning("dialogue_stack.frame.unknown_type", data=data)
            raise NotImplementedError
