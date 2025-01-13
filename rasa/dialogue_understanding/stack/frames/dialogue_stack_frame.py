from __future__ import annotations

import dataclasses
import inspect
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, Dict, List, Tuple, Type

import structlog

import rasa.shared.utils.common
from rasa.shared.exceptions import RasaException
from rasa.shared.utils.io import random_string

structlogger = structlog.get_logger()


def generate_stack_frame_id() -> str:
    """Generates a stack frame ID.

    Returns:
        The generated stack frame ID.
    """
    return random_string(8)


@lru_cache
def _get_all_subclasses() -> List[Type[DialogueStackFrame]]:
    stack_frame_subclasses = rasa.shared.utils.common.all_subclasses(DialogueStackFrame)

    # Get all the subclasses of DialogueStackFrame from the patterns package
    # in case these are not all imported at runtime
    modules = rasa.shared.utils.common.import_package_modules(
        "rasa.dialogue_understanding.patterns"
    )
    extra_subclasses = [
        clazz
        for module in modules
        for _, clazz in inspect.getmembers(module, inspect.isclass)
        if issubclass(clazz, DialogueStackFrame) and clazz not in stack_frame_subclasses
    ]

    stack_frame_subclasses.extend(extra_subclasses)
    return stack_frame_subclasses


class InvalidStackFrameType(RasaException):
    """Raised if a stack frame type is invalid."""

    def __init__(self, frame_type: str) -> None:
        """Creates a `InvalidStackFrameType`.

        Args:
            frame_type: The invalid frame type.
        """
        super().__init__(f"Invalid stack frame type '{frame_type}'.")


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

        def custom_asdict_factory(fields: List[Tuple[str, Any]]) -> Dict[str, Any]:
            """Converts enum values to their value."""

            def rename_internal(field_name: str) -> str:
                return field_name[:-1] if field_name.endswith("_") else field_name

            return {
                rename_internal(field): value.value
                if isinstance(value, Enum)
                else value
                for field, value in fields
            }

        data = dataclasses.asdict(self, dict_factory=custom_asdict_factory)
        data["type"] = self.type()
        return data

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
        typ = data.get("type")

        for clazz in _get_all_subclasses():
            try:
                if typ == clazz.type():
                    return clazz.from_dict(data)
            except NotImplementedError:
                # we don't want to raise an error if the frame type is not
                # implemented, as this is ok to be raised by an abstract class
                pass
        else:
            structlogger.warning("dialogue_stack.frame.unknown_type", data=data)
            raise InvalidStackFrameType(typ)
