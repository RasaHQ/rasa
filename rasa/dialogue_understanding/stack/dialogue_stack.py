from __future__ import annotations
import copy

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import structlog
import typing
import jsonpatch

if typing.TYPE_CHECKING:
    from rasa.dialogue_understanding.stack.frames import DialogueStackFrame

structlogger = structlog.get_logger()


@dataclass
class DialogueStack:
    """Represents the current dialogue stack."""

    frames: List["DialogueStackFrame"]

    @staticmethod
    def from_dict(data: List[Dict[str, Any]]) -> DialogueStack:
        """Creates a `DialogueStack` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStack` from.

        Returns:
            The created `DialogueStack`.
        """
        from rasa.dialogue_understanding.stack.frames import DialogueStackFrame

        return DialogueStack(
            frames=[DialogueStackFrame.create_typed_frame(frame) for frame in data]
        )

    @staticmethod
    def empty() -> DialogueStack:
        """Creates an empty `DialogueStack`.

        Returns:
            The created empty `DialogueStack`.
        """
        return DialogueStack(frames=[])

    def as_dict(self) -> List[Dict[str, Any]]:
        """Returns the `DialogueStack` as a dictionary.

        Returns:
            The `DialogueStack` as a dictionary.
        """
        return [frame.as_dict() for frame in self.frames]

    def copy(self) -> DialogueStack:
        return copy.deepcopy(self)

    def push(self, frame: "DialogueStackFrame", index: Optional[int] = None) -> None:
        """Pushes a new frame onto the stack.

        If the frame shouldn't be put on top of the stack, the index can be
        specified. Not specifying an index equals `push(frame, index=len(frames))`.

        Args:
            frame: The frame to push onto the stack.
            index: The index to insert the frame at. If `None`, the frame
                is put on top of the stack.
        """
        if index is None:
            self.frames.append(frame)
        else:
            self.frames.insert(index, frame)

    def update(self, frame: "DialogueStackFrame") -> None:
        """Updates the topmost frame.

        Args:
            frame: The frame to update.
        """
        if not self.is_empty():
            self.pop()

        self.push(frame)

    def pop(self) -> "DialogueStackFrame":
        """Pops the topmost frame from the stack.

        Returns:
            The popped frame.
        """
        return self.frames.pop()

    def current_context(self) -> Dict[str, Any]:
        """Returns the context of the topmost frame.

        Returns:
            The context of the topmost frame.
        """
        if self.is_empty():
            return {}

        return self.frames[-1].context_as_dict(self.frames[:-1])

    def top(
        self,
        ignore: Optional[Callable[["DialogueStackFrame"], bool]] = None,
    ) -> Optional["DialogueStackFrame"]:
        """Returns the topmost frame from the stack.

        Args:
            ignore_frame: The ID of the flow to ignore. Picks the top most
                frame that has a different flow ID.

        Returns:
            The topmost frame.
        """
        for frame in reversed(self.frames):
            if ignore and ignore(frame):
                continue
            return frame
        return None

    def is_empty(self) -> bool:
        """Checks if the stack is empty.

        Returns:
            `True` if the stack is empty, `False` otherwise.
        """
        return len(self.frames) == 0

    def update_from_patch(self, patch_dump: str) -> DialogueStack:
        """Updates the stack from a patch.

        Args:
            patch_dump: The patch to apply to the stack.

        Returns:
            The updated stack."""
        patch = jsonpatch.JsonPatch.from_string(patch_dump)
        dialogue_stack_dump = patch.apply(self.as_dict())
        return DialogueStack.from_dict(dialogue_stack_dump)

    def create_stack_patch(self, updated_stack: DialogueStack) -> Optional[str]:
        """Creates a patch to update the stack to the updated stack state.

        Example:
            > stack = DialogueStack.from_dict([
            >     {
            >         "type": "flow",
            >         "frame_type": "regular",
            >         "flow_id": "foo",
            >         "step_id": "START",
            >         "frame_id": "test",
            >     }
            > ])
            > updated_stack = DialogueStack.from_dict([
            >     {
            >         "type": "flow",
            >         "frame_type": "regular",
            >         "flow_id": "foo",
            >         "step_id": "1",
            >         "frame_id": "test",
            >     }
            > ])
            > stack.create_stack_patch(updated_stack)
            '[{"op": "replace", "path": "/0/step_id", "value": "1"}]'

        Args:
            updated_stack: The updated stack.

        Returns:
            The patch to update the stack to the updated stack state.
        """
        patch = jsonpatch.JsonPatch.from_diff(self.as_dict(), updated_stack.as_dict())

        if patch:
            return patch.to_string()
        return None
