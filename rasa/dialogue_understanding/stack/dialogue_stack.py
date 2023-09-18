from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
from rasa.dialogue_understanding.stack.frames import DialogueStackFrame
from rasa.shared.core.constants import (
    DIALOGUE_STACK_SLOT,
)
from rasa.shared.core.trackers import (
    DialogueStateTracker,
)
import structlog

structlogger = structlog.get_logger()


@dataclass
class DialogueStack:
    """Represents the current dialogue stack."""

    frames: List[DialogueStackFrame]

    @staticmethod
    def from_dict(data: List[Dict[str, Any]]) -> DialogueStack:
        """Creates a `DialogueStack` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStack` from.

        Returns:
            The created `DialogueStack`.
        """
        return DialogueStack(
            [DialogueStackFrame.create_typed_frame(frame) for frame in data]
        )

    def as_dict(self) -> List[Dict[str, Any]]:
        """Returns the `DialogueStack` as a dictionary.

        Returns:
            The `DialogueStack` as a dictionary.
        """
        return [frame.as_dict() for frame in self.frames]

    def push(self, frame: DialogueStackFrame, index: Optional[int] = None) -> None:
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

    def update(self, frame: DialogueStackFrame) -> None:
        """Updates the topmost frame.

        Args:
            frame: The frame to update.
        """
        if not self.is_empty():
            self.pop()

        self.push(frame)

    def pop(self) -> DialogueStackFrame:
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
        ignore: Optional[Callable[[DialogueStackFrame], bool]] = None,
    ) -> Optional[DialogueStackFrame]:
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

    @staticmethod
    def get_persisted_stack(tracker: DialogueStateTracker) -> List[Dict[str, Any]]:
        """Returns the persisted stack from the tracker.

        The stack is stored on a slot on the tracker. If the slot is not set,
        an empty list is returned.

        Args:
            tracker: The tracker to get the stack from.

        Returns:
            The persisted stack as a dictionary."""
        return tracker.get_slot(DIALOGUE_STACK_SLOT) or []

    @staticmethod
    def from_tracker(tracker: DialogueStateTracker) -> DialogueStack:
        """Creates a `DialogueStack` from a tracker.

        The stack is read from a slot on the tracker. If the slot is not set,
        an empty stack is returned.

        Args:
            tracker: The tracker to create the `DialogueStack` from.

        Returns:
            The created `DialogueStack`.
        """
        return DialogueStack.from_dict(DialogueStack.get_persisted_stack(tracker))
