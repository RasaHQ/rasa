from __future__ import annotations

import copy
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import jsonpatch
import structlog

from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    AgentState,
    FlowStackFrameType,
    UserFlowStackFrame,
)

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

    def move_frames_to_top(self, frames_to_move: List["DialogueStackFrame"]) -> None:
        """Moves specified frames to top of stack while preserving their relative order.

        Args:
            frames_to_move: The frames to move to the top of the stack.
        """
        # Get frames that are not being moved
        frames_to_keep = [frame for frame in self.frames if frame not in frames_to_move]

        # Reorder: keep frames first, then moved frames
        self.frames = frames_to_keep + frames_to_move

        # set all frames to interrupt except for LINK and CALL
        for frame in self.frames:
            if (
                isinstance(frame, UserFlowStackFrame)
                and frame.frame_type == FlowStackFrameType.LINK
            ):
                continue
            if (
                isinstance(frame, UserFlowStackFrame)
                and frame.frame_type == FlowStackFrameType.CALL
            ):
                continue
            if (
                isinstance(frame, UserFlowStackFrame)
                and frame.frame_type == FlowStackFrameType.REGULAR
            ):
                frame.frame_type = FlowStackFrameType.INTERRUPT

        # set the first frame to regular
        for frame in self.frames:
            if isinstance(frame, UserFlowStackFrame):
                frame.frame_type = FlowStackFrameType.REGULAR
                return

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
            ignore: The ID of the flow to ignore. Picks the top most
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
        The updated stack.
        """
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

    def _find_agent_frame_by_predicate(
        self, predicate: Callable[[AgentStackFrame], bool]
    ) -> List[AgentStackFrame]:
        stack_frames: List[AgentStackFrame] = []
        for stack_frame in reversed(self.frames):
            if isinstance(stack_frame, AgentStackFrame) and predicate(stack_frame):
                stack_frames.append(stack_frame)
        return stack_frames

    def find_active_agent_frame(self) -> Optional[AgentStackFrame]:
        stack_frames = self._find_agent_frame_by_predicate(
            lambda frame: frame.state == AgentState.WAITING_FOR_INPUT
        )
        if stack_frames:
            return stack_frames[0]
        return None

    def find_agent_stack_frame_by_agent(
        self, agent_id: str
    ) -> Optional[AgentStackFrame]:
        """Get the agent stack frame for a specific agent ID.

        May also include the agent stack frame in the INTERRUPTED state.
        """
        stack_frames = self._find_agent_frame_by_predicate(
            lambda frame: frame.agent_id == agent_id
        )
        if stack_frames:
            return stack_frames[0]
        return None

    def find_active_agent_stack_frame_for_flow(
        self, flow_id: str
    ) -> Optional[AgentStackFrame]:
        """Get the agent stack frame of a specific flow."""
        stack_frames = self._find_agent_frame_by_predicate(
            lambda frame: frame.flow_id == flow_id
        )
        for stack_frame in stack_frames:
            if stack_frame.state == AgentState.WAITING_FOR_INPUT:
                return stack_frame
        return None

    def get_active_agent_id(self) -> Optional[typing.Text]:
        agent_frame = self.find_active_agent_frame()
        if agent_frame:
            return agent_frame.agent_id
        return None

    def agent_is_active(self) -> bool:
        return self.find_active_agent_frame() is not None
