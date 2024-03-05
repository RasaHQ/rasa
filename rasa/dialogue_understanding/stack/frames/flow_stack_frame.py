from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from rasa.dialogue_understanding.stack.frames import DialogueStackFrame
from rasa.shared.core.flows.steps.constants import START_STEP
from rasa.shared.core.flows import Flow, FlowsList, FlowStep
from rasa.shared.exceptions import RasaException


class InvalidFlowStackFrameType(RasaException):
    """Raised if the stack frame type is invalid."""

    def __init__(self, frame_type: Optional[str]) -> None:
        """Creates a `InvalidFlowStackFrameType`.

        Args:
            frame_type: The invalid stack frame type.
        """
        super().__init__(f"Invalid stack frame type '{frame_type}'.")


class FlowStackFrameType(str, Enum):
    INTERRUPT = "interrupt"
    """The frame is an interrupt frame.

    This means that the previous flow was interrupted by this flow. An
    interrupt should be used for frames that span multiple turns and
    where we expect the user needing help to get back to the previous
    flow."""
    LINK = "link"
    """The frame is a link frame.

    This means that the previous flow linked to this flow."""

    CALL = "call"
    """The frame is a flow that is called from another flow."""

    REGULAR = "regular"
    """The frame is a regular frame.

    In all other cases, this is the case."""

    @staticmethod
    def from_str(typ: Optional[str]) -> "FlowStackFrameType":
        """Creates a `FlowStackFrameType` from a string.

        Args:
            typ: The string to create the `FlowStackFrameType` from.

        Returns:
            The created `FlowStackFrameType`."""
        if typ is None:
            return FlowStackFrameType.REGULAR
        elif typ == FlowStackFrameType.INTERRUPT.value:
            return FlowStackFrameType.INTERRUPT
        elif typ == FlowStackFrameType.CALL.value:
            return FlowStackFrameType.CALL
        elif typ == FlowStackFrameType.LINK.value:
            return FlowStackFrameType.LINK
        elif typ == FlowStackFrameType.REGULAR.value:
            return FlowStackFrameType.REGULAR
        else:
            raise InvalidFlowStackFrameType(typ)


class InvalidFlowIdException(Exception):
    """Raised if the flow ID is invalid."""

    def __init__(self, flow_id: str) -> None:
        """Creates a `InvalidFlowIdException`.

        Args:
            flow_id: The invalid flow ID.
        """
        super().__init__(f"Invalid flow ID '{flow_id}'.")


class InvalidFlowStepIdException(Exception):
    """Raised if the flow step ID is invalid."""

    def __init__(self, flow_id: str, step_id: str) -> None:
        """Creates a `InvalidFlowStepIdException`.

        Args:
            flow_id: The invalid flow ID.
            step_id: The invalid flow step ID.
        """
        super().__init__(f"Invalid flow step ID '{step_id}' for flow '{flow_id}'.")


@dataclass
class BaseFlowStackFrame(DialogueStackFrame):
    flow_id: str = ""  # needed to avoid "default arg before non-default" error
    """The ID of the current flow."""
    step_id: str = START_STEP
    """The ID of the current step."""

    def flow(self, all_flows: FlowsList) -> Flow:
        """Returns the current flow.

        Args:
            all_flows: All flows in the assistant.

        Returns:
            The current flow."""
        flow = all_flows.flow_by_id(self.flow_id)
        if not flow:
            # we shouldn't ever end up with a frame that belongs to a non
            # existing flow, but if we do, we should raise an error
            raise InvalidFlowIdException(self.flow_id)
        return flow

    def step(self, all_flows: FlowsList) -> FlowStep:
        """Returns the current flow step.

        Args:
            all_flows: All flows in the assistant.

        Returns:
            The current flow step."""
        flow = self.flow(all_flows)
        step = flow.step_by_id(self.step_id)
        if not step:
            # we shouldn't ever end up with a frame that belongs to a non
            # existing step, but if we do, we should raise an error
            raise InvalidFlowStepIdException(self.flow_id, self.step_id)
        return step


@dataclass
class UserFlowStackFrame(BaseFlowStackFrame):
    frame_type: FlowStackFrameType = FlowStackFrameType.REGULAR
    """The type of the frame. Defaults to `StackFrameType.REGULAR`."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return "flow"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> UserFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return UserFlowStackFrame(
            frame_id=data["frame_id"],
            flow_id=data["flow_id"],
            step_id=data["step_id"],
            frame_type=FlowStackFrameType.from_str(data.get("frame_type")),
        )
