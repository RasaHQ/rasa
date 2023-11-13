from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog
from rasa.core.actions import action
from rasa.core.channels.channel import OutputChannel
from rasa.core.nlg.generator import NaturalLanguageGenerator
from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from rasa.shared.core.constants import ACTION_CLARIFY_FLOWS
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event
from rasa.shared.core.trackers import DialogueStateTracker


structlogger = structlog.get_logger()

FLOW_PATTERN_CLARIFICATION = RASA_DEFAULT_FLOW_PATTERN_PREFIX + "clarification"


@dataclass
class ClarifyPatternFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame which helps the user clarify their action."""

    flow_id: str = FLOW_PATTERN_CLARIFICATION
    """The ID of the flow."""
    names: List[str] = field(default_factory=list)
    """The names of the flows that the user can choose from."""
    clarification_options: str = ""
    """The options that the user can choose from as a string."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_CLARIFICATION

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ClarifyPatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return ClarifyPatternFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
            names=data["names"],
            clarification_options=data["clarification_options"],
        )


class ActionClarifyFlows(action.Action):
    """Action which clarifies which flow to start."""

    def name(self) -> str:
        """Return the flow name."""
        return ACTION_CLARIFY_FLOWS

    @staticmethod
    def assemble_options_string(names: List[str]) -> str:
        """Concatenate options to a human-readable string."""
        clarification_message = ""
        for i, name in enumerate(names):
            if i == 0:
                clarification_message += name
            elif i == len(names) - 1:
                clarification_message += f" or {name}"
            else:
                clarification_message += f", {name}"
        return clarification_message

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Event]:
        """Correct the slots."""
        stack = tracker.stack
        if not (top := stack.top()):
            structlogger.warning("action.clarify_flows.no_active_flow")
            return []

        if not isinstance(top, ClarifyPatternFlowStackFrame):
            structlogger.warning("action.clarify_flows.no_clarification_frame", top=top)
            return []

        options_string = self.assemble_options_string(top.names)
        top.clarification_options = options_string
        # since we modified the stack frame, we need to update the stack
        return tracker.create_stack_updated_events(stack)
