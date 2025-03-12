from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import structlog

from rasa.dialogue_understanding.commands.command import Command
from rasa.dialogue_understanding.patterns.cannot_handle import (
    CannotHandlePatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.handle_digressions import (
    HandleDigressionsPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.utils import (
    top_flow_frame,
)
from rasa.shared.core.events import Event
from rasa.shared.core.flows import FlowsList
from rasa.shared.core.flows.steps import CollectInformationFlowStep
from rasa.shared.core.flows.utils import ALL_LABEL
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.nlu.constants import HANDLE_DIGRESSIONS_COMMAND

structlogger = structlog.get_logger()


@dataclass
class HandleDigressionsCommand(Command):
    """A command to handle digressions during an active flow."""

    flow: str
    """The interrupting flow."""

    @classmethod
    def command(cls) -> str:
        """Returns the command type."""
        return HANDLE_DIGRESSIONS_COMMAND

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> HandleDigressionsCommand:
        """Converts the dictionary to a command.

        Returns:
            The converted dictionary.
        """
        try:
            return HandleDigressionsCommand(flow=data["flow"])
        except KeyError as e:
            raise ValueError(
                f"Missing parameter '{e}' while parsing HandleDigressionsCommand."
            ) from e

    def run_command_on_tracker(
        self,
        tracker: DialogueStateTracker,
        all_flows: FlowsList,
        original_tracker: DialogueStateTracker,
    ) -> List[Event]:
        """Runs the command on the tracker.

        Args:
            tracker: The tracker to run the command on.
            all_flows: All flows in the assistant.
            original_tracker: The tracker before any command was executed.

        Returns:
            The events to apply to the tracker.
        """
        stack = tracker.stack
        original_stack = original_tracker.stack

        if self.flow not in all_flows.flow_ids:
            structlogger.debug(
                "command_executor.push_cannot_handle.start_invalid_flow_id",
                command=self,
            )
            stack.push(CannotHandlePatternFlowStackFrame())
            return tracker.create_stack_updated_events(stack)

        # this allows to include called user flows in the stack search
        latest_user_frame = top_flow_frame(original_stack, ignore_call_frames=False)

        if latest_user_frame is None:
            structlogger.debug(
                "command_executor.skip_command.no_top_flow", command=self
            )
            return []

        original_top_flow = latest_user_frame.flow(all_flows)
        current_step = original_top_flow.step_by_id(latest_user_frame.step_id)
        if not isinstance(current_step, CollectInformationFlowStep):
            structlogger.debug(
                "command_executor.skip_command.not_at_a_collect_step", command=self
            )
            return []

        ask_confirm_digressions = set(
            current_step.ask_confirm_digressions
            + original_top_flow.ask_confirm_digressions
        )

        block_digressions = set(
            current_step.block_digressions + original_top_flow.block_digressions
        )

        if block_digressions:
            if ALL_LABEL in block_digressions:
                block_digressions.remove(ALL_LABEL)
                block_digressions.add(self.flow)

        if ask_confirm_digressions:
            if ALL_LABEL in ask_confirm_digressions:
                ask_confirm_digressions.remove(ALL_LABEL)
                ask_confirm_digressions.add(self.flow)

        structlogger.debug(
            "command_executor.push_handle_digressions",
            interrupting_flow_id=self.flow,
            interrupted_flow_id=original_top_flow.id,
            interrupted_step_id=current_step.id,
            ask_confirm_digressions=ask_confirm_digressions,
            block_digressions=block_digressions,
        )
        stack.push(
            HandleDigressionsPatternFlowStackFrame(
                interrupting_flow_id=self.flow,
                interrupted_flow_id=original_top_flow.id,
                interrupted_step_id=current_step.id,
                ask_confirm_digressions=ask_confirm_digressions,
                block_digressions=block_digressions,
            )
        )

        return tracker.create_stack_updated_events(stack)

    def __hash__(self) -> int:
        return hash(self.flow)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, HandleDigressionsCommand):
            return False

        return other.flow == self.flow
