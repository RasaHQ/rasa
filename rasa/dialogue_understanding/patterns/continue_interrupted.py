from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Text, Tuple

import structlog

from rasa.core.actions.action import Action
from rasa.core.channels import OutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.dialogue_understanding.commands.utils import (
    resume_flow,
)
from rasa.dialogue_understanding.patterns.cancel import CancelPatternFlowStackFrame
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.dialogue_understanding.stack.frames import PatternFlowStackFrame
from rasa.dialogue_understanding.stack.frames.dialogue_stack_frame import (
    DialogueStackFrame,
)
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    AgentStackFrame,
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.dialogue_understanding.stack.utils import (
    get_active_pattern_frame,
)
from rasa.shared.constants import RASA_DEFAULT_FLOW_PATTERN_PREFIX
from rasa.shared.core.constants import (
    ACTION_CANCEL_INTERRUPTED_FLOWS,
    ACTION_CONTINUE_INTERRUPTED_FLOW,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import AgentCancelled, Event, FlowCancelled, SlotSet
from rasa.shared.core.trackers import DialogueStateTracker

FLOW_PATTERN_CONTINUE_INTERRUPTED = (
    RASA_DEFAULT_FLOW_PATTERN_PREFIX + "continue_interrupted"
)
INTERRUPTED_FLOW_TO_CONTINUE_SLOT = "interrupted_flow_to_continue"
CONTINUE_INTERRUPTED_FLOW_CONFIRMATION_SLOT = "continue_interrupted_flow_confirmation"


structlogger = structlog.get_logger()


@dataclass
class ContinueInterruptedPatternFlowStackFrame(PatternFlowStackFrame):
    """A pattern flow stack frame that gets added if an interruption is completed."""

    flow_id: str = FLOW_PATTERN_CONTINUE_INTERRUPTED
    """The ID of the flow."""
    interrupted_flow_names: List[str] = field(default_factory=list)
    """The names of the previous flows that were interrupted."""
    interrupted_flow_ids: List[str] = field(default_factory=list)
    """The ids of the previous flows that were interrupted."""
    interrupted_flow_options: str = ""
    """The options that the user can choose from as a string."""
    multiple_flows_interrupted: bool = False
    """Whether the user has interrupted multiple flows."""

    @classmethod
    def type(cls) -> str:
        """Returns the type of the frame."""
        return FLOW_PATTERN_CONTINUE_INTERRUPTED

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> ContinueInterruptedPatternFlowStackFrame:
        """Creates a `DialogueStackFrame` from a dictionary.

        Args:
            data: The dictionary to create the `DialogueStackFrame` from.

        Returns:
            The created `DialogueStackFrame`.
        """
        return ContinueInterruptedPatternFlowStackFrame(
            frame_id=data["frame_id"],
            step_id=data["step_id"],
            interrupted_flow_names=data["interrupted_flow_names"],
            interrupted_flow_ids=data["interrupted_flow_ids"],
            interrupted_flow_options=data["interrupted_flow_options"],
            multiple_flows_interrupted=len(data["interrupted_flow_names"]) > 1,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ContinueInterruptedPatternFlowStackFrame):
            return False
        return (
            self.flow_id == other.flow_id
            and self.step_id == other.step_id
            and self.interrupted_flow_names == other.interrupted_flow_names
            and self.interrupted_flow_ids == other.interrupted_flow_ids
            and self.interrupted_flow_options == other.interrupted_flow_options
        )


class ActionContinueInterruptedFlow(Action):
    def name(self) -> str:
        return ACTION_CONTINUE_INTERRUPTED_FLOW

    async def run(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        # get the pattern frame from the stack
        pattern_frame = get_active_pattern_frame(
            tracker.stack, ContinueInterruptedPatternFlowStackFrame
        )

        if pattern_frame is None or not isinstance(
            pattern_frame, ContinueInterruptedPatternFlowStackFrame
        ):
            structlogger.warning(
                "action.continue_interrupted_flows.no_continue_interrupted_frame"
            )
            return []

        interrupted_flow_ids = pattern_frame.interrupted_flow_ids
        interrupted_flow_names = pattern_frame.interrupted_flow_names
        multiple_flows_interrupted = pattern_frame.multiple_flows_interrupted

        flow_to_continue = None
        if not multiple_flows_interrupted:
            # the user confirmed that they want to continue the flow
            # as only one flow was interrupted, we can just continue the first one
            flow_to_continue = interrupted_flow_ids[0]
        else:
            # the user mentioned the flow they want to continue
            # check if the flow is in the list of interrupted flows
            selected_flow = tracker.get_slot(INTERRUPTED_FLOW_TO_CONTINUE_SLOT)
            if selected_flow in interrupted_flow_ids:
                flow_to_continue = selected_flow
            elif selected_flow in interrupted_flow_names:
                # the user mentioned the flow by name
                # find the flow id for the flow name
                # the list of names and ids are in the same order
                flow_to_continue = interrupted_flow_ids[
                    interrupted_flow_names.index(selected_flow)
                ]

        # if the user did not select a valid flow,
        # we need to ask them to select a valid flow
        if flow_to_continue is None:
            await output_channel.send_text_message(
                tracker.sender_id,
                "You haven't selected a valid task to resume. "
                "Please specify the task you would like to continue. "
                "The options are: {{context.interrupted_flow_options}}",
            )
            return []

        # resume the flow the user selected
        events = resume_flow(flow_to_continue, tracker, tracker.stack)

        return events + [
            SlotSet(INTERRUPTED_FLOW_TO_CONTINUE_SLOT, None),
            SlotSet(CONTINUE_INTERRUPTED_FLOW_CONFIRMATION_SLOT, None),
        ]


class ActionCancelInterruptedFlows(Action):
    def name(self) -> str:
        return ACTION_CANCEL_INTERRUPTED_FLOWS

    async def run(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        # get the pattern frame from the stack
        pattern_frame = get_active_pattern_frame(
            tracker.stack, ContinueInterruptedPatternFlowStackFrame
        )

        if pattern_frame is None or not isinstance(
            pattern_frame, ContinueInterruptedPatternFlowStackFrame
        ):
            structlogger.warning(
                "action.continue_interrupted_flows.no_continue_interrupted_frame"
            )
            return []

        interrupted_flow_ids = pattern_frame.interrupted_flow_ids

        event_list: List[Event] = []

        # cancel all interrupted flows
        for flow_id in interrupted_flow_ids:
            event_list.extend(self.cancel_flow(tracker, tracker.stack, flow_id))

        return event_list + [
            SlotSet(INTERRUPTED_FLOW_TO_CONTINUE_SLOT, None),
            SlotSet(CONTINUE_INTERRUPTED_FLOW_CONFIRMATION_SLOT, None),
        ]

    def cancel_flow(
        self,
        tracker: DialogueStateTracker,
        stack: DialogueStack,
        flow_id: str,
    ) -> List[Event]:
        """Cancels a flow by flow id."""
        applied_events: List[Event] = []

        frames_to_cancel, user_frame_to_cancel = self._collect_frames_to_cancel(
            stack, flow_id
        )

        # if the flow is not on the stack, do nothing
        if user_frame_to_cancel is None:
            structlogger.error(
                "cancel_flow.no_user_frame_to_cancel",
                flow_id=flow_id,
            )
            return []

        frames_ids_to_cancel = [frame.frame_id for frame in frames_to_cancel]

        stack.push(
            CancelPatternFlowStackFrame(
                canceled_name=flow_id,
                canceled_frames=frames_ids_to_cancel,
            )
        )

        # create flow cancelled event
        applied_events.extend(
            [
                FlowCancelled(
                    user_frame_to_cancel.flow_id, user_frame_to_cancel.step_id
                ),
            ]
        )
        # create agent cancelled events for any agent frames that are on the stack
        for frame in frames_to_cancel:
            if isinstance(frame, AgentStackFrame):
                applied_events.append(
                    AgentCancelled(
                        frame.agent_id, frame.flow_id, reason="Flow was cancelled"
                    )
                )

        update_stack_events = tracker.create_stack_updated_events(stack)

        return applied_events + update_stack_events

    def _collect_frames_to_cancel(
        self, stack: DialogueStack, target_flow_id: str
    ) -> Tuple[List[DialogueStackFrame], Optional[UserFlowStackFrame]]:
        """Collect frames that need to be cancelled.

        Args:
            stack: The stack to collect frames from.
            target_flow_id: The ID of the flow to cancel.

        Returns:
            A tuple containing (frames_to_cancel, frame_to_cancel).
        """
        frames_to_cancel: List[DialogueStackFrame] = []
        frame_found = False
        frame_to_cancel = None

        # collect all frames that belong to the target flow
        # i.e. we want to cancel all frames that are on the stack and between
        # the user flow frame that belongs to the target flow and the next user
        # flow frame that belongs to a different flow
        # this includes any pattern frames or agent frames as well
        for frame in stack.frames:
            if isinstance(frame, UserFlowStackFrame) and (
                frame.frame_type == FlowStackFrameType.REGULAR
                or frame.frame_type == FlowStackFrameType.INTERRUPT
            ):
                if frame.flow_id == target_flow_id:
                    frames_to_cancel.append(frame)
                    frame_to_cancel = frame
                    frame_found = True
                    continue
                elif frame_found:
                    break

            if frame_found:
                frames_to_cancel.append(frame)

        return list(frames_to_cancel), frame_to_cancel
