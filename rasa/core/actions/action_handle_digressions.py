from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog

from rasa.core.actions.action import Action, create_bot_utterance
from rasa.core.channels import OutputChannel
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.core.utils import add_bot_utterance_metadata
from rasa.dialogue_understanding.patterns.continue_interrupted import (
    ContinueInterruptedPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.handle_digressions import (
    HandleDigressionsPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.frames.flow_stack_frame import (
    FlowStackFrameType,
    UserFlowStackFrame,
)
from rasa.shared.core.constants import (
    ACTION_BLOCK_DIGRESSION,
    ACTION_CONTINUE_DIGRESSION,
)
from rasa.shared.core.domain import Domain
from rasa.shared.core.events import Event, FlowInterrupted
from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()


class ActionBlockDigressions(Action):
    """Action which blocks an interruption and continues the current flow."""

    def name(self) -> str:
        """Return the action name."""
        return ACTION_BLOCK_DIGRESSION

    async def run(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Event]:
        """Update the stack."""
        structlogger.debug("action_block_digressions.run")
        top_frame = tracker.stack.top()

        if not isinstance(top_frame, HandleDigressionsPatternFlowStackFrame):
            return []

        blocked_flow_id = top_frame.interrupting_flow_id
        frame_type = FlowStackFrameType.REGULAR

        stack = tracker.stack
        stack.push(
            UserFlowStackFrame(flow_id=blocked_flow_id, frame_type=frame_type), 0
        )
        stack.push(
            ContinueInterruptedPatternFlowStackFrame(
                previous_flow_name=blocked_flow_id
            ),
            1,
        )
        events = tracker.create_stack_updated_events(stack)

        utterance = "utter_block_digressions"
        message = await nlg.generate(
            utterance,
            tracker,
            output_channel.name(),
        )

        if message is None:
            structlogger.error(
                "action_block_digressions.run.failed.finding.utter",
                utterance=utterance,
            )
        else:
            message = add_bot_utterance_metadata(
                message, utterance, nlg, domain, tracker
            )
            events.append(create_bot_utterance(message))

        return events


class ActionContinueDigression(Action):
    """Action which continues with an interruption."""

    def name(self) -> str:
        """Return the action name."""
        return ACTION_CONTINUE_DIGRESSION

    async def run(
        self,
        output_channel: OutputChannel,
        nlg: NaturalLanguageGenerator,
        tracker: DialogueStateTracker,
        domain: Domain,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Event]:
        """Update the stack."""
        structlogger.debug("action_continue_digression.run")
        top_frame = tracker.stack.top()

        if not isinstance(top_frame, HandleDigressionsPatternFlowStackFrame):
            return []

        blocked_flow_id = top_frame.interrupting_flow_id
        frame_type = FlowStackFrameType.INTERRUPT
        stack = tracker.stack
        stack.push(UserFlowStackFrame(flow_id=blocked_flow_id, frame_type=frame_type))

        events = [
            FlowInterrupted(
                flow_id=top_frame.interrupted_flow_id,
                step_id=top_frame.interrupted_step_id,
            )
        ] + tracker.create_stack_updated_events(stack)

        utterance = "utter_continue_interruption"
        message = await nlg.generate(
            utterance,
            tracker,
            output_channel.name(),
        )

        if message is None:
            structlogger.error(
                "action_continue_digression.run.failed.finding.utter",
                utterance=utterance,
            )
        else:
            message = add_bot_utterance_metadata(
                message, utterance, nlg, domain, tracker
            )
            events.append(create_bot_utterance(message))

        return events
