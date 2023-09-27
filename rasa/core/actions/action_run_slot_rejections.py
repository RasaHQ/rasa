from typing import Any, Dict, List, Optional, TYPE_CHECKING, Text

import structlog
from jinja2 import Template
from pypred import Predicate

from rasa.core.actions.action import Action, create_bot_utterance
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.core.constants import ACTION_RUN_SLOT_REJECTIONS_NAME
from rasa.shared.core.events import Event, SlotSet

if TYPE_CHECKING:
    from rasa.core.nlg import NaturalLanguageGenerator
    from rasa.core.channels.channel import OutputChannel
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()


class ActionRunSlotRejections(Action):
    """Action which evaluates the predicate checks under rejections."""

    def name(self) -> Text:
        """Return the name of the action."""
        return ACTION_RUN_SLOT_REJECTIONS_NAME

    async def run(
        self,
        output_channel: "OutputChannel",
        nlg: "NaturalLanguageGenerator",
        tracker: "DialogueStateTracker",
        domain: "Domain",
        metadata: Optional[Dict[Text, Any]] = None,
    ) -> List[Event]:
        """Run the predicate checks."""
        events: List[Event] = []
        violation = False
        utterance = None
        internal_error = False

        dialogue_stack = DialogueStack.from_tracker(tracker)
        top_frame = dialogue_stack.top()
        if not isinstance(top_frame, CollectInformationPatternFlowStackFrame):
            return []

        if not top_frame.rejections:
            return []

        slot_name = top_frame.collect
        slot_instance = tracker.slots.get(slot_name)
        if slot_instance and not slot_instance.has_been_set:
            # this is the first time the assistant asks for the slot value,
            # therefore we skip the predicate validation because the slot
            # value has not been provided
            structlogger.debug(
                "first.collect.slot.not.set",
                slot_name=slot_name,
                slot_value=slot_instance.value,
            )
            return []

        slot_value = tracker.get_slot(slot_name)

        current_context = dialogue_stack.current_context()
        current_context[slot_name] = slot_value

        structlogger.debug("run.predicate.context", context=current_context)
        document = current_context.copy()

        for rejection in top_frame.rejections:
            condition = rejection.if_
            utterance = rejection.utter

            try:
                rendered_template = Template(condition).render(current_context)
                predicate = Predicate(rendered_template)
                violation = predicate.evaluate(document)
                structlogger.debug(
                    "run.predicate.result",
                    predicate=predicate.description(),
                    violation=violation,
                )
            except (TypeError, Exception) as e:
                structlogger.error(
                    "run.predicate.error",
                    predicate=condition,
                    document=document,
                    error=str(e),
                )
                violation = True
                internal_error = True

            if violation:
                break

        if not violation:
            return []

        # reset slot value that was initially filled with an invalid value
        events.append(SlotSet(top_frame.collect, None))

        if internal_error:
            utterance = "utter_internal_error_rasa"

        if not isinstance(utterance, str):
            structlogger.error(
                "run.rejection.missing.utter",
                utterance=utterance,
            )
            return events

        message = await nlg.generate(
            utterance,
            tracker,
            output_channel.name(),
        )

        if message is None:
            structlogger.error(
                "run.rejection.failed.finding.utter",
                utterance=utterance,
            )
        else:
            message["utter_action"] = utterance
            events.append(create_bot_utterance(message))

        return events
