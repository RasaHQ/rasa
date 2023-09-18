from typing import Any, Dict, List, Optional, TYPE_CHECKING, Text

import structlog
from jinja2 import Template
from pypred import Predicate

from rasa.core.actions.action import Action, create_bot_utterance
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.stack.dialogue_stack import DialogueStack
from rasa.shared.core.constants import ACTION_EVALUATE_PREDICATE_REJECTIONS_NAME
from rasa.shared.core.events import Event, SlotSet

if TYPE_CHECKING:
    from rasa.core.nlg import NaturalLanguageGenerator
    from rasa.core.channels.channel import OutputChannel
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()


class ActionEvaluatePredicateRejection(Action):
    """Action which evaluates the predicate checks under rejections."""

    def name(self) -> Text:
        """Return the name of the action."""
        return ACTION_EVALUATE_PREDICATE_REJECTIONS_NAME

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

        dialogue_stack = DialogueStack.from_tracker(tracker)
        top_frame = dialogue_stack.top()
        if not isinstance(top_frame, CollectInformationPatternFlowStackFrame):
            return []

        if top_frame.rejections is None:
            return []

        slot_name = top_frame.collect_information
        slot_value = tracker.get_slot(slot_name)

        current_context = dialogue_stack.current_context()
        current_context[slot_name] = slot_value

        structlogger.debug("collect.predicate.context", context=current_context)
        document = current_context.copy()

        for rejection in top_frame.rejections:
            check_text = rejection.get("if")
            utterance = rejection.get("utter")
            rendered_template = Template(check_text).render(current_context)
            predicate = Predicate(rendered_template)
            try:
                result = predicate.evaluate(document)
                structlogger.debug(
                    "collect.predicate.result",
                    result=result,
                )
            except (TypeError, Exception) as e:
                structlogger.error(
                    "collect.predicate.error",
                    predicate=predicate,
                    document=document,
                    error=str(e),
                )
                continue

            if result is False:
                continue

            if current_context.get("number_of_tries", 0) < 2:
                # reset slot value that was initially filled with an invalid value
                events.append(SlotSet(top_frame.collect_information, None))

            if utterance is None:
                structlogger.debug(
                    "collect.rejection.missing.utter",
                    predicate=predicate,
                    document=document,
                )
                break

            message = await nlg.generate(
                utterance,
                tracker,
                output_channel.name(),
            )
            if message is None:
                structlogger.debug(
                    "collect.rejection.failed.finding.utter",
                    utterance=utterance,
                )
            else:
                message["utter_action"] = utterance
                events.append(create_bot_utterance(message))
            return events

        return events
