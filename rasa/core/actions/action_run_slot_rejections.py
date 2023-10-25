from typing import Any, Dict, List, Optional, TYPE_CHECKING, Text, Union

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
from rasa.shared.core.slots import (
    BooleanSlot,
    CategoricalSlot,
    FloatSlot,
    bool_from_any,
)

if TYPE_CHECKING:
    from rasa.core.nlg import NaturalLanguageGenerator
    from rasa.core.channels.channel import OutputChannel
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()


def is_none_value(value: str) -> bool:
    return value in {
        "[missing information]",
        "[missing]",
        "None",
        "undefined",
        "null",
    }


def coerce_slot_value(
    slot_value: str, slot_name: str, tracker: "DialogueStateTracker"
) -> Union[str, bool, float, None]:
    """Coerce the slot value to the correct type.

    Tries to coerce the slot value to the correct type. If the
    conversion fails, `None` is returned.

    Args:
        value: the value to coerce
        slot_name: the name of the slot
        tracker: the tracker

    Returns:
        the coerced value or `None` if the conversion failed."""
    nullable_value = slot_value if not is_none_value(slot_value) else None
    if slot_name not in tracker.slots:
        return nullable_value

    slot = tracker.slots[slot_name]
    if isinstance(slot, BooleanSlot):
        try:
            return bool_from_any(nullable_value)
        except (ValueError, TypeError):
            return None
    elif isinstance(slot, FloatSlot):
        try:
            return float(nullable_value)
        except (ValueError, TypeError):
            return None
    else:
        return nullable_value


def run_rejections(slot_value, slot_name, top_frame, dialogue_stack):
    """Run the predicate checks under rejections."""
    violation = False
    internal_error = False
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
        return None, None
    if internal_error:
        utterance = "utter_internal_error_rasa"
    # reset slot value that was initially filled with an invalid value
    return SlotSet(top_frame.collect, None), utterance


def run_categorical_slot_validation(slot_value, slot_name, slot_instance):
    """Run categorical slot validation."""
    if (
        isinstance(slot_instance, CategoricalSlot)
        and slot_value not in slot_instance.values
    ):
        # only fill categorical slots with values that are present in the domain
        structlogger.debug(
            "run.rejection.categorical_slot_value_not_in_domain",
            rejection=slot_value,
        )
        return SlotSet(slot_name, None), "utter_categorical_slot_rejection"
    return None, None


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

        dialogue_stack = DialogueStack.from_tracker(tracker)
        top_frame = dialogue_stack.top()
        if not isinstance(top_frame, CollectInformationPatternFlowStackFrame):
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
        typed_slot_value = coerce_slot_value(slot_value, slot_name, tracker)
        if top_frame.rejections:
            event, utterance = run_rejections(
                typed_slot_value, slot_name, top_frame, dialogue_stack
            )
        else:
            event, utterance = run_categorical_slot_validation(
                typed_slot_value, slot_name, slot_instance
            )
        events: List[Event] = [event] if event else []

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
