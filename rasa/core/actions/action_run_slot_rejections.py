from typing import TYPE_CHECKING, Any, Dict, List, Optional, Text, Union

import structlog
from jinja2 import Template

from rasa.core.actions.action import Action, create_bot_utterance
from rasa.core.utils import add_bot_utterance_metadata
from rasa.dialogue_understanding.patterns.collect_information import (
    CollectInformationPatternFlowStackFrame,
)
from rasa.dialogue_understanding.patterns.validate_slot import (
    ValidateSlotPatternFlowStackFrame,
)
from rasa.shared.core.constants import ACTION_RUN_SLOT_REJECTIONS_NAME
from rasa.shared.core.events import Event, SlotSet
from rasa.shared.core.slots import (
    BooleanSlot,
    CategoricalSlot,
    FloatSlot,
    Slot,
    SlotRejection,
)
from rasa.utils.pypred import Predicate

if TYPE_CHECKING:
    from rasa.core.channels.channel import OutputChannel
    from rasa.core.nlg import NaturalLanguageGenerator
    from rasa.shared.core.domain import Domain
    from rasa.shared.core.trackers import DialogueStateTracker

structlogger = structlog.get_logger()

UTTERANCE_NOT_DEFINED_BY_USER = "TO.UPDATE.VALUE.BUT.MISSING.UTTERANCE"


def utterance_for_slot_type(slot: Slot) -> Optional[str]:
    """Return the utterance to use for the slot type."""
    if isinstance(slot, BooleanSlot):
        return "utter_boolean_slot_rejection"
    elif isinstance(slot, FloatSlot):
        return "utter_float_slot_rejection"
    elif isinstance(slot, CategoricalSlot):
        return "utter_categorical_slot_rejection"
    return None


def coerce_slot_value(
    slot_value: str, slot_name: str, tracker: "DialogueStateTracker"
) -> Union[str, bool, float, None]:
    """Coerce the slot value to the correct type.

    Tries to coerce the slot value to the correct type. If the
    conversion fails, `None` is returned.

    Args:
        slot_value: The value to coerce.
        slot_name: The name of the slot.
        tracker: The tracker containing the current state of the conversation.

    Returns:
        The coerced value or `None` if the conversion failed.
    """
    if slot_name not in tracker.slots:
        return slot_value

    slot = tracker.slots[slot_name]

    if not slot.is_valid_value(slot_value):
        structlogger.debug(
            "run.rejection.slot_value_not_valid",
            rejection=slot_value,
        )
        return None

    return slot.coerce_value(slot_value)


def run_rejections(
    slot_value: Union[str, bool, float, None],
    slot_name: str,
    rejections: List[SlotRejection],
) -> Optional[str]:
    """Run the predicate checks under rejections."""
    violation = False
    internal_error = False
    current_context = {"slots": {slot_name: slot_value}}

    structlogger.debug("run.predicate.context", context=current_context)
    document = current_context.copy()

    for rejection in rejections:
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
                error=str(e),
            )
            violation = True
            internal_error = True

        if violation:
            break

    if not violation:
        return None
    if internal_error:
        utterance = "utter_internal_error_rasa"
    if not isinstance(utterance, str):
        utterance = UTTERANCE_NOT_DEFINED_BY_USER
    return utterance


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
        utterance = None
        top_frame = tracker.stack.top()
        if not isinstance(
            top_frame,
            (
                CollectInformationPatternFlowStackFrame,
                ValidateSlotPatternFlowStackFrame,
            ),
        ):
            return []
        elif isinstance(top_frame, CollectInformationPatternFlowStackFrame):
            slot_name = top_frame.collect
        elif isinstance(top_frame, ValidateSlotPatternFlowStackFrame):
            slot_name = top_frame.validate

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

        if slot_instance and slot_instance.has_been_set and slot_value is None:
            return []

        typed_slot_value = coerce_slot_value(slot_value, slot_name, tracker)
        if typed_slot_value is None:
            # the slot value could not be coerced to the correct type
            utterance = utterance_for_slot_type(slot_instance)
        elif top_frame.rejections:
            # run the predicate checks under rejections
            utterance = run_rejections(
                typed_slot_value, slot_name, top_frame.rejections
            )

        events: List[Event] = []
        if utterance:
            # the slot value has been rejected
            events.append(SlotSet(slot_name, None))
        elif slot_value != typed_slot_value or type(slot_value) != type(
            typed_slot_value
        ):
            # the slot value has been coerced to the correct type
            return [SlotSet(slot_name, typed_slot_value)]
        elif slot_value == typed_slot_value:
            # the slot value has not changed and no utterance present
            return []

        if utterance == UTTERANCE_NOT_DEFINED_BY_USER:
            structlogger.error(
                "run.rejection.missing.utter",
                utterance=None,
            )
            return events

        message = await nlg.generate(
            utterance,
            tracker,
            output_channel,
            value=slot_value,
        )

        if message is None:
            structlogger.error(
                "run.rejection.failed.finding.utter",
                utterance=utterance,
            )
        else:
            message = add_bot_utterance_metadata(
                message, utterance, nlg, domain, tracker
            )
            events.append(create_bot_utterance(message))

        return events
