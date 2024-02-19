from typing import Any, Dict, TYPE_CHECKING, Text, List
import re

from rasa.shared.core.slots import CategoricalSlot

if TYPE_CHECKING:
    from rasa.shared.core.trackers import DialogueStateTracker


def _get_substituted_predicate_using_regex(
    predicate: Text, slot_instance: CategoricalSlot
) -> Text:
    """Replace a pattern in a predicate with a value."""
    for value in slot_instance.values:
        # we need to escape the value to make sure that it is not interpreted
        # as a regular expression pattern and that we only replace whole words
        # (e.g. we don't want to replace "foo" in "foobar") and we need to use
        # lookbehind and lookahead assertions to make sure that we only
        # replace the value if it is surrounded by quotes.
        pattern = r"(?<=[\"'])" + re.escape(value) + r"(?=[\"'])"
        predicate = re.sub(pattern, value, predicate, flags=re.IGNORECASE)
    return predicate


def get_case_insensitive_predicate(
    predicate: Text, slots: List[str], tracker: "DialogueStateTracker"
) -> Text:
    """Replace categorical slot values in a predicate with lower case replacement."""
    for slot in slots:
        slot_instance = tracker.slots.get(slot)
        if slot_instance and isinstance(slot_instance, CategoricalSlot):
            predicate = _get_substituted_predicate_using_regex(predicate, slot_instance)

    return predicate


def get_case_insensitive_predicate_given_slot_instance(
    predicate: Text, slots: Dict[str, Any]
) -> Text:
    """Replace categorical slot values in a predicate with lower case replacement."""
    for slot_name, slot_instance in slots.items():
        if (
            isinstance(slot_instance, CategoricalSlot)
            and f"slots.{slot_name}" in predicate
        ):
            predicate = _get_substituted_predicate_using_regex(predicate, slot_instance)

    return predicate
